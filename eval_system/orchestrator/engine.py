"""
执行编排层：把"跑评测"工程化。

职责：
- 并发：对 (模型 × 用例) 笛卡尔积并发执行，每个模型独立限流（模拟各家 rate limit）。
- 重试：区分瞬时错误（可重试）与永久错误（不重试），带指数退避。
- 缓存：(模型版本 + 任务 + Agent 配置) 命中缓存则跳过执行，省钱、支持断点续跑。
- 落库：逐条结果写入 Store，跑一半中断也不丢已完成部分。

规模化路径：本层是进程内 asyncio；生产可把每个 (model, case) 任务投递到
Redis + Celery/Arq 队列，由 worker 池消费，Store 换 Postgres，即可横向扩展到
千万级用例、20+ 模型、分钟级调度。上层接口不变。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Callable, Optional

from ..adapters.registry import get_model
from ..agent.runner import AgentRunner
from ..agent.tools import default_tools
from ..models import TestCase, CaseResult, Run, new_id
from ..scoring.router import score_case
from ..store.base import Store


# 永久错误关键字：命中则不重试（如认证、参数错误、内容拒绝）
_PERMANENT_MARKERS = ("api key", "unauthorized", "invalid", "400", "401", "403", "not found")


def _is_retryable(err: Exception) -> bool:
    msg = str(err).lower()
    if any(m in msg for m in _PERMANENT_MARKERS):
        return False
    return True  # 超时、429、5xx、网络抖动等默认可重试


def _cache_key(model_name: str, model_version: str, case: TestCase, max_steps: int) -> str:
    payload = {
        "model": model_name, "version": model_version,
        "case": case.id, "input": case.input, "max_steps": max_steps,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class Orchestrator:
    def __init__(
        self,
        store: Store,
        judge_model: str,
        max_concurrency: int = 8,
        max_retries: int = 2,
        max_agent_steps: int = 6,
        enable_cache: bool = True,
    ):
        self.store = store
        self.judge_model = judge_model
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.enable_cache = enable_cache
        self.runner = AgentRunner(tools=default_tools(), max_steps=max_agent_steps)
        self.max_agent_steps = max_agent_steps

    async def run_eval(
        self,
        dataset: str,
        models: list[str],
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """对一个数据集跑指定模型集合，返回 run_id。"""
        cases = self.store.get_cases(dataset)
        if not cases:
            raise ValueError(f"数据集 {dataset} 无用例")

        run = Run(id=new_id("run_"), dataset=dataset, models=models)
        self.store.create_run(run)

        # 每个模型一个信号量，隔离限流
        sems = {m: asyncio.Semaphore(self.max_concurrency) for m in models}
        total = len(cases) * len(models)
        done = 0
        done_lock = asyncio.Lock()

        async def one(model_name: str, case: TestCase):
            nonlocal done
            async with sems[model_name]:
                result = await self._eval_one(run.id, model_name, case)
            self.store.add_result(result)
            async with done_lock:
                done += 1
                if progress_cb:
                    progress_cb(done, total)

        tasks = [one(m, c) for m in models for c in cases]
        try:
            await asyncio.gather(*tasks)
            self.store.finish_run(run.id, "completed")
        except Exception:
            self.store.finish_run(run.id, "failed")
            raise
        return run.id

    async def _eval_one(self, run_id: str, model_name: str, case: TestCase) -> CaseResult:
        model = get_model(model_name)
        ck = _cache_key(model_name, model.model_version, case, self.max_agent_steps)

        # 缓存命中：直接复用（断点续跑 / 重跑省钱）
        if self.enable_cache:
            cached = self.store.cache_get(ck)
            if cached:
                cached["id"] = new_id("res_")
                cached["run_id"] = run_id
                return CaseResult(**cached)

        # 带重试的执行
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                traj = await self.runner.run(model, case.input)
                score = await score_case(case, traj, self.judge_model)

                result = CaseResult(
                    id=new_id("res_"), run_id=run_id, case_id=case.id,
                    model=model_name, dataset=case.dataset,
                    task_type=case.task_type.value,
                    trajectory=traj.to_dict(),
                    score=score.value, passed=score.passed, scorer=score.scorer,
                    reason=score.reason,
                    latency_ms=float(traj.total_steps),  # 简化：以步数近似
                    tokens=0, cost=0.0, error=None,
                )
                if self.enable_cache:
                    payload = result.__dict__.copy()
                    payload.pop("id"); payload.pop("run_id")
                    self.store.cache_set(ck, payload)
                return result
            except Exception as e:
                last_err = e
                if not _is_retryable(e) or attempt == self.max_retries:
                    break
                await asyncio.sleep(0.2 * (2 ** attempt))  # 指数退避

        # 彻底失败：记录为 error 结果（不中断整体评测）
        return CaseResult(
            id=new_id("res_"), run_id=run_id, case_id=case.id,
            model=model_name, dataset=case.dataset, task_type=case.task_type.value,
            trajectory={"steps": [], "final_answer": "", "stopped_reason": "error",
                        "total_steps": 0, "tools_used": []},
            score=0.0, passed=False, scorer="none",
            reason=f"执行失败: {last_err}", latency_ms=0.0, tokens=0, cost=0.0,
            error=str(last_err),
        )
