"""
Mock 适配器：无需任何 API Key 即可端到端跑通评测。

它不是随机吐字，而是一个"能理解任务并调用工具"的确定性假模型：
- 识别用例里的计算 / 查询意图，输出遵循 Agent 运行器约定的 JSON 动作（调用工具或给最终答案）。
- 通过 quality 参数模拟不同能力的模型（mock-smart vs mock-fast），
  让选型对比能产生有区分度的分数，便于验证整条链路。
- mock-judge 用于 LLM 裁判，同样确定性。

Agent 运行器与模型的协议（见 agent/runner.py）：模型每步返回一段 JSON：
  {"thought": "...", "action": "tool", "tool": "calculator", "args": {"expression": "2+2"}}
或
  {"thought": "...", "action": "final", "answer": "4"}
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from .base import ModelAdapter, ChatMessage, ChatResponse


class MockAdapter(ModelAdapter):
    def __init__(self, name: str, quality: float = 1.0, latency_ms: float = 20.0):
        super().__init__(name=name, model_version=f"{name}-v1")
        # quality: 0~1，越高越"聪明"（越会正确调用工具、答案越准）。
        self.quality = quality
        self._latency = latency_ms

    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> ChatResponse:
        t0 = time.perf_counter()
        await asyncio.sleep(self._latency / 1000.0)  # 模拟网络延迟

        # 取最后一条 user/tool 消息作为当前上下文
        last = messages[-1].content if messages else ""
        # 已经拿到工具结果？（上一步是 tool 消息）
        prev_tool_result = None
        if messages and messages[-1].role == "tool":
            prev_tool_result = messages[-1].content

        reply = self._decide(last, prev_tool_result, messages)
        content = json.dumps(reply, ensure_ascii=False)

        latency = (time.perf_counter() - t0) * 1000
        tokens = max(1, len(content) // 4)
        return ChatResponse(
            content=content,
            tokens=tokens,
            latency_ms=latency,
            cost=tokens * 1e-6,   # 假成本
            raw={"mock": True, "quality": self.quality},
        )

    # ---- 决策逻辑（确定性）----
    def _decide(self, text: str, prev_tool_result: str | None, messages) -> dict:
        # 找到原始任务（第一条 user 消息）
        task = ""
        for m in messages:
            if m.role == "user":
                task = m.content
                break

        # 若已有工具结果，直接据此给最终答案
        if prev_tool_result is not None:
            return {"thought": "已获得工具结果，据此作答。",
                    "action": "final", "answer": prev_tool_result.strip()}

        # 识别算术任务 -> 调 calculator 工具
        expr = self._extract_arithmetic(task)
        if expr is not None:
            # 低质量模型有概率"忘记"用工具，直接口算（可能错），制造区分度
            if self.quality < 0.6:
                wrong = self._bad_mental_math(expr)
                return {"thought": "直接口算。", "action": "final", "answer": wrong}
            return {"thought": f"这是算术题，调用计算器：{expr}",
                    "action": "tool", "tool": "calculator",
                    "args": {"expression": expr}}

        # 识别查询任务 -> 调 kv_lookup 工具
        key = self._extract_lookup_key(task)
        if key is not None:
            if self.quality < 0.6:
                return {"thought": "凭记忆回答。", "action": "final",
                        "answer": "不确定"}
            return {"thought": f"需要查询 {key}，调用 kv_lookup。",
                    "action": "tool", "tool": "kv_lookup", "args": {"key": key}}

        # 生成式任务：直接生成答案，质量影响详尽程度
        return {"thought": "这是开放式问题，直接作答。",
                "action": "final", "answer": self._generative_answer(task)}

    @staticmethod
    def _extract_arithmetic(text: str) -> str | None:
        m = re.search(r"(\d+(?:\s*[\+\-\*/]\s*\d+)+)", text)
        return m.group(1).replace(" ", "") if m else None

    @staticmethod
    def _bad_mental_math(expr: str) -> str:
        try:
            return str(eval(expr) + 1)  # 故意错 1，模拟口算失误
        except Exception:
            return "?"

    @staticmethod
    def _extract_lookup_key(text: str) -> str | None:
        # 约定：任务里出现 [key:XXX] 表示要查 XXX
        m = re.search(r"\[key:([^\]]+)\]", text)
        return m.group(1) if m else None

    def _generative_answer(self, task: str) -> str:
        base = f"针对「{task[:40]}」的回答"
        if self.quality >= 0.8:
            return base + "：要点一、要点二、要点三，结构完整且有依据。"
        elif self.quality >= 0.6:
            return base + "：给出了基本要点。"
        return base + "：简略。"


class MockJudgeAdapter(MockAdapter):
    """LLM 裁判用的 mock 模型：读到 rubric+答案，输出结构化打分。"""

    async def chat(self, messages, temperature=0.0, max_tokens=1024, **kwargs):
        t0 = time.perf_counter()
        await asyncio.sleep(self._latency / 1000.0)
        prompt = messages[-1].content if messages else ""
        # 极简启发式：答案越长、包含"要点/结构/依据"越多分（模拟裁判偏好）。
        answer = ""
        m = re.search(r"<answer>(.*?)</answer>", prompt, re.S)
        if m:
            answer = m.group(1)
        score = 0.4
        for kw in ["要点", "结构", "依据", "完整"]:
            if kw in answer:
                score += 0.15
        score = min(1.0, round(score, 2))
        reason = f"答案长度 {len(answer)}，命中质量关键词，判定 {score}。"
        content = json.dumps({"score": score, "reason": reason}, ensure_ascii=False)
        latency = (time.perf_counter() - t0) * 1000
        return ChatResponse(content=content, tokens=len(content) // 4,
                            latency_ms=latency, cost=0.0, raw={"judge": True})
