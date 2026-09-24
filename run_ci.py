"""
CI 回归门禁入口。

对指定数据集跑指定模型的评测，若任一模型平均分低于阈值则以非零退出码退出，
可直接接入 CI 流水线阻断上线。

用法:
  python run_ci.py --dataset smoke --models mock-smart,mock-fast --threshold 0.7
"""
import argparse
import asyncio
import sys

import config
from eval_system.adapters.registry import load_from_store
from eval_system.orchestrator.engine import Orchestrator
from eval_system.orchestrator.aggregate import aggregate_run
from eval_system.settings import SettingsService
from eval_system.store.sqlite_store import SQLiteStore


async def _run(args) -> int:
    store = SQLiteStore(config.DB_PATH)
    settings = SettingsService(store)
    load_from_store(store)  # 加载 DB 中配置的模型 + 默认 mock
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # 阈值：命令行显式给了就用命令行，否则用设置页/DB 的值
    threshold = args.threshold if args.threshold is not None else settings.threshold()

    orch = Orchestrator(
        store=store, judge_model=settings.judge_model(),
        max_concurrency=settings.max_concurrency(), max_retries=settings.max_retries(),
        max_agent_steps=settings.max_agent_steps(), enable_cache=settings.enable_cache(),
    )

    def progress(done, total):
        print(f"\r进度 {done}/{total}", end="", flush=True)

    run_id = await orch.run_eval(args.dataset, models, progress_cb=progress)
    print()
    report = aggregate_run(store, run_id)

    print(f"\n=== CI 门禁报告 run={run_id} 数据集={args.dataset} 阈值={threshold} ===")
    failed_models = []
    for m in report["model_summaries"]:
        ok = m["avg_score"] >= threshold
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {m['model']:<14} 平均分={m['avg_score']:.3f} "
              f"通过率={m['pass_rate']:.0%} 错误={m['error_count']}")
        if not ok:
            failed_models.append(m["model"])

    if failed_models:
        print(f"\n门禁未通过：{', '.join(failed_models)} 低于阈值 {threshold}")
        return 1
    print("\n门禁通过 ✅")
    return 0


def main():
    p = argparse.ArgumentParser(description="Agent 评测 CI 门禁")
    p.add_argument("--dataset", default="smoke")
    p.add_argument("--models", default="mock-smart,mock-fast")
    # 默认 None：表示未显式指定，运行时恢复到设置页/DB 的阈值
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
