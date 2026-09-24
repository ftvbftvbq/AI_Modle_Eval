"""
结果聚合：把逐条结果汇总成可对比、可下钻的报告数据。

不只看平均分（呼应设计原则）：同时给通过率、分位数、按 task_type 分组，
并保留失败用例列表供下钻。
"""
from __future__ import annotations

from statistics import mean, median
from typing import Any

from ..store.base import Store


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return round(s[idx], 3)


def aggregate_run(store: Store, run_id: str) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"run {run_id} 不存在")
    results = store.get_results(run_id)

    # 按模型分组
    by_model: dict[str, list] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    model_summaries = []
    for model, rs in by_model.items():
        scores = [r.score for r in rs]
        passed = sum(1 for r in rs if r.passed)
        # 按任务类型细分
        by_type: dict[str, list[float]] = {}
        for r in rs:
            by_type.setdefault(r.task_type, []).append(r.score)

        model_summaries.append({
            "model": model,
            "count": len(rs),
            "avg_score": round(mean(scores), 3) if scores else 0.0,
            "median_score": round(median(scores), 3) if scores else 0.0,
            "p10": _percentile(scores, 10),
            "p90": _percentile(scores, 90),
            "pass_rate": round(passed / len(rs), 3) if rs else 0.0,
            "by_task_type": {
                t: round(mean(v), 3) for t, v in by_type.items()
            },
            "error_count": sum(1 for r in rs if r.error),
        })

    # 按平均分排序（选型对比）
    model_summaries.sort(key=lambda m: m["avg_score"], reverse=True)

    # 失败用例（供下钻）
    failures = [
        {
            "model": r.model, "case_id": r.case_id, "task_type": r.task_type,
            "score": r.score, "reason": r.reason,
            "stopped_reason": r.trajectory.get("stopped_reason"),
        }
        for r in results if not r.passed
    ]

    return {
        "run_id": run_id,
        "dataset": run.dataset,
        "status": run.status,
        "models": run.models,
        "created_at": run.created_at,
        "model_summaries": model_summaries,
        "total_results": len(results),
        "failures": failures,
    }
