"""
规则裁判：用于 deterministic 任务（有标准答案）。
快、便宜、稳定、完全可复现。
"""
from __future__ import annotations

import re

from ..models import TestCase, Trajectory, Score
from .base import Scorer, trajectory_metrics


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip().lower())


class RuleScorer(Scorer):
    name = "rule"

    def __init__(self, pass_threshold: float = 1.0):
        # deterministic 任务默认要求精确匹配才算过
        self.pass_threshold = pass_threshold

    async def score(self, case: TestCase, trajectory: Trajectory) -> Score:
        expected = _normalize(case.expected_answer or "")
        actual = _normalize(trajectory.final_answer)

        breakdown = trajectory_metrics(case, trajectory)

        if not expected:
            # 没配标准答案却走了规则裁判：判为不可评分
            return Score(value=0.0, passed=False, scorer=self.name,
                         reason="deterministic 用例缺少 expected_answer",
                         breakdown=breakdown)

        # 精确匹配 or 期望答案是实际答案的子串（宽松一档）
        exact = expected == actual
        contains = expected in actual and len(expected) >= 1
        value = 1.0 if exact else (0.6 if contains else 0.0)

        # 轨迹异常（循环/预算耗尽）时对最终分做惩罚
        if trajectory.stopped_reason in ("loop_detected", "budget_exhausted", "error"):
            value = min(value, 0.3)

        passed = value >= self.pass_threshold
        reason = (f"期望={case.expected_answer!r} 实际={trajectory.final_answer!r} "
                  f"匹配={'精确' if exact else ('包含' if contains else '否')} "
                  f"轨迹={trajectory.stopped_reason}")
        return Score(value=round(value, 3), passed=passed, scorer=self.name,
                     reason=reason, breakdown=breakdown)
