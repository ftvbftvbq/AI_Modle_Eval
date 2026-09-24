"""
评分路由：按用例 task_type 选择裁判方式（混合任务的关键）。
- deterministic -> 规则裁判
- generative    -> LLM 裁判
"""
from __future__ import annotations

from ..adapters.registry import get_model
from ..models import TestCase, TaskType, Trajectory, Score
from .rule_scorer import RuleScorer
from .llm_judge import LLMJudgeScorer


async def score_case(case: TestCase, trajectory: Trajectory, judge_model_name: str) -> Score:
    if case.task_type == TaskType.DETERMINISTIC:
        return await RuleScorer().score(case, trajectory)
    else:
        judge = get_model(judge_model_name)
        return await LLMJudgeScorer(judge).score(case, trajectory)
