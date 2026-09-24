"""评分器抽象。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import TestCase, Trajectory, Score


class Scorer(ABC):
    name: str = "base"

    @abstractmethod
    async def score(self, case: TestCase, trajectory: Trajectory) -> Score:
        """对一条用例的 Agent 轨迹打分，返回归一化 0~1 的分数。"""
        ...


def trajectory_metrics(case: TestCase, traj: Trajectory) -> dict[str, float]:
    """
    轨迹级通用维度（对两类任务都适用），作为分维度 breakdown。
    这些不直接决定通过与否，但用于下钻分析 Agent 行为质量。
    """
    metrics: dict[str, float] = {}

    # 工具调用正确率：期望用到的工具是否都用了
    if case.expected_tools:
        used = set(traj.tools_used)
        hit = len(used & set(case.expected_tools))
        metrics["tool_correctness"] = round(hit / len(case.expected_tools), 3)

    # 步数效率：越少步完成越好（完成时才有意义）
    if traj.stopped_reason == "completed" and traj.total_steps > 0:
        metrics["step_efficiency"] = round(1.0 / traj.total_steps, 3)

    # 是否正常完成（未触发预算耗尽/循环/错误）
    metrics["completed"] = 1.0 if traj.stopped_reason == "completed" else 0.0
    return metrics
