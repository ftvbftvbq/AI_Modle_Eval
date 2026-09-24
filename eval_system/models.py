"""
核心数据模型。

设计要点：
- 用例(TestCase) 带 task_type，驱动评分层选择规则裁判还是 LLM 裁判。
- Agent 输出是"轨迹"(Trajectory) 而非单条文本 —— 评测带工具的 Agent 必须评多步行为。
- Result 保留完整原始输出与裁判理由，保证可复现、可下钻。
- 一切带 metadata 字段，便于扩展而不改表结构。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional
import time
import uuid


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def now_ts() -> float:
    return time.time()


class TaskType(str, Enum):
    """任务类型决定评分方式。"""
    DETERMINISTIC = "deterministic"  # 有标准答案 -> 规则裁判
    GENERATIVE = "generative"        # 开放式生成 -> LLM 裁判


# ---------------- 数据集与用例 ----------------

@dataclass
class TestCase:
    """一条评测用例。"""
    id: str
    dataset: str                      # 所属数据集名（如 smoke / regression）
    task_type: TaskType
    input: str                        # 给 Agent 的任务描述 / 用户输入
    # 期望：deterministic 用 expected_answer；generative 用 rubric（评分标准）
    expected_answer: Optional[str] = None
    rubric: Optional[str] = None
    # 期望 Agent 用到的工具（用于评工具调用正确率，可空）
    expected_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)  # 难度/领域等
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict:
        d = asdict(self)
        d["task_type"] = self.task_type.value
        return d


# ---------------- Agent 轨迹 ----------------

@dataclass
class ToolCall:
    """Agent 的一次工具调用记录。"""
    tool: str
    args: dict[str, Any]
    result: Any
    error: Optional[str] = None


@dataclass
class TrajectoryStep:
    """Agent 单步：思考 + 可选工具调用。"""
    step: int
    thought: str
    tool_call: Optional[ToolCall] = None


@dataclass
class Trajectory:
    """Agent 完成一条用例的完整多步轨迹。"""
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_answer: str = ""
    stopped_reason: str = ""          # completed / budget_exhausted / loop_detected / error
    total_steps: int = 0
    tools_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "steps": [
                {
                    "step": s.step,
                    "thought": s.thought,
                    "tool_call": (
                        {
                            "tool": s.tool_call.tool,
                            "args": s.tool_call.args,
                            "result": s.tool_call.result,
                            "error": s.tool_call.error,
                        }
                        if s.tool_call else None
                    ),
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "stopped_reason": self.stopped_reason,
            "total_steps": self.total_steps,
            "tools_used": self.tools_used,
        }


# ---------------- 评分结果 ----------------

@dataclass
class Score:
    """单条用例在单个模型上的评分结果。"""
    value: float                      # 归一化 0~1
    passed: bool
    scorer: str                       # rule / llm_judge
    reason: str = ""                  # 裁判理由（LLM 裁判尤其重要，供下钻）
    breakdown: dict[str, float] = field(default_factory=dict)  # 分维度得分


@dataclass
class CaseResult:
    """一条用例 × 一个模型 的完整结果，保留原始轨迹与评分。"""
    id: str
    run_id: str
    case_id: str
    model: str
    dataset: str
    task_type: str
    trajectory: dict                  # Trajectory.to_dict()
    score: float
    passed: bool
    scorer: str
    reason: str
    latency_ms: float
    tokens: int
    cost: float
    error: Optional[str] = None
    created_at: float = field(default_factory=now_ts)


@dataclass
class Run:
    """一次评测运行（可覆盖多模型 × 一个数据集）。"""
    id: str
    dataset: str
    models: list[str]
    status: str = "running"           # running / completed / failed
    created_at: float = field(default_factory=now_ts)
    finished_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
