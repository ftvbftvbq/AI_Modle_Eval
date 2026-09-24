"""
LLM 裁判：用于 generative 任务（开放式生成，无唯一标准答案）。

工程要点（呼应思考题第 5 点的裁判偏差问题）：
- 给裁判明确的 rubric，要求输出结构化 {score, reason}。
- 用独立的裁判模型（JUDGE_MODEL），避免被评模型自评（self-preference bias）。
- 骨架用 mock-judge；生产可换成强模型，并定期用黄金人工标注集校准一致率。
"""
from __future__ import annotations

import json
import re

from ..adapters.base import ModelAdapter, ChatMessage
from ..models import TestCase, Trajectory, Score
from .base import Scorer, trajectory_metrics

JUDGE_PROMPT = """你是严格的评测裁判。请根据评分标准给答案打分（0~1 的小数）。

任务：{task}

评分标准(rubric)：
{rubric}

被评答案：
<answer>{answer}</answer>

只输出 JSON：{{"score": 0.x, "reason": "简要理由"}}"""


class LLMJudgeScorer(Scorer):
    name = "llm_judge"

    def __init__(self, judge_model: ModelAdapter, pass_threshold: float = 0.6):
        self.judge = judge_model
        self.pass_threshold = pass_threshold

    async def score(self, case: TestCase, trajectory: Trajectory) -> Score:
        breakdown = trajectory_metrics(case, trajectory)
        rubric = case.rubric or "答案是否准确、完整、相关。"
        prompt = JUDGE_PROMPT.format(
            task=case.input, rubric=rubric, answer=trajectory.final_answer
        )
        resp = await self.judge.chat([ChatMessage("user", prompt)], temperature=0.0)

        parsed = self._parse(resp.content)
        value = parsed.get("score", 0.0)
        judge_reason = parsed.get("reason", "")

        # 轨迹异常惩罚
        if trajectory.stopped_reason in ("loop_detected", "budget_exhausted", "error"):
            value = min(value, 0.3)

        passed = value >= self.pass_threshold
        reason = f"[裁判={self.judge.name}] {judge_reason} 轨迹={trajectory.stopped_reason}"
        return Score(value=round(float(value), 3), passed=passed,
                     scorer=self.name, reason=reason, breakdown=breakdown)

    @staticmethod
    def _parse(text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        return {"score": 0.0, "reason": "裁判输出无法解析"}
