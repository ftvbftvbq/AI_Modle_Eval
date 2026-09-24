"""
带工具的 Agent 运行器（ReAct 式多步循环）。

流程：把任务 + 工具清单给模型 -> 模型输出 JSON 动作 ->
若是 tool 则执行工具、把结果回灌 -> 若是 final 则结束。
产出完整 Trajectory（供轨迹级评分与下钻）。

内置两道护栏（呼应思考题 4）：
- 迭代预算 MAX_AGENT_STEPS：防止无限循环耗尽资源。
- 重复动作检测：同一 (tool, args) 连续出现即判定死循环并终止，
  而不是傻等到预算耗尽。
"""
from __future__ import annotations

import json
import re

from ..adapters.base import ModelAdapter, ChatMessage
from ..models import Trajectory, TrajectoryStep, ToolCall
from .tools import ToolRegistry

SYSTEM_TEMPLATE = """你是一个会使用工具的智能体。可用工具：
{tools}

每一步只输出一个 JSON 对象，格式二选一：
1) 调用工具: {{"thought": "推理", "action": "tool", "tool": "工具名", "args": {{...}}}}
2) 给出最终答案: {{"thought": "推理", "action": "final", "answer": "最终答案"}}
不要输出 JSON 以外的内容。"""


def _extract_json(text: str) -> dict | None:
    """从模型输出里稳健地抽取第一个 JSON 对象。"""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


class AgentRunner:
    def __init__(self, tools: ToolRegistry, max_steps: int = 6):
        self.tools = tools
        self.max_steps = max_steps

    async def run(self, model: ModelAdapter, task: str) -> Trajectory:
        traj = Trajectory()
        messages: list[ChatMessage] = [
            ChatMessage("system", SYSTEM_TEMPLATE.format(tools=self.tools.describe())),
            ChatMessage("user", task),
        ]
        seen_actions: set[str] = set()

        for step in range(1, self.max_steps + 1):
            resp = await model.chat(messages, temperature=0.0)
            action = _extract_json(resp.content)

            if action is None:
                # 模型没给出合法动作：记录并终止（避免无意义重试）
                traj.steps.append(TrajectoryStep(step, thought="[无法解析模型输出]"))
                traj.final_answer = resp.content.strip()
                traj.stopped_reason = "error"
                break

            thought = str(action.get("thought", ""))

            if action.get("action") == "final":
                traj.steps.append(TrajectoryStep(step, thought=thought))
                traj.final_answer = str(action.get("answer", "")).strip()
                traj.stopped_reason = "completed"
                break

            if action.get("action") == "tool":
                tool_name = action.get("tool", "")
                args = action.get("args", {}) or {}

                # 重复动作检测：同一 (tool, args) 再次出现 -> 判定死循环
                fingerprint = f"{tool_name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                if fingerprint in seen_actions:
                    traj.steps.append(TrajectoryStep(
                        step, thought=thought,
                        tool_call=ToolCall(tool_name, args, result=None,
                                           error="重复调用，判定循环")))
                    traj.stopped_reason = "loop_detected"
                    break
                seen_actions.add(fingerprint)

                tool = self.tools.get(tool_name)
                if tool is None:
                    tc = ToolCall(tool_name, args, result=None,
                                  error=f"未知工具 {tool_name}")
                    traj.steps.append(TrajectoryStep(step, thought=thought, tool_call=tc))
                    # 把错误回灌，让模型有机会纠正
                    messages.append(ChatMessage("tool", f"错误：未知工具 {tool_name}"))
                    continue

                try:
                    result = tool.run(**args)
                    error = None
                except Exception as e:
                    result = None
                    error = str(e)

                tc = ToolCall(tool_name, args, result=result, error=error)
                traj.steps.append(TrajectoryStep(step, thought=thought, tool_call=tc))
                if tool_name not in traj.tools_used:
                    traj.tools_used.append(tool_name)

                # 工具结果回灌到对话（关键：没有反馈 Agent 会盲目重试）
                feedback = f"错误：{error}" if error else str(result)
                messages.append(ChatMessage("tool", feedback))
                continue

            # 未知 action 类型
            traj.steps.append(TrajectoryStep(step, thought=thought))
            traj.stopped_reason = "error"
            break
        else:
            # for 正常结束 = 用尽预算仍未 final
            traj.stopped_reason = "budget_exhausted"

        traj.total_steps = len(traj.steps)
        return traj
