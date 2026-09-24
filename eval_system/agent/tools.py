"""
工具注册表与内置工具。

被评测的 Agent 可调用这些工具。评测系统据此评"工具调用正确率"。
新增工具只需 @registry.register 装饰，并写清描述（描述会进模型上下文）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]

    def run(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str):
        def deco(func: Callable[..., Any]):
            self._tools[name] = Tool(name=name, description=description, func=func)
            return func
        return deco

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        """生成给模型看的工具清单（进 system prompt）。"""
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())


def default_tools() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register("calculator", "计算算术表达式，参数 {expression: str}，返回数值字符串。")
    def calculator(expression: str) -> str:
        # 受限求值：只允许数字与四则运算符，避免任意代码执行。
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            raise ValueError("表达式含非法字符")
        return str(eval(expression))  # 已白名单字符，安全

    @reg.register("kv_lookup", "查询内置知识库，参数 {key: str}，返回对应值。")
    def kv_lookup(key: str) -> str:
        kb = {
            "首都-中国": "北京",
            "首都-日本": "东京",
            "光速": "约 299792458 米/秒",
        }
        return kb.get(key, "未找到")

    return reg
