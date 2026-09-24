"""统一模型接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ChatMessage:
    role: str          # system / user / assistant / tool
    content: str


@dataclass
class ChatResponse:
    """统一响应。屏蔽各家格式差异，统一采集可观测指标。"""
    content: str
    tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)  # 保留原始响应，供下钻


class ModelAdapter(ABC):
    """所有模型适配器的基类。"""

    def __init__(self, name: str, model_version: str = ""):
        self.name = name                    # 评测系统内的唯一标识
        self.model_version = model_version  # 真实模型版本（写入结果，保证可复现）

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> ChatResponse:
        """给定对话历史，返回一次补全。Agent 运行器每一步都会调用它。"""
        ...

    def cache_key_fields(self) -> dict:
        """参与缓存 key 的字段（模型版本变了，缓存自动失效）。"""
        return {"name": self.name, "version": self.model_version}
