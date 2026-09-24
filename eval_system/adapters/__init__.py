"""
模型适配层。

对上层暴露统一的 ModelAdapter.chat() 接口，屏蔽各家 API 差异，
统一采集 token / 延迟。新增一个 provider 只需实现一个适配器并注册。
"""
from .base import ModelAdapter, ChatMessage, ChatResponse
from .registry import (
    register_model, get_model, list_models, register_defaults, load_from_store,
)

__all__ = [
    "ModelAdapter", "ChatMessage", "ChatResponse",
    "register_model", "get_model", "list_models", "register_defaults",
    "load_from_store",
]
