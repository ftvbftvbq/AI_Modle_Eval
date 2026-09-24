"""
模型注册表。

系统按 name 查找适配器实例。生产环境可改为从配置文件 / DB 加载 20+ 模型。
"""
from __future__ import annotations

from .base import ModelAdapter

_REGISTRY: dict[str, ModelAdapter] = {}


def register_model(adapter: ModelAdapter) -> None:
    _REGISTRY[adapter.name] = adapter


def get_model(name: str) -> ModelAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"未注册的模型: {name}. 已注册: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    return list(_REGISTRY)


def register_defaults() -> None:
    """注册默认的 mock 模型集合，保证零配置可跑。"""
    from .mock import MockAdapter, MockJudgeAdapter

    if _REGISTRY:
        return
    # 两个能力不同的假模型，用于选型对比（分数会拉开差距）
    register_model(MockAdapter("mock-smart", quality=0.9, latency_ms=30))
    register_model(MockAdapter("mock-fast", quality=0.5, latency_ms=10))
    # 裁判模型
    register_model(MockJudgeAdapter("mock-judge", quality=1.0, latency_ms=15))


def _adapter_from_config(cfg: dict):
    """根据 DB 里的 model_configs 记录构造适配器实例。"""
    from .mock import MockAdapter, MockJudgeAdapter
    from .openai_compat import OpenAICompatAdapter

    kind = cfg.get("kind", "mock")
    name = cfg["name"]
    if kind == "openai":
        return OpenAICompatAdapter(
            name=name,
            model=cfg.get("model") or name,
            base_url=cfg.get("base_url") or "https://api.openai.com/v1",
            api_key_env=cfg.get("api_key_env") or "OPENAI_API_KEY",
            price_per_1k=float(cfg.get("price_per_1k", 0) or 0),
        )
    # mock 类：裁判用 MockJudgeAdapter
    if cfg.get("is_judge"):
        return MockJudgeAdapter(name, quality=float(cfg.get("quality", 1.0) or 1.0))
    return MockAdapter(name, quality=float(cfg.get("quality", 0.8) or 0.8))


def load_from_store(store) -> None:
    """
    从数据库的 model_configs 表加载并注册模型（覆盖同名）。
    先保证 mock 默认模型在册，再叠加 DB 中用户添加的模型。
    """
    register_defaults()
    for cfg in store.list_model_configs():
        try:
            register_model(_adapter_from_config(cfg))
        except Exception:
            # 单个模型配置有误不应阻断整体
            continue
