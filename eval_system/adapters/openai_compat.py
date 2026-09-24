"""
OpenAI 兼容适配器（可选，需真实 API Key）。

覆盖 OpenAI / vLLM / 众多国产模型（多数提供 OpenAI 兼容端点）。
未配置 key 时不影响系统运行 —— 骨架默认用 mock 跑通。
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .base import ModelAdapter, ChatMessage, ChatResponse


class OpenAICompatAdapter(ModelAdapter):
    def __init__(
        self,
        name: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        price_per_1k: float = 0.0,
    ):
        super().__init__(name=name, model_version=model)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.price_per_1k = price_per_1k

    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> ChatResponse:
        if not self.api_key:
            raise RuntimeError(
                f"模型 {self.name} 未配置 API Key，请设置对应环境变量后重试。"
            )
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        latency = (time.perf_counter() - t0) * 1000
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        cost = tokens / 1000.0 * self.price_per_1k
        return ChatResponse(
            content=content, tokens=tokens, latency_ms=latency, cost=cost, raw=data
        )
