"""
配置服务：运行期可调配置的统一入口。

原则：DB 有值就用 DB 的（Web 可改，不用改代码/重启），没有则恢复到 config.py 的原始默认值。
这样 config.py 只承担"原始默认值"角色，真正的运行配置存在数据库里、由设置页管理。
"""
from __future__ import annotations

import config
from .store.base import Store

# 可调项的键名 + 默认值来源（config.py）+ 类型转换
# key: (config默认值, 类型)
SETTING_SPEC = {
    "ci_pass_threshold": (config.CI_PASS_THRESHOLD, float),
    "judge_model": (config.JUDGE_MODEL, str),
    "max_concurrency": (config.MAX_CONCURRENCY, int),
    "max_retries": (config.MAX_RETRIES, int),
    "max_agent_steps": (config.MAX_AGENT_STEPS, int),
    "enable_cache": (1 if config.ENABLE_CACHE else 0, int),
}

# 每项的中文标签与说明，供设置页展示
SETTING_LABELS = {
    "ci_pass_threshold": ("CI 达标阈值", "模型平均分 ≥ 此值判为达标（0~1）"),
    "judge_model": ("裁判模型", "用于给生成式任务打分的模型名"),
    "max_concurrency": ("最大并发", "每个模型的最大并发请求数"),
    "max_retries": ("最大重试", "瞬时错误的最大重试次数"),
    "max_agent_steps": ("Agent 步数预算", "单条用例的最大工具调用轮数"),
    "enable_cache": ("启用缓存", "1 启用 / 0 关闭；命中缓存跳过重复执行"),
}


class SettingsService:
    def __init__(self, store: Store):
        self.store = store

    def _get(self, key: str):
        default, caster = SETTING_SPEC[key]
        raw = self.store.get_setting(key)
        if raw is None:
            return default
        try:
            return caster(raw)
        except (ValueError, TypeError):
            return default

    # 便捷取值
    def threshold(self) -> float:
        return float(self._get("ci_pass_threshold"))

    def judge_model(self) -> str:
        return str(self._get("judge_model"))

    def max_concurrency(self) -> int:
        return int(self._get("max_concurrency"))

    def max_retries(self) -> int:
        return int(self._get("max_retries"))

    def max_agent_steps(self) -> int:
        return int(self._get("max_agent_steps"))

    def enable_cache(self) -> bool:
        return bool(int(self._get("enable_cache")))

    def all_effective(self) -> dict:
        """返回所有可调项的当前生效值（DB 覆盖后的），供设置页回显。"""
        return {k: self._get(k) for k in SETTING_SPEC}

    def save(self, updates: dict) -> None:
        """保存一批设置（只接受已知键）。"""
        for k, v in updates.items():
            if k in SETTING_SPEC:
                self.store.set_setting(k, str(v))
