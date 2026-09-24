"""
存储层抽象接口。

把存储与其余逻辑解耦：骨架用 SQLite 实现，生产切 Postgres 时
只需新增一个实现类，上层（编排、Web）代码不变。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..models import TestCase, CaseResult, Run


class Store(ABC):
    # ---- 数据集 / 用例 ----
    @abstractmethod
    def init_schema(self) -> None: ...

    @abstractmethod
    def add_cases(self, cases: list[TestCase]) -> None: ...

    @abstractmethod
    def get_cases(self, dataset: str) -> list[TestCase]: ...

    @abstractmethod
    def delete_case(self, case_id: str) -> bool:
        """删除一条用例，返回是否删到。"""
        ...

    @abstractmethod
    def list_datasets(self) -> list[dict]:
        """返回 [{name, case_count}]。"""
        ...

    # ---- 运行 ----
    @abstractmethod
    def create_run(self, run: Run) -> None: ...

    @abstractmethod
    def finish_run(self, run_id: str, status: str) -> None: ...

    @abstractmethod
    def get_run(self, run_id: str) -> Optional[Run]: ...

    @abstractmethod
    def list_runs(self, limit: int = 50) -> list[Run]: ...

    # ---- 结果 ----
    @abstractmethod
    def add_result(self, result: CaseResult) -> None: ...

    @abstractmethod
    def get_results(self, run_id: str) -> list[CaseResult]: ...

    # ---- 缓存（key=模型+prompt+参数 的哈希）----
    @abstractmethod
    def cache_get(self, key: str) -> Optional[dict]: ...

    @abstractmethod
    def cache_set(self, key: str, value: dict) -> None: ...

    # ---- 设置（键值）----
    @abstractmethod
    def get_setting(self, key: str) -> Optional[str]: ...

    @abstractmethod
    def get_all_settings(self) -> dict: ...

    @abstractmethod
    def set_setting(self, key: str, value: str) -> None: ...

    # ---- 模型接入配置 ----
    @abstractmethod
    def list_model_configs(self) -> list[dict]: ...

    @abstractmethod
    def add_model_config(self, cfg: dict) -> None: ...

    @abstractmethod
    def delete_model_config(self, name: str) -> bool: ...
