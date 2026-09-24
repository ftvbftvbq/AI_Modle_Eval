"""
全局配置。所有可调项集中在此，便于本地 / 服务化环境切换。
生产环境建议用环境变量覆盖（os.environ）。
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
DATASET_DIR = DATA_DIR / "datasets"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)

# ---- 存储 ----
# 最小骨架用 SQLite；服务化后把 DB_URL 换成 postgresql://... 并替换 store 实现即可。
DB_PATH = os.environ.get("EVAL_DB_PATH", str(DATA_DIR / "eval.db"))

# ---- 执行编排 ----
# 每个模型的最大并发请求数（模拟各家 API 的 rate limit，实际应按 provider 分别配置）。
MAX_CONCURRENCY = int(os.environ.get("EVAL_MAX_CONCURRENCY", "8"))
# 瞬时错误最大重试次数
MAX_RETRIES = int(os.environ.get("EVAL_MAX_RETRIES", "2"))
# 是否启用结果缓存（key = 模型版本+prompt+参数），重跑时命中缓存直接返回，省钱。
ENABLE_CACHE = os.environ.get("EVAL_ENABLE_CACHE", "1") == "1"

# ---- Agent 运行器 ----
# 单个用例的最大工具调用轮数（迭代预算，防止死循环耗尽资源）。
MAX_AGENT_STEPS = int(os.environ.get("EVAL_MAX_AGENT_STEPS", "6"))

# ---- LLM 裁判 ----
# 用作裁判的模型 name（必须已在 adapters 注册）。避免用被评模型自评（self-preference bias）。
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "mock-judge")

# ---- CI 门禁默认阈值（0~1）----
CI_PASS_THRESHOLD = float(os.environ.get("EVAL_CI_THRESHOLD", "0.7"))
