"""
生成示例数据集（混合任务）。

- smoke：小而快，供 CI 分钟级门禁。
- regression：更全面，供离线全量回归 / 选型对比。

deterministic 用例带 expected_answer（走规则裁判）；
generative 用例带 rubric（走 LLM 裁判）。
"""
import config
from eval_system.models import TestCase, TaskType, new_id
from eval_system.store.sqlite_store import SQLiteStore


def build_cases() -> list[TestCase]:
    cases: list[TestCase] = []

    # ---- 确定性：算术（期望 Agent 调用 calculator）----
    arithmetic = [
        ("计算 12 + 30 * 2 的结果是多少？", "72"),
        ("请计算 100 - 45 + 8", "63"),
        ("算一下 6 * 7 等于几", "42"),
        ("(15 + 5) / 4 的值", "5.0"),
    ]
    for q, a in arithmetic:
        for ds in ("smoke", "regression"):
            cases.append(TestCase(
                id=new_id("case_"), dataset=ds, task_type=TaskType.DETERMINISTIC,
                input=q, expected_answer=a, expected_tools=["calculator"],
                tags=["arithmetic", "easy"],
            ))

    # ---- 确定性：知识库查询（期望调用 kv_lookup）----
    lookups = [
        ("请查询 [key:首都-中国] 的答案", "北京"),
        ("查一下 [key:首都-日本]", "东京"),
    ]
    for q, a in lookups:
        cases.append(TestCase(
            id=new_id("case_"), dataset="regression", task_type=TaskType.DETERMINISTIC,
            input=q, expected_answer=a, expected_tools=["kv_lookup"],
            tags=["lookup"],
        ))

    # ---- 生成式：开放问答（走 LLM 裁判）----
    generative = [
        ("简述如何设计一个模型评测系统的关键要点。",
         "答案应覆盖数据集、指标、裁判方式等要点，结构完整，有依据。"),
        ("解释什么是 ReAct 循环。",
         "答案应说明思考-行动-观察的循环结构，要点清晰。"),
    ]
    for q, rub in generative:
        for ds in ("smoke", "regression"):
            cases.append(TestCase(
                id=new_id("case_"), dataset=ds, task_type=TaskType.GENERATIVE,
                input=q, rubric=rub, tags=["generative"],
            ))

    return cases


def main():
    store = SQLiteStore(config.DB_PATH)
    cases = build_cases()
    store.add_cases(cases)
    print(f"已写入 {len(cases)} 条用例。")
    for d in store.list_datasets():
        print(f"  数据集 {d['name']}: {d['case_count']} 条")
    print(f"数据库: {config.DB_PATH}")
    print("下一步: python run_web.py  或  python run_ci.py --dataset smoke")


if __name__ == "__main__":
    main()
