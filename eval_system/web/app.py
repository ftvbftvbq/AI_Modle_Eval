"""
FastAPI 服务 + 轻量 Web 界面。

页面：
- 首页：数据集列表、历史运行、触发评测表单。
- 运行报告：模型对比矩阵（可下钻到失败用例）。

API（也可被 CI / 前端调用）：
- POST /api/runs        触发一次评测（后台异步跑）
- GET  /api/runs/{id}   运行报告 JSON
- GET  /api/models      已注册模型
- GET  /api/datasets    数据集
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import config
from ..adapters.registry import load_from_store, list_models
from ..models import TestCase, TaskType, new_id
from ..orchestrator.engine import Orchestrator
from ..orchestrator.aggregate import aggregate_run
from ..settings import SettingsService, SETTING_SPEC, SETTING_LABELS
from ..store.sqlite_store import SQLiteStore

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app() -> FastAPI:
    app = FastAPI(title="Agent 评测系统")
    store = SQLiteStore(config.DB_PATH)
    settings = SettingsService(store)
    # 从 DB 加载模型（含用户在设置页添加的），叠加默认 mock 模型
    load_from_store(store)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def make_orchestrator() -> Orchestrator:
        # 每次都按当前设置构建，保证设置页改动即时生效（无需重启）
        load_from_store(store)
        return Orchestrator(
            store=store,
            judge_model=settings.judge_model(),
            max_concurrency=settings.max_concurrency(),
            max_retries=settings.max_retries(),
            max_agent_steps=settings.max_agent_steps(),
            enable_cache=settings.enable_cache(),
        )

    # ---------------- 页面 ----------------
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        judge = settings.judge_model()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "datasets": store.list_datasets(),
            "models": [m for m in list_models() if m != judge],
            "runs": store.list_runs(20),
        })

    @app.get("/datasets/{name}", response_class=HTMLResponse)
    async def dataset_detail(request: Request, name: str):
        cases = store.get_cases(name)
        # 统计任务类型分布
        type_counts: dict[str, int] = {}
        for c in cases:
            type_counts[c.task_type.value] = type_counts.get(c.task_type.value, 0) + 1
        return templates.TemplateResponse("dataset.html", {
            "request": request, "name": name, "cases": cases,
            "type_counts": type_counts,
        })

    @app.post("/cases/add")
    async def add_case(
        dataset: str = Form(...),
        task_type: str = Form(...),
        input: str = Form(...),
        expected_answer: str = Form(""),
        rubric: str = Form(""),
        expected_tools: str = Form(""),
        tags: str = Form(""),
    ):
        # 逗号分隔字符串 -> 列表（去空白、去空项）
        def _split(s: str) -> list[str]:
            return [x.strip() for x in s.split(",") if x.strip()]

        tt = TaskType(task_type) if task_type in ("deterministic", "generative") else TaskType.DETERMINISTIC
        case = TestCase(
            id=new_id("case_"),
            dataset=dataset.strip(),
            task_type=tt,
            input=input.strip(),
            expected_answer=(expected_answer.strip() or None) if tt == TaskType.DETERMINISTIC else None,
            rubric=(rubric.strip() or None) if tt == TaskType.GENERATIVE else None,
            expected_tools=_split(expected_tools),
            tags=_split(tags),
        )
        store.add_cases([case])
        return RedirectResponse(url=f"/datasets/{case.dataset}", status_code=303)

    @app.post("/cases/{case_id}/delete")
    async def delete_case(case_id: str, dataset: str = Form(...)):
        store.delete_case(case_id)
        return RedirectResponse(url=f"/datasets/{dataset}", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_report(request: Request, run_id: str):
        report = aggregate_run(store, run_id)
        return templates.TemplateResponse("report.html", {
            "request": request, "report": report,
            "threshold": settings.threshold(),  # 模型平均分达标线，用于标红/绿
        })

    @app.post("/trigger")
    async def trigger_form(dataset: str = Form(...), models: list[str] = Form(...)):
        orch = make_orchestrator()
        run_id = await orch.run_eval(dataset, models)
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

    # ---------------- 设置页 ----------------
    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        load_from_store(store)
        # 组装可调项：键、标签、说明、当前生效值
        effective = settings.all_effective()
        items = [
            {"key": k, "label": SETTING_LABELS[k][0],
             "help": SETTING_LABELS[k][1], "value": effective[k]}
            for k in SETTING_SPEC
        ]
        return templates.TemplateResponse("settings.html", {
            "request": request,
            "items": items,
            "model_configs": store.list_model_configs(),
            "all_models": list_models(),
        })

    @app.post("/settings/save")
    async def settings_save(request: Request):
        form = await request.form()
        updates = {k: form[k] for k in SETTING_SPEC if k in form}
        settings.save(updates)
        return RedirectResponse(url="/settings", status_code=303)

    @app.post("/settings/models/add")
    async def settings_add_model(
        name: str = Form(...),
        kind: str = Form(...),
        model: str = Form(""),
        base_url: str = Form(""),
        api_key_env: str = Form(""),
        price_per_1k: str = Form("0"),
        quality: str = Form("0.8"),
        is_judge: str = Form("0"),
    ):
        store.add_model_config({
            "name": name.strip(), "kind": kind,
            "model": model.strip() or None,
            "base_url": base_url.strip() or None,
            "api_key_env": api_key_env.strip() or None,
            "price_per_1k": price_per_1k or 0,
            "quality": quality or 0.8,
            "is_judge": 1 if is_judge in ("1", "on", "true") else 0,
        })
        load_from_store(store)
        return RedirectResponse(url="/settings", status_code=303)

    @app.post("/settings/models/{name}/delete")
    async def settings_delete_model(name: str):
        store.delete_model_config(name)
        return RedirectResponse(url="/settings", status_code=303)

    # ---------------- API ----------------
    @app.get("/api/models")
    async def api_models():
        return {"models": list_models(), "judge": settings.judge_model()}

    @app.get("/api/datasets")
    async def api_datasets():
        return {"datasets": store.list_datasets()}

    @app.get("/api/datasets/{name}/cases")
    async def api_dataset_cases(name: str):
        return {"dataset": name, "cases": [c.to_row() for c in store.get_cases(name)]}

    @app.post("/api/cases")
    async def api_add_case(payload: dict):
        tt = TaskType(payload.get("task_type", "deterministic"))
        case = TestCase(
            id=new_id("case_"),
            dataset=payload["dataset"],
            task_type=tt,
            input=payload["input"],
            expected_answer=payload.get("expected_answer"),
            rubric=payload.get("rubric"),
            expected_tools=payload.get("expected_tools", []),
            tags=payload.get("tags", []),
        )
        store.add_cases([case])
        return {"ok": True, "case_id": case.id}

    @app.delete("/api/cases/{case_id}")
    async def api_delete_case(case_id: str):
        return {"ok": store.delete_case(case_id)}

    @app.post("/api/runs")
    async def api_create_run(payload: dict):
        dataset = payload["dataset"]
        models = payload["models"]
        orch = make_orchestrator()
        run_id = await orch.run_eval(dataset, models)
        return JSONResponse({"run_id": run_id, "report": aggregate_run(store, run_id)})

    @app.get("/api/runs/{run_id}")
    async def api_get_run(run_id: str):
        return aggregate_run(store, run_id)

    @app.get("/api/settings")
    async def api_get_settings():
        return {"settings": settings.all_effective(),
                "models": store.list_model_configs()}

    @app.post("/api/settings")
    async def api_save_settings(payload: dict):
        settings.save(payload)
        return {"ok": True, "settings": settings.all_effective()}

    return app
