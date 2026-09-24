"""启动 Web 服务：python run_web.py，浏览器打开 http://localhost:8000"""
import uvicorn

from eval_system.web import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
