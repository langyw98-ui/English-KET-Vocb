import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

# 确保 src 目录在 sys.path 中, 使 from flow... 导入无缝解析
src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from flow.common import default_llm_service, logger
from flow.ket_partner.graph import build_agent
from src.api.llm_key import LlmKeyStatus
from src.api.routes import chat, llm, messages, report
from src.api.settings import Settings
from src.persistence.bootstrap import init_db

DEFAULT_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "KET_vocabulary.csv"
)


from typing import AsyncGenerator

from fastapi.responses import HTMLResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    db_path = os.environ.get("DB_PATH")
    if db_path:
        settings = Settings(DB_PATH=db_path)
    else:
        settings = Settings()

    csv_path = settings.CSV_PATH or (
        DEFAULT_CSV if os.path.exists(DEFAULT_CSV) else None
    )

    db = None
    try:
        db = await init_db(
            settings.DB_PATH,
            csv_path=csv_path,
            default_nickname=settings.KID_NICKNAME,
            default_age=settings.KID_AGE,
        )

        checkpointer = AsyncSqliteSaver(db)
        await checkpointer.setup()

        agent = await build_agent(default_llm_service, checkpointer=checkpointer)
        app.state.settings = settings
        app.state.db = db
        app.state.agent = agent
        app.state.llm_key_status = LlmKeyStatus()

        yield

    finally:
        if hasattr(app.state, "agent"):
            inner = getattr(app.state.agent, "agent", None)
            if inner is not None:
                try:
                    await inner.aclose()
                except (RuntimeError, asyncio.TimeoutError) as e:
                    logger.warning(
                        f"agent.aclose() failed during shutdown: {e}", exc_info=True
                    )
        if db is not None:
            try:
                await db.close()
            except (RuntimeError, OSError, aiosqlite.Error) as e:
                logger.warning(f"db.close() failed during shutdown: {e}", exc_info=True)


from fastapi.openapi.docs import get_swagger_ui_html

app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=app.openapi_url or "/openapi.json",
        title=app.title + " - Swagger UI",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    )


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "internal error"})


app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(report.router, prefix="/api/report", tags=["report"])
app.include_router(messages.router, prefix="/api/messages", tags=["messages"])
app.include_router(llm.router, prefix="/api/llm", tags=["llm"])


if Path("web/dist").exists():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse("web/dist/index.html")

    app.mount("/", StaticFiles(directory="web/dist", html=True), name="static")
