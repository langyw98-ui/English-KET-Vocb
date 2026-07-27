from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from flow.common import llm_flash, llm_max, logger
from flow.ket_partner.agent import build_agent
from flow.ket_partner.db import init_db
from src.api.routes import chat, messages, report
from src.api.settings import Settings

DEFAULT_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "KET_vocabulary.csv"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = os.environ.get("DB_PATH")
    if db_path:
        settings = Settings(DB_PATH=db_path)
    else:
        settings = Settings()

    csv_path = settings.CSV_PATH or (
        DEFAULT_CSV if os.path.exists(DEFAULT_CSV) else None
    )

    db = await init_db(
        settings.DB_PATH,
        csv_path=csv_path,
        default_nickname=settings.KID_NICKNAME,
        default_age=settings.KID_AGE,
    )

    checkpointer = AsyncSqliteSaver(db)
    await checkpointer.setup()

    agent = await build_agent(llm_flash, llm_max, db, checkpointer=checkpointer)
    app.state.settings = settings
    app.state.db = db
    app.state.agent = agent

    yield

    inner = getattr(agent, "agent", None)
    if inner is not None:
        try:
            await inner.aclose()
        except Exception as e:
            logger.warning(
                f"agent.aclose() failed during shutdown: {e}", exc_info=True
            )
    await app.state.db.close()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception(f"unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "internal error"})


app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(report.router, prefix="/api/report", tags=["report"])
app.include_router(messages.router, prefix="/api/messages", tags=["messages"])

if Path("web/dist").exists():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    @app.get("/")
    async def index():
        return FileResponse("web/dist/index.html")

    app.mount("/", StaticFiles(directory="web/dist", html=True), name="static")
