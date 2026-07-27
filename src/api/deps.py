import aiosqlite
from fastapi import HTTPException, Request
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from src.api.settings import Settings


class User(BaseModel):
    id: str
    nickname: str
    age: int


async def get_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_db(request: Request) -> aiosqlite.Connection:
    return request.app.state.db


async def get_agent(request: Request) -> CompiledStateGraph:
    return request.app.state.agent


async def get_current_user(request: Request) -> User:
    settings: Settings = request.app.state.settings
    if settings.AUTH_MODE == "disabled":
        db: aiosqlite.Connection = request.app.state.db
        async with db.execute(
            "SELECT id, nickname, age FROM users WHERE id='default'"
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=500, detail="default user not seeded")
        return User(id=row[0], nickname=row[1], age=row[2])

    raise NotImplementedError("JWT auth not implemented yet")
