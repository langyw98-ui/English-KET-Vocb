import asyncio
import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from flow.ket_partner.db import Repos
from src.api.deps import User, get_agent, get_current_user, get_db, get_settings
from src.api.schemas import ChatRequest, ChatResponse
from src.api.settings import Settings

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
    agent: CompiledStateGraph = Depends(get_agent),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    repos = Repos.for_user(db, user.id)
    user_info = {"nickname": user.nickname, "age": user.age}

    try:
        result_state = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [HumanMessage(content=req.text)]},
                config={
                    "configurable": {
                        "thread_id": f"{user.id}:main",
                        "user_id": user.id,
                        "repos": repos,
                        "user_info": user_info,
                    }
                },
            ),
            timeout=settings.REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="agent timeout")

    messages = result_state.get("messages", [])
    if not messages:
        raise HTTPException(status_code=500, detail="agent returned empty messages")

    ai_text = str(messages[-1].content)
    profile = await repos.profile.get()
    turn_id = profile.get("total_turns", 0)

    return ChatResponse(ai_reply=ai_text, turn_id=turn_id)
