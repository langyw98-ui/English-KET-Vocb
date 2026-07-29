import asyncio
import logging
from typing import Any, Final

import aiosqlite
import openai
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from flow.ket_partner.db import Repos
from src.api.deps import User, get_agent, get_current_user, get_db, get_llm_key_status, get_settings
from src.api.llm_key import LlmKeyStatus, _read_current_key
from src.api.schemas import ChatRequest, ChatResponse
from src.api.settings import Settings

logger = logging.getLogger("ket_partner")

LLM_AUTH_ERROR_MSG: Final[str] = "API key 无效或无权限"

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
    agent: CompiledStateGraph[Any, None, Any, Any] = Depends(get_agent),
    settings: Settings = Depends(get_settings),
    llm_key_status: LlmKeyStatus = Depends(get_llm_key_status),
) -> ChatResponse:
    if not _read_current_key():
        raise HTTPException(status_code=503, detail="LLM key not configured")

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
    except asyncio.TimeoutError as e:
        logger.warning("agent execution timeout: %s", e, exc_info=True)
        raise HTTPException(status_code=504, detail="agent timeout")
    except openai.APITimeoutError as e:
        logger.warning("LLM SDK timeout: %s", e, exc_info=True)
        raise HTTPException(status_code=504, detail="LLM timeout")
    except (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError) as e:
        llm_key_status.set_error(LLM_AUTH_ERROR_MSG)
        logger.warning("LLM auth/key failed: %s", e, exc_info=True)
        raise HTTPException(status_code=401, detail="LLM auth failed")
    except (openai.APIConnectionError, openai.RateLimitError) as e:
        logger.warning("LLM transient failure: %s", e, exc_info=True)
        status_code = 429 if isinstance(e, openai.RateLimitError) else 502
        raise HTTPException(status_code=status_code, detail=str(e) or "transient error")

    llm_key_status.clear_error()

    messages = result_state.get("messages", [])
    if not messages:
        raise HTTPException(status_code=500, detail="agent returned empty messages")

    ai_text = str(messages[-1].content).strip()
    if not ai_text:
        raise HTTPException(status_code=500, detail="agent returned blank reply")

    profile = await repos.profile.get()
    turn_id = profile.get("total_turns", 0)

    return ChatResponse(ai_reply=ai_text, turn_id=turn_id)
