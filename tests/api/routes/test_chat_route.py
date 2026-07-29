import asyncio
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from httpx import AsyncClient

from src.api.llm_key import LlmKeyStatus


@pytest.mark.asyncio
async def test_chat_returns_503_when_no_key(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "")
    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 503
    assert mock_agent.ainvoke.await_count == 0
    assert llm_key_status.last_error is None


@pytest.mark.asyncio
async def test_chat_401_on_auth_error(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "sk-test")
    mock_agent.ainvoke.side_effect = openai.AuthenticationError(
        message="invalid key", response=MagicMock(), body=None
    )
    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 401
    assert llm_key_status.last_error == "API key 无效或无权限"
    assert llm_key_status.state == "red"
    mock_agent.ainvoke.assert_awaited()


@pytest.mark.asyncio
async def test_chat_401_on_bad_request(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "sk-test")
    mock_agent.ainvoke.side_effect = openai.BadRequestError(
        message="bad format key", response=MagicMock(), body=None
    )
    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 401
    assert llm_key_status.last_error == "API key 无效或无权限"
    mock_agent.ainvoke.assert_awaited()


@pytest.mark.asyncio
async def test_chat_clears_error_on_success(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "sk-test")
    llm_key_status.set_error("previous error")
    mock_msg = AsyncMock()
    mock_msg.content = "AI response"
    mock_agent.ainvoke.return_value = {"messages": [mock_msg]}

    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 200
    assert llm_key_status.last_error is None
    assert llm_key_status.state == "green"
    mock_agent.ainvoke.assert_awaited()


@pytest.mark.asyncio
async def test_chat_504_on_timeout(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "sk-test")
    mock_agent.ainvoke.side_effect = asyncio.TimeoutError()
    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 504
    assert llm_key_status.last_error is None
    mock_agent.ainvoke.assert_awaited()


@pytest.mark.asyncio
async def test_chat_504_on_sdk_timeout(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "sk-test")
    mock_agent.ainvoke.side_effect = openai.APITimeoutError(request=MagicMock())
    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 504
    assert llm_key_status.last_error is None
    mock_agent.ainvoke.assert_awaited()


@pytest.mark.asyncio
async def test_chat_500_on_code_bug(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "sk-test")
    mock_agent.ainvoke.side_effect = KeyError("bug_field")
    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 500
    assert llm_key_status.last_error is None
    mock_agent.ainvoke.assert_awaited()
