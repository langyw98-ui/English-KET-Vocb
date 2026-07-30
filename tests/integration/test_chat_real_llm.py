import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app
from src.flow.common import _resolve_dashscope_api_key


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    not _resolve_dashscope_api_key(),
    reason="requires DASHSCOPE_API_KEY or ~/.config/pet/config.yaml"
)
async def test_chat_succeeds_with_real_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_file = str(tmp_path / "test_chat_real.db")
    monkeypatch.setenv("DB_PATH", db_file)

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post("/api/chat", json={"text": "Hello"})
        assert response.status_code == 200
        data = response.json()
        assert "ai_reply" in data
        assert "turn_id" in data
        assert app.state.llm_key_status.last_error is None
