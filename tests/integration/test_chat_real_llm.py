import os
import pytest
from httpx import ASGITransport, AsyncClient
from src.api.app import app


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("DASHSCOPE_API_KEY"),
    reason="requires DASHSCOPE_API_KEY"
)
async def test_chat_succeeds_with_real_key(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
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
