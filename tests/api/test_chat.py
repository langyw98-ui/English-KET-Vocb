import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app


@pytest.mark.asyncio
async def test_post_chat(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_chat.db")
    monkeypatch.setenv("DB_PATH", db_file)

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post("/api/chat", json={"text": "Hello"})
        assert response.status_code == 200
        data = response.json()
        assert "ai_reply" in data
        assert "turn_id" in data
