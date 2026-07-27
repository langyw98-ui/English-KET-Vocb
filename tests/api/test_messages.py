import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app


@pytest.mark.asyncio
async def test_get_messages(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_messages.db")
    monkeypatch.setenv("DB_PATH", db_file)

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/api/messages?limit=15")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

        # Test invalid limit (<1 or >100)
        inv_low = await ac.get("/api/messages?limit=0")
        assert inv_low.status_code == 400

        inv_high = await ac.get("/api/messages?limit=101")
        assert inv_high.status_code == 400
