import pytest
from httpx import ASGITransport, AsyncClient
from src.api.app import app


@pytest.fixture
async def client(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    db_file = str(tmp_path / "test_integration.db")
    monkeypatch.setenv("DB_PATH", db_file)

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
