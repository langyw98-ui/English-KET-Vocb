from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app


@pytest.mark.asyncio
async def test_status_green_initial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_file = str(tmp_path / "test_llm_status.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setattr("src.api.routes.llm._read_current_key", lambda: "sk-abcdefghijklmno")

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        res = await ac.get("/api/llm/status")
        assert res.status_code == 200
        data = res.json()
        assert data == {
            "state": "green",
            "masked_key": "sk-a***lmno",
            "last_error": None,
        }


@pytest.mark.asyncio
async def test_status_red_when_no_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_file = str(tmp_path / "test_llm_status_nokey.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setattr("src.api.routes.llm._read_current_key", lambda: "")

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        res = await ac.get("/api/llm/status")
        assert res.status_code == 200
        data = res.json()
        assert data["state"] == "red"
        assert data["masked_key"] is None


@pytest.mark.asyncio
async def test_status_red_with_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_file = str(tmp_path / "test_llm_status_err.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setattr("src.api.routes.llm._read_current_key", lambda: "sk-abcdefghijklmno")

    async with app.router.lifespan_context(app):
        app.state.llm_key_status.set_error("auth error")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res = await ac.get("/api/llm/status")
            assert res.status_code == 200
            data = res.json()
            assert data["state"] == "red"
            assert data["last_error"] == "auth error"
