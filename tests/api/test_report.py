import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app


@pytest.mark.asyncio
async def test_get_report_counts(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_report.db")
    monkeypatch.setenv("DB_PATH", db_file)

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/api/report")
        assert response.status_code == 200
        data = response.json()
        assert "mastered_count" in data
        assert "learning_count" in data
        assert "struggling_count" in data
        assert "used_count" in data
        assert "unused_count" in data
        assert "total_words" in data


@pytest.mark.asyncio
async def test_get_report_by_category(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_report_cat.db")
    monkeypatch.setenv("DB_PATH", db_file)

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Test valid category
        response = await ac.get("/api/report/learning?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "learning"
        assert data["page"] == 1
        assert data["page_size"] == 10
        assert "total" in data
        assert "words" in data

        # Test invalid category
        inv_cat_resp = await ac.get("/api/report/invalid_cat")
        assert inv_cat_resp.status_code == 400

        # Test invalid page params
        inv_page_resp = await ac.get("/api/report/learning?page=0")
        assert inv_page_resp.status_code == 400

        inv_size_resp = await ac.get("/api/report/learning?page_size=600")
        assert inv_size_resp.status_code == 400
