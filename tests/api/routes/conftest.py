import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock

from langgraph.graph.state import CompiledStateGraph
from src.api.app import app
from src.api.deps import get_agent
from src.api.llm_key import LlmKeyStatus


@pytest.fixture
def mock_agent() -> AsyncMock:
    agent = AsyncMock(spec=CompiledStateGraph)
    agent.ainvoke = AsyncMock(return_value={"messages": []})
    return agent


@pytest.fixture
def llm_key_status() -> LlmKeyStatus:
    return LlmKeyStatus()


@pytest.fixture
async def client(mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    db_file = str(tmp_path / "test_chat_route.db")
    monkeypatch.setenv("DB_PATH", db_file)

    app.dependency_overrides[get_agent] = lambda: mock_agent

    async with app.router.lifespan_context(app):
        app.state.llm_key_status = llm_key_status

        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    app.dependency_overrides.clear()
