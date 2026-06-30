import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def mock_llm():
    from unittest.mock import AsyncMock, MagicMock

    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock()
    llm.bind = MagicMock(return_value=llm)
    return llm
