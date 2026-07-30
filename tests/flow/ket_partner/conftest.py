# tests/flow/ket_partner/conftest.py
import os
import tempfile

import pytest


@pytest.fixture
def temp_db_path():
    """File-based temp DB path. The file is removed if it exists so that
    aiosqlite.connect can create it fresh."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)
