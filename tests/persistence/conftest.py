# tests/persistence/conftest.py
import os
import tempfile

import pytest


@pytest.fixture
def temp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)
