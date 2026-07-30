# tests/reporting/conftest.py
"""Shared fixtures for tests/reporting/.

Mirrors tests/ket_partner/conftest.py subset relevant to reporting:
- temp_db_path: file path under pytest's tmp_path; file is created by
  aiosqlite.connect, removed by tmp_path teardown.
- _close_aiosqlite_connections (autouse): track every aiosqlite connection
  opened during a test and close them on teardown. Without this, aiosqlite
  worker threads block on their tx queue forever, holding a reference to
  the event loop. pytest-asyncio closes the loop but the process won't
  exit (thread still alive), which looks like a hang on Windows.
"""
import pytest


@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture(autouse=True)
async def _close_aiosqlite_connections():
    """Track every aiosqlite connection opened during a test and close
    them on teardown.

    Without this, aiosqlite worker threads block on their tx queue
    forever, holding a reference to the event loop. pytest-asyncio
    closes the loop but the process won't exit (thread still alive),
    which means anything buffering stdout until exit (RTK hook, or a
    simple `pytest | tail` waiting on pipe EOF) appears to hang.
    """
    import aiosqlite

    original_connect = aiosqlite.connect
    opened: list = []

    async def _tracking(*args, **kwargs):
        conn = await original_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    aiosqlite.connect = _tracking
    try:
        yield
    finally:
        aiosqlite.connect = original_connect
        for conn in opened:
            try:
                await conn.close()
            except Exception:  # noqa: BLE001, S110
                pass
