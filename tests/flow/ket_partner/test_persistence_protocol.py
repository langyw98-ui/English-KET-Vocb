# tests/flow/ket_partner/test_persistence_protocol.py
# Imports use `src.persistence.*` (not bare `persistence.*`) to avoid
# shadowing when this test runs in the same pytest session as
# tests/persistence/ — that package's tests/__init__.py turns
# `persistence` into a top-level package on sys.path, shadowing the
# real `src/persistence/` package. Mirrors tests/persistence/test_repos.py.
import pytest

from src.persistence.bootstrap import init_db
from src.persistence.repos import Repos  # concrete impl


@pytest.mark.asyncio
async def test_repos_satisfies_protocol(temp_db_path):
    """Spec §11: isinstance(Repos(...), KETPartnerRepos) must pass —
    runtime_checkable verifies attribute existence."""
    from flow.ket_partner.persistence import KETPartnerRepos

    db = await init_db(temp_db_path, csv_path=None)
    repos = Repos.for_user(db, "default")
    assert isinstance(repos, KETPartnerRepos)
    await repos.close()


def test_get_repos_extracts_from_config():
    """get_repos pulls repos out of config['configurable']['repos']."""
    from flow.ket_partner.persistence import get_repos

    sentinel = object()  # not a real Repos; just verifying the extraction
    config = {"configurable": {"repos": sentinel}}
    assert get_repos(config) is sentinel
