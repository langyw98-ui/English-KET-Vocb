"""Top-level persistence package.

Re-exports only the public API. SCHEMA_SQL, _import_csv, and
migrate_old_schema_if_needed stay private to their modules.
"""
from src.persistence.bootstrap import init_db
from src.persistence.models import MASTERY_CAP, WordRef, derive_status
from src.persistence.repos import (
    LogRepo,
    ProfileRepo,
    RecentSentencesRepo,
    Repos,
    StatsRepo,
    VocabRepo,
)

__all__ = [
    "init_db",
    "WordRef", "MASTERY_CAP", "derive_status",
    "VocabRepo", "StatsRepo", "ProfileRepo", "LogRepo",
    "RecentSentencesRepo", "Repos",
]
