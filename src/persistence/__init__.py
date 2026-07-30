"""Top-level persistence package.

Re-exports grow task-by-task. Full set lands in Task 8.
"""
from src.persistence.bootstrap import init_db

__all__ = ["init_db"]
