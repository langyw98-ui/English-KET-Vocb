# src/flow/ket_partner/persistence.py
"""Agent-facing persistence contract. flow/ket_partner/ has ZERO runtime
dependency on persistence/ — WordRef is referenced only via TYPE_CHECKING.

KETPartnerRepos is a runtime_checkable Protocol; persistence/repos.Repos
structurally satisfies it (no explicit registration needed).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from src.persistence.models import WordRef


class VocabRepoProtocol(Protocol):
    async def get_topics_for_word(self, word: str, context: str = "") -> list[str]: ...
    async def get_ket_word(self, word: str, context: str = "") -> WordRef | None: ...
    async def get_ket_word_any_context(self, word: str) -> WordRef | None: ...
    async def words_in_topic_without_stats(self, topic: str) -> list[WordRef]: ...
    async def unexposed_notopic_words(self) -> list[WordRef]: ...
    async def topics_with_unmastered(self, exclude: str | None = None) -> list[str]: ...
    async def total_count(self) -> int: ...


class StatsRepoProtocol(Protocol):
    async def get(self, word: str, context: str = "") -> dict | None: ...
    async def apply_delta(
        self, word: str, context: str = "", delta: int = 0,
        exposed: bool = False, is_target: bool = False,
    ) -> dict | None: ...
    async def learning_count(self) -> int: ...
    async def oldest_learning_word(self) -> WordRef | None: ...
    async def increment_exposed(
        self, word: str, context: str = "", is_target: bool = False,
    ) -> None: ...
    async def list_all_with_vocab(self) -> list[dict]: ...


class ProfileRepoProtocol(Protocol):
    async def get(self) -> dict: ...
    async def update(self, **fields) -> None: ...


class LogRepoProtocol(Protocol):
    async def append(
        self, role: str, content: str,
        words_used: list[str] | None = None,
        target_words: list[dict[str, str]] | None = None,
        turn_id: int | None = None,
    ) -> None: ...
    async def recent(self, limit: int = 5) -> list[dict]: ...
    async def append_session_start(self) -> None: ...
    async def last_ai_message(self) -> dict | None: ...


class RecentSentencesRepoProtocol(Protocol):
    async def list_recent(self, limit: int = 20) -> list[str]: ...
    async def append(self, sentence: str, window: int = 20) -> None: ...
    async def list_recent_scaffolding(self, window: int = 20) -> list[list[str]]: ...


@runtime_checkable
class KETPartnerRepos(Protocol):
    """Agent-side persistence contract.
    Concrete impl: persistence/repos.py::Repos.
    """
    vocab: VocabRepoProtocol
    stats: StatsRepoProtocol
    profile: ProfileRepoProtocol
    log: LogRepoProtocol
    recent: RecentSentencesRepoProtocol


def get_repos(config: RunnableConfig) -> KETPartnerRepos:
    """Single access point for repos in node methods."""
    return config["configurable"]["repos"]
