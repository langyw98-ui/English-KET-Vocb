import asyncio

from langchain_core.runnables import RunnableConfig

from flow.common import logger
from flow.ket_partner import nodes
from flow.ket_partner.dialogue_domain import run_profile_summary
from flow.ket_partner.persistence import KETPartnerRepos
from flow.ket_partner.state import BTPKetState


class KETPartnerAgent:
    def __init__(self, llm_flash, llm_smart, config):
        self.llm_flash = llm_flash
        self.llm_smart = llm_smart
        self.config = config
        self._bg_tasks = set()

    async def init_state(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return await nodes.init_state(state, config, self)

    async def classify_intent_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return await nodes.classify_intent_node(state, config, self)

    async def evaluate_translation_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return await nodes.evaluate_translation_node(state, config, self)

    async def lookup_target_meaning_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return await nodes.lookup_target_meaning_node(state, config, self)

    async def lookup_asked_meaning_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return await nodes.lookup_asked_meaning_node(state, config, self)

    async def update_mastery_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return await nodes.update_mastery_node(state, config, self)

    async def select_target_word_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return await nodes.select_target_word_node(state, config, self)

    async def generate_sentence_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return await nodes.generate_sentence_node(state, config, self)

    async def format_output_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return await nodes.format_output_node(state, config, self)

    async def explain_meaning_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return await nodes.explain_meaning_node(state, config, self)

    async def redirect_to_translate_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return await nodes.redirect_to_translate_node(state, config, self)

    async def compliance_redirect_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return await nodes.compliance_redirect_node(state, config, self)

    async def persist_turn_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return await nodes.persist_turn_node(state, config, self)

    async def _run_summary_safe(self, repos: KETPartnerRepos) -> None:
        try:
            await run_profile_summary(self.llm_smart, repos)
        except (TimeoutError, RuntimeError, ValueError, AttributeError, TypeError) as e:
            logger.warning(f"background summary failed: {e}", exc_info=True)

    async def aclose(self, timeout: float = 2.0) -> None:
        if not self._bg_tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._bg_tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"aclose: {len(self._bg_tasks)} background task(s) did not finish within {timeout}s; cancelling"
            )
            for t in self._bg_tasks:
                t.cancel()
