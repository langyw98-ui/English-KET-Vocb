import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.messages import HumanMessage

from flow.ket_partner.agent import build_agent
from flow.ket_partner.db import init_db
from flow.ket_partner.input_classifier import IntentClassification
from flow.ket_partner.translation_evaluator import TranslationEval
from flow.ket_partner.word_meaning_lookup import WordMeaning


@pytest.fixture
async def setup(temp_db_path):
    csv_text = "word,part_of_speech,topic\ncat,n,Animals\ndog,n,Animals\nbig,adj,\nthe,det,\nis,v,\n"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = f.name
    repos = await init_db(temp_db_path, csv_path=csv_path)
    yield repos
    await repos.close()


def _mock_llm_with_responses(responses: dict):
    llm = MagicMock()
    def structured(schema):
        bound = MagicMock()
        bound.ainvoke = AsyncMock(return_value=responses.get(schema))
        return bound
    llm.with_structured_output = MagicMock(side_effect=structured)
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=MagicMock(content="The big cat is here."))
    llm.bind = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_graph_first_turn_generates_sentence(setup):
    repos = setup
    llm = _mock_llm_with_responses({})
    agent = await build_agent(llm_flash=llm, llm_smart=llm, repos=repos, info={"nickname_kid": "test", "age": 8})
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "t1"}},
    )
    ai_msg = result["messages"][-1].content
    # Tightened (was `or`): the 🔤 prefix is always emitted even when the
    # sentence is empty (C1 bug). Require BOTH the prefix AND a real sentence
    # body that contains the target word.
    assert "🔤" in ai_msg and "big cat" in ai_msg
    # R2: explicit non-empty content assertion.
    assert "big cat" in ai_msg.lower()


@pytest.mark.asyncio
async def test_graph_translation_correct_flow(setup):
    repos = setup
    responses = {
        IntentClassification: IntentClassification(intent="translation", asked_word=None),
        TranslationEval: TranslationEval(wrong_words=[], correct_meanings={}),
    }
    llm = _mock_llm_with_responses(responses)
    agent = await build_agent(llm_flash=llm, llm_smart=llm, repos=repos, info={"nickname_kid": "test", "age": 8})

    # Turn 1: first sentence
    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "t2"}},
    )
    # Turn 2: correct translation
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="那只大猫在这里")]},
        config={"configurable": {"thread_id": "t2"}},
    )
    cat = await repos.stats.get("cat")
    assert cat["mastery_score"] >= 1
