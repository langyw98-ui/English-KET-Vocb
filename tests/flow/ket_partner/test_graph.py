import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from flow.ket_partner.graph import build_agent
from flow.ket_partner.input_classifier import IntentClassification
from flow.ket_partner.sentence_naturalness import NaturalnessResult
from flow.ket_partner.translation_evaluator import TranslationEval
from src.persistence.bootstrap import init_db
from src.persistence.repos import Repos


@pytest.fixture
async def setup(temp_db_path):
    csv_text = "word,part_of_speech,topic\ncat,n,Animals\ndog,n,Animals\nbig,adj,\nthe,det,\nis,v,\n"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = f.name
    db = await init_db(temp_db_path, csv_path=csv_path)
    repos = Repos.for_user(db, "default")
    yield repos
    await db.close()


def _mock_llm_with_responses(responses: dict):
    responses.setdefault(NaturalnessResult, NaturalnessResult(ok=True, reason=""))
    llm = MagicMock()
    def structured(schema, **kwargs):
        bound = MagicMock()
        bound.ainvoke = AsyncMock(return_value=responses.get(schema))
        return bound
    llm.with_structured_output = MagicMock(side_effect=structured)
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=MagicMock(content="The big cat is here."))
    llm.bind = MagicMock(return_value=bound)
    return llm


def _make_config(repos, thread_id="t1"):
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": repos._user_id,
            "repos": repos,
            "user_info": {"nickname": "test", "age": 8},
        }
    }


@pytest.mark.asyncio
async def test_graph_first_turn_generates_sentence(setup):
    repos = setup
    llm = _mock_llm_with_responses({})
    agent = await build_agent(llm_flash=llm, llm_smart=llm, db=repos._db)
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config=_make_config(repos, thread_id="t1"),
    )
    ai_msg = result["messages"][-1].content
    assert "请把这句译成中文" in ai_msg and "big cat" in ai_msg
    assert "big cat" in ai_msg.lower()


@pytest.mark.asyncio
async def test_graph_translation_correct_flow(setup):
    repos = setup
    responses = {
        IntentClassification: IntentClassification(intent="translation", asked_word=None),
        TranslationEval: TranslationEval(correct_translation="那只大猫在这里", wrong_words=[]),
    }
    llm = _mock_llm_with_responses(responses)
    agent = await build_agent(llm_flash=llm, llm_smart=llm, db=repos._db)

    # Turn 1: first sentence
    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config=_make_config(repos, thread_id="t2"),
    )
    # Turn 2: correct translation
    await agent.ainvoke(
        {"messages": [HumanMessage(content="那只大猫在这里")]},
        config=_make_config(repos, thread_id="t2"),
    )
    cat = await repos.stats.get("cat")
    assert cat["mastery_score"] >= 1
