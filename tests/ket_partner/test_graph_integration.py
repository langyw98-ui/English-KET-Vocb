import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.messages import HumanMessage

from flow.ket_partner.agent import build_agent
from flow.ket_partner.db import init_db
from flow.ket_partner.input_classifier import IntentClassification
from flow.ket_partner.translation_evaluator import TranslationEval
from flow.ket_partner.word_meaning_lookup import WordMeaning


def _mock_llm(intent_resp, eval_resp=None, meaning_resp=None, sentence_text="The cat is on the bed."):
    llm = MagicMock()
    responses = {IntentClassification: intent_resp}
    if eval_resp:
        responses[TranslationEval] = eval_resp
    if meaning_resp:
        responses[WordMeaning] = meaning_resp

    def structured(schema):
        bound = MagicMock()
        bound.ainvoke = AsyncMock(return_value=responses.get(schema))
        return bound
    llm.with_structured_output = MagicMock(side_effect=structured)

    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=MagicMock(content=sentence_text))
    llm.bind = MagicMock(return_value=bound)
    return llm


@pytest.fixture
async def setup(temp_db_path):
    csv_text = "word,part_of_speech,topic\ncat,n,Animals\ndog,n,Animals\nbed,n,Home\nthe,det,\non,prep,\nis,v,\n"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = f.name
    repos = await init_db(temp_db_path, csv_path=csv_path)
    yield repos
    await repos.close()


@pytest.mark.asyncio
async def test_first_turn_generates_sentence(setup):
    llm = _mock_llm(intent_resp=None, sentence_text="The cat is on the bed.")
    agent = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "first"}},
    )
    assert "🔤" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_correct_translation_increments_mastery(setup):
    llm = _mock_llm(
        intent_resp=IntentClassification(intent="translation", asked_word=None),
        eval_resp=TranslationEval(wrong_words=[], correct_meanings={}),
        sentence_text="The cat is on the bed.",
    )
    agent = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})

    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "correct"}},
    )
    await agent.ainvoke(
        {"messages": [HumanMessage(content="猫在床上")]},
        config={"configurable": {"thread_id": "correct"}},
    )
    cat = await setup.stats.get("cat")
    assert cat["mastery_score"] >= 1


@pytest.mark.asyncio
async def test_wrong_translation_deducts(setup):
    llm = _mock_llm(
        intent_resp=IntentClassification(intent="translation", asked_word=None),
        eval_resp=TranslationEval(wrong_words=["cat"], correct_meanings={"cat": "猫"}),
        sentence_text="The cat is on the bed.",
    )
    agent = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})
    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "wrong"}},
    )
    await agent.ainvoke(
        {"messages": [HumanMessage(content="狗在床上")]},
        config={"configurable": {"thread_id": "wrong"}},
    )
    cat = await setup.stats.get("cat")
    assert cat["wrong_count"] >= 1


@pytest.mark.asyncio
async def test_idk_deducts_target_only(setup):
    llm = _mock_llm(
        intent_resp=IntentClassification(intent="idk", asked_word=None),
        meaning_resp=WordMeaning(meaning="猫"),
        sentence_text="The cat is on the bed.",
    )
    agent = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})
    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "idk"}},
    )
    await agent.ainvoke(
        {"messages": [HumanMessage(content="我不会")]},
        config={"configurable": {"thread_id": "idk"}},
    )
    cat = await setup.stats.get("cat")
    bed = await setup.stats.get("bed")
    assert cat["wrong_count"] >= 1
    if bed:
        assert bed["wrong_count"] == 0


@pytest.mark.asyncio
async def test_asks_meaning_deducts_asked_word(setup):
    llm = _mock_llm(
        intent_resp=IntentClassification(intent="asks_meaning", asked_word="cat"),
        meaning_resp=WordMeaning(meaning="猫"),
        sentence_text="The cat is on the bed.",
    )
    agent = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})
    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "ask"}},
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="cat 是什么意思")]},
        config={"configurable": {"thread_id": "ask"}},
    )
    cat = await setup.stats.get("cat")
    assert cat["wrong_count"] >= 1
    assert "猫" in result["messages"][-1].content
