import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.messages import AIMessage, HumanMessage

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
    ai_msg = result["messages"][-1].content
    # Tightened (was `assert "🔤" in ...`): the prefix is always emitted even
    # when the sentence is empty (C1 bug). Require both prefix and a real
    # non-empty sentence.
    assert "🔤" in ai_msg
    assert "cat" in ai_msg.lower()
    # The quoted sentence body must not be empty.
    assert '""' not in ai_msg


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
    # The turn-1 target word is picked via ORDER BY RANDOM() (topic then word),
    # so read the actual target from the log rather than hard-coding `cat`.
    history = await setup.log.recent(limit=20)
    ai_messages = [h for h in history if h["role"] == "ai"]
    assert ai_messages, "expected at least one AI message"
    t1_target = ai_messages[0]["target_words"][0] if ai_messages[0]["target_words"] else None
    assert t1_target is not None, "turn 1 should have set a target word"

    # After idk, the turn-1 target should have wrong_count >= 1
    target_stats = await setup.stats.get(t1_target)
    assert target_stats is not None
    assert target_stats["wrong_count"] >= 1

    # idk should only deduct the target — no other word should have wrong_count > 0
    async with setup.log._db.execute(
        "SELECT word, wrong_count FROM vocab_stats WHERE wrong_count > 0"
    ) as cur:
        rows = await cur.fetchall()
    wrong_words = {r[0] for r in rows}
    assert wrong_words == {t1_target}, f"only the target should be deducted, got {wrong_words}"


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


# ---------------------------------------------------------------------------
# Regression tests added in task-15 fix pass.
#
# These exercise the PRODUCTION calling convention from main.py:48 which sends
# `messages[-5:]` (a window of up to 5 messages). On turn 2 that is
# [HumanMsg1, AIMsg1, HumanMsg2] (length 3). The original integration tests
# all passed single-message inputs, masking the C1/C2 bugs.
# ---------------------------------------------------------------------------


def _mock_llm_seq(
    intent_resp,
    eval_resp=None,
    meaning_resp=None,
    sentence_texts=("The cat is on the bed.",),
):
    """Like _mock_llm but cycles through `sentence_texts` on successive
    .bind().ainvoke calls (one per generate_sentence call)."""
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

    seq = list(sentence_texts)

    def make_bound(*args, **kwargs):
        b = MagicMock()

        async def ainvoke(*a, **kw):
            content = seq.pop(0) if seq else (sentence_texts[-1] if sentence_texts else "")
            return MagicMock(content=content)
        b.ainvoke = AsyncMock(side_effect=ainvoke)
        return b

    # Each .bind() call returns a fresh bound that pulls from the shared seq.
    llm.bind = MagicMock(side_effect=make_bound)
    return llm


@pytest.mark.asyncio
async def test_e2e_multi_turn_with_windowed_messages(setup):
    """R1: reproduces main.py's `messages[-5:]` calling convention.

    Before fixes:
      - C1: turn-1 AI message had an empty sentence body (`""`).
      - C2: turn-2 with 3-message window reset `last_english_sentence`,
        causing route_after_init to re-select a target word (infinite loop)
        and evaluate_translation ran against an empty sentence so mastery
        never incremented.
    """
    llm = _mock_llm_seq(
        intent_resp=IntentClassification(intent="translation", asked_word=None),
        eval_resp=TranslationEval(wrong_words=[], correct_meanings={}),
        sentence_texts=("The cat is on the bed.", "The dog is on the bed."),
    )
    agent = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})

    # Turn 1: first message, single-element window.
    r1 = await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "win"}},
    )
    ai1 = r1["messages"][-1].content
    # C1 regression: sentence must be non-empty AND contain the target word.
    assert "🔤" in ai1, "translation prompt prefix must be present"
    assert "cat" in ai1.lower(), "turn-1 AI message must contain the target word 'cat'"
    assert '""' not in ai1, "turn-1 AI message must NOT have an empty quoted sentence (C1)"

    # Turn 2: production pattern — `messages[-5:]` yields 3 messages.
    window = [
        HumanMessage(content="hi"),
        AIMessage(content=ai1),
        HumanMessage(content="猫在床上"),
    ]
    r2 = await agent.ainvoke(
        {"messages": window},
        config={"configurable": {"thread_id": "win"}},
    )
    ai2 = r2["messages"][-1].content

    # C2 regression A: mastery for `cat` must increment after a correct
    # translation (was stuck at 0 before because evaluate_translation ran
    # against an empty sentence).
    cat = await setup.stats.get("cat")
    assert cat is not None, "cat stats row should exist after turn 1"
    assert cat["mastery_score"] >= 1, (
        f"cat.mastery_score should be >=1 after correct translation, got {cat['mastery_score']} (C2)"
    )

    # C2 regression B: turn 2 must route to evaluate the translation, not
    # re-select a target word. The mastery assertion above is the load-bearing
    # check: if route_after_init had re-selected (the C2 bug), evaluate_translation
    # would never run and mastery would stay at 0. As an additional guard
    # against the infinite-loop symptom, verify turn 2 produced a NEW sentence
    # (the second entry in our mock seq, "The dog is on the bed.") — that only
    # happens after a successful evaluate→update_mastery→select_target_word→
    # generate_sentence cycle, which is the correct turn-2 path.
    assert "dog" in ai2.lower(), (
        "turn-2 AI message should contain the NEXT sentence's target word 'dog' "
        "(confirms turn 2 completed the full evaluate→generate cycle, not re-selected)"
    )


@pytest.mark.asyncio
async def test_asks_meaning_does_not_re_expose_words(setup):
    """R3: persist_turn must not re-increment exposed_count for words that
    came from a PRIOR sentence. Only the NEW sentence's words should be
    counted (spec §11.9).

    Before I1 fix: turn 2 (asks_meaning) re-incremented exposure for the
    turn-1 sentence's words even though no new sentence was generated.
    """
    llm = _mock_llm_seq(
        intent_resp=IntentClassification(intent="asks_meaning", asked_word="cat"),
        meaning_resp=WordMeaning(meaning="猫"),
        sentence_texts=("The cat is on the bed.",),
    )
    agent = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})

    # Turn 1: produces a sentence containing [cat, bed] (per validator).
    r1 = await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "expose"}},
    )
    cat_after_t1 = await setup.stats.get("cat")
    bed_after_t1 = await setup.stats.get("bed")
    assert cat_after_t1 is not None and cat_after_t1["exposed_count"] >= 1, (
        "cat should be exposed after turn 1"
    )
    assert bed_after_t1 is not None and bed_after_t1["exposed_count"] >= 1, (
        "bed should be exposed after turn 1"
    )
    cat_exposed_t1 = cat_after_t1["exposed_count"]
    bed_exposed_t1 = bed_after_t1["exposed_count"]

    # Turn 2: asks_meaning — no new sentence generated, just explanation.
    await agent.ainvoke(
        {"messages": [HumanMessage(content="cat 是什么意思")]},
        config={"configurable": {"thread_id": "expose"}},
    )

    cat_after_t2 = await setup.stats.get("cat")
    bed_after_t2 = await setup.stats.get("bed")
    assert cat_after_t2["exposed_count"] == cat_exposed_t1, (
        f"cat.exposed_count must not change on asks_meaning turn "
        f"(was {cat_exposed_t1}, now {cat_after_t2['exposed_count']}) — I1"
    )
    assert bed_after_t2["exposed_count"] == bed_exposed_t1, (
        f"bed.exposed_count must not change on asks_meaning turn "
        f"(was {bed_exposed_t1}, now {bed_after_t2['exposed_count']}) — I1"
    )


@pytest.mark.asyncio
async def test_asks_meaning_non_ket_word_does_not_deduct(setup):
    """R4 / I2: asking about a NON-KET word (e.g. a proper noun like
    'Beijing') must NOT deduct mastery — apply_mastery_updates guards with
    `is_ket_word`. The CSV in this fixture does not contain 'beijing'."""
    llm = _mock_llm_seq(
        intent_resp=IntentClassification(intent="asks_meaning", asked_word="beijing"),
        meaning_resp=WordMeaning(meaning="北京"),
        sentence_texts=("The cat is on the bed.",),
    )
    agent = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})

    # Turn 1: establish a sentence.
    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "nonket"}},
    )
    # Seed a 'beijing' stats row so we can detect any deduction.
    await setup.stats.apply_delta("beijing", delta=0, exposed=False)
    beijing_before = await setup.stats.get("beijing")
    assert beijing_before is not None
    score_before = beijing_before["mastery_score"]
    wrong_before = beijing_before["wrong_count"]

    # Turn 2: ask about a non-KET word.
    await agent.ainvoke(
        {"messages": [HumanMessage(content="beijing 是什么意思")]},
        config={"configurable": {"thread_id": "nonket"}},
    )

    beijing_after = await setup.stats.get("beijing")
    assert beijing_after["mastery_score"] == score_before, (
        "asking about a non-KET word must not change mastery_score (I2)"
    )
    assert beijing_after["wrong_count"] == wrong_before, (
        "asking about a non-KET word must not change wrong_count (I2)"
    )


@pytest.mark.asyncio
async def test_agent_aclose_drains_background_tasks(setup):
    """I3: build_agent attaches the KETPartnerAgent as `.agent` on the graph,
    and `aclose()` drains in-flight background summary tasks instead of
    silently dropping them on shutdown.
    """
    llm = _mock_llm(intent_resp=None, sentence_text="The cat is on the bed.")
    graph = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})

    # The agent instance must be reachable from the graph (per the I3 fix).
    agent_instance = getattr(graph, "agent", None)
    assert agent_instance is not None, "graph.agent must be attached for shutdown drain (I3)"

    # Manually schedule a background task that mutates a flag, then ensure
    # aclose() actually awaits it.
    completed = {"value": False}

    async def fake_summary():
        await asyncio.sleep(0.05)
        completed["value"] = True

    task = asyncio.create_task(fake_summary())
    agent_instance._bg_tasks.add(task)
    task.add_done_callback(agent_instance._bg_tasks.discard)

    await agent_instance.aclose(timeout=2.0)
    assert completed["value"] is True, "aclose() must await in-flight background tasks (I3)"
    # After aclose, the bg set should be empty (task completed + removed via callback).
    assert not agent_instance._bg_tasks, "bg_tasks should be empty after aclose()"


@pytest.mark.asyncio
async def test_agent_aclose_no_tasks_is_noop(setup):
    """I3: aclose() must be safe to call when there are no background tasks."""
    llm = _mock_llm(intent_resp=None, sentence_text="The cat is on the bed.")
    graph = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})
    agent_instance = graph.agent
    # No tasks scheduled — should return quickly without error.
    await agent_instance.aclose()
    assert agent_instance._bg_tasks == set()
