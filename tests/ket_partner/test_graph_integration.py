import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.messages import AIMessage, HumanMessage

from flow.ket_partner.agent import build_agent
from flow.ket_partner.db import init_db
from flow.ket_partner.input_classifier import IntentClassification
from flow.ket_partner.sentence_naturalness import NaturalnessResult
from flow.ket_partner.translation_evaluator import TranslationEval
from flow.ket_partner.word_meaning_lookup import SentenceTranslation, WordMeaning


def _mock_llm(intent_resp, eval_resp=None, meaning_resp=None, sentence_translation_resp=None, naturalness_resp=None, sentence_text="The cat is on the bed."):
    llm = MagicMock()
    responses = {
        IntentClassification: intent_resp,
        # Default naturalness to "ok" — most tests don't care about it and
        # would otherwise NoOp through the retry loop with None.
        NaturalnessResult: naturalness_resp if naturalness_resp is not None else NaturalnessResult(ok=True, reason=""),
    }
    if eval_resp:
        responses[TranslationEval] = eval_resp
    if meaning_resp:
        responses[WordMeaning] = meaning_resp
    if sentence_translation_resp:
        responses[SentenceTranslation] = sentence_translation_resp

    def structured(schema, **kwargs):
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
    csv_text = "word,part_of_speech,topic\ncat,n,Animals\ndog,n,Animals\nbed,n,Home\nthe,det,\non,prep,\nin,prep,\nbox,n,\nis,v,\n"
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
    # Require both the translation prompt and a real non-empty sentence.
    assert "请把这句译成中文" in ai_msg
    assert "cat" in ai_msg.lower()
    # The quoted sentence body must not be empty.
    assert '""' not in ai_msg


@pytest.mark.asyncio
async def test_correct_translation_increments_mastery(setup):
    llm = _mock_llm(
        intent_resp=IntentClassification(intent="translation", asked_word=None),
        eval_resp=TranslationEval(correct_translation="猫在床上", wrong_words=[]),
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
    from flow.ket_partner.translation_evaluator import WrongWord
    llm = _mock_llm(
        intent_resp=IntentClassification(intent="translation", asked_word=None),
        eval_resp=TranslationEval(
            correct_translation="猫在床上",
            wrong_words=[WrongWord(word="cat", kid_translation="狗", correct_translation="猫")],
        ),
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
        sentence_translation_resp=SentenceTranslation(translation="猫在床上"),
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
    sentence_translation_resp=None,
    naturalness_resp=None,
    sentence_texts=("The cat is on the bed.",),
):
    """Like _mock_llm but cycles through `sentence_texts` on successive
    .bind().ainvoke calls (one per generate_sentence call)."""
    llm = MagicMock()
    responses = {
        IntentClassification: intent_resp,
        NaturalnessResult: naturalness_resp if naturalness_resp is not None else NaturalnessResult(ok=True, reason=""),
    }
    if eval_resp:
        responses[TranslationEval] = eval_resp
    if meaning_resp:
        responses[WordMeaning] = meaning_resp
    if sentence_translation_resp:
        responses[SentenceTranslation] = sentence_translation_resp

    def structured(schema, **kwargs):
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
        eval_resp=TranslationEval(correct_translation="猫在床上", wrong_words=[]),
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
    assert "请把这句译成中文" in ai1, "translation prompt must be present"
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

    # Verify both turns' user inputs were logged to conversation_log.
    # (Latent bug: with REPLACE reducer and AI-only return dicts, messages[-2]
    # in persist_turn_node saw no HumanMessage, so user inputs were never logged.)
    async with setup.log._db.execute(
        "SELECT role, content FROM conversation_log WHERE role = 'user' ORDER BY id"
    ) as cur:
        user_rows = await cur.fetchall()
    user_contents = {r[1] for r in user_rows}
    assert "hi" in user_contents, "turn-1 user input must be logged"
    assert "猫在床上" in user_contents, "turn-2 user input must be logged"


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


@pytest.mark.asyncio
async def test_wrong_word_rendered_and_deducted(setup):
    """When the kid mistranslates a word (e.g. preposition "in" → 上), the
    unified wrong_words entry must render the correct translation AND the
    word must be deducted from mastery (post schema-merge: function words
    now affect mastery — that was the bug motivating the merge).
    """
    from flow.ket_partner.translation_evaluator import WrongWord
    llm = _mock_llm(
        intent_resp=IntentClassification(intent="translation", asked_word=None),
        eval_resp=TranslationEval(
            correct_translation="猫在盒子里",
            wrong_words=[WrongWord(
                word="in",
                kid_translation="上",
                correct_translation="里",
            )],
        ),
        sentence_text="The cat is in the box.",
    )
    agent = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})

    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "fw"}},
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="猫在盒子上")]},
        config={"configurable": {"thread_id": "fw"}},
    )
    ai_msg = result["messages"][-1].content

    # The full correct translation must be shown first.
    assert "正确翻译：猫在盒子里" in ai_msg, "correct_translation must be rendered before the wrong-word list"
    # Then the wrong-word section with per-word correction.
    assert "你的翻译有误:" in ai_msg, "wrong_words must render under 你的翻译有误:"
    assert "in" in ai_msg
    assert "里" in ai_msg

    # After the merge, prepositions in wrong_words DO affect mastery.
    in_stats = await setup.stats.get("in")
    assert in_stats is not None, "in must be tracked in vocab_stats"
    assert in_stats["wrong_count"] == 1, "in's wrong_count must increment (post-merge)"


@pytest.mark.asyncio
async def test_evaluate_node_dedupes_duplicate_wrong_words(setup):
    """Regression: the LLM emitted two entries for the same word with
    contradictory meanings. Without dedup the UI rendered back-to-back
    duplicate lines. Dedup by word in evaluate_translation_node so each
    word appears at most once — first entry wins.
    """
    from flow.ket_partner.translation_evaluator import WrongWord
    llm = _mock_llm(
        intent_resp=IntentClassification(intent="translation", asked_word=None),
        eval_resp=TranslationEval(
            correct_translation="猫在盒子里",
            wrong_words=[
                WrongWord(word="in", kid_translation="上", correct_translation="里"),
                WrongWord(word="in", kid_translation="上", correct_translation="内部"),
            ],
        ),
        sentence_text="The cat is in the box.",
    )
    agent = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})

    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "dup"}},
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="猫在盒子上")]},
        config={"configurable": {"thread_id": "dup"}},
    )
    ai_msg = result["messages"][-1].content

    # `in 的意思是：` must appear exactly once, not twice.
    assert ai_msg.count("in 的意思是：") == 1, (
        "duplicate wrong_words entries for the same word must be deduped"
    )


@pytest.mark.asyncio
async def test_evaluate_node_drops_non_ket_words(setup):
    """Regression: when generate_sentence_node exhausts retries and accepts
    a draft with non_ket_words, the displayed sentence contains non-KET
    words. If the LLM (correctly or incorrectly) flags one of those as
    wrong, evaluate_translation_node must drop it before it reaches
    vocab_stats — otherwise apply_delta creates a phantom stats row for
    a word that isn't in KET vocabulary.
    """
    from flow.ket_partner.translation_evaluator import WrongWord
    # LLM flags both a KET word (in) and a non-KET word (xyzzy).
    llm = _mock_llm(
        intent_resp=IntentClassification(intent="translation", asked_word=None),
        eval_resp=TranslationEval(
            correct_translation="猫在盒子里",
            wrong_words=[
                WrongWord(word="in", kid_translation="上", correct_translation="里"),
                WrongWord(word="xyzzy", kid_translation="?", correct_translation="?"),
            ],
        ),
        sentence_text="The cat is in the box.",
    )
    agent = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})

    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "nonket-1"}},
    )
    await agent.ainvoke(
        {"messages": [HumanMessage(content="猫在盒子")]},
        config={"configurable": {"thread_id": "nonket-1"}},
    )

    # The KET wrong word must be tracked normally.
    in_stats = await setup.stats.get("in")
    assert in_stats is not None and in_stats["wrong_count"] == 1

    # The non-KET word must NOT have a stats row.
    xyzzy_stats = await setup.stats.get("xyzzy")
    assert xyzzy_stats is None, "non-KET words must be filtered out before DB write"


@pytest.mark.asyncio
async def test_displayed_sentence_words_are_tracked_not_stale_retry(setup, monkeypatch):
    """Regression: when validation fails after all retries, the agent must
    track exposure for words in the FINAL displayed sentence, not the
    previous retry's sentence.

    Before the fix, the for-loop in generate_sentence_node would regenerate
    `sentence` after a failed validation but leave `result` holding the
    previous validation. The kid saw the new sentence but exposure was
    tracked for words that weren't in it.

    Note: under the current acceptance policy, sentences with ≤1 non-KET
    word are accepted immediately. To force retries through exhaustion we
    use 2 non-KET words per draft.
    """
    from flow.ket_partner import agent as agent_module
    from flow.ket_partner.sentence_validator import ValidationResult

    # Sentence sequence: retry 1 → "alpha beta"; retry 2 (final) → "gamma delta".
    # Both have 2 non-KET words so the retry loop keeps regenerating.
    sentence_seq = iter(["The alpha beta.", "The gamma delta."])

    async def fake_generate(*a, **kw):
        return next(sentence_seq)

    async def fake_validate(sentence, repos):
        if "alpha" in sentence:
            return ValidationResult(ok=False, words_used=[], non_ket_words=["alpha", "beta"])
        return ValidationResult(ok=False, words_used=[], non_ket_words=["gamma", "delta"])

    monkeypatch.setattr(agent_module, "generate_sentence", fake_generate)
    monkeypatch.setattr(agent_module, "validate_sentence", fake_validate)

    llm = _mock_llm(intent_resp=None, sentence_text="ignored")
    graph = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})

    # Force a small retry limit so the test doesn't loop forever.
    original_limit = graph.agent.config.validate_retry_limit
    graph.agent.config.validate_retry_limit = 1

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "stale"}},
    )
    graph.agent.config.validate_retry_limit = original_limit

    ai_msg = result["messages"][-1].content
    # The kid saw the FINAL sentence ("gamma delta"), not the first ("alpha beta").
    assert "gamma" in ai_msg, "kid must see the final regenerated sentence"
    assert "alpha" not in ai_msg, "kid must NOT see the abandoned first draft"


@pytest.mark.asyncio
async def test_generate_node_regens_when_more_than_one_non_ket(setup, monkeypatch):
    """Acceptance policy: >1 non-KET word must trigger a full regen. (The
    rewrite path was removed — only regen now.)"""
    from flow.ket_partner import agent as agent_module
    from flow.ket_partner.sentence_validator import ValidationResult

    call_log = {"generate": 0}

    async def fake_generate(*a, **kw):
        call_log["generate"] += 1
        return "alpha bravo charlie delta echo"

    async def fake_validate(sentence, repos):
        return ValidationResult(
            ok=False, words_used=[], non_ket_words=["alpha", "bravo", "charlie", "delta"],
        )

    monkeypatch.setattr(agent_module, "generate_sentence", fake_generate)
    monkeypatch.setattr(agent_module, "validate_sentence", fake_validate)

    llm = _mock_llm(intent_resp=None, sentence_text="ignored")
    graph = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})
    graph.agent.config.validate_retry_limit = 2

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "regen"}},
    )
    # >1 non-KET must keep regenerating until retries exhaust.
    assert call_log["generate"] >= 3, (
        f"expected ≥3 generate calls (1 initial + 2 retries), got {call_log['generate']}"
    )


@pytest.mark.asyncio
async def test_generate_node_regen_when_sentence_is_duplicate(setup, monkeypatch):
    """Regression: 'The cat likes to dance in the rain.' appeared 3 times
    in a row during manual testing. The retry loop must detect an exact
    match against recent sentences and force a full regen."""
    from flow.ket_partner import agent as agent_module
    from flow.ket_partner.sentence_validator import ValidationResult

    call_log = {"generate": 0}
    # First call (initial draft) returns the duplicate; second call (regen
    # after detection) returns a fresh sentence.
    generate_seq = iter(["The cat likes to dance.", "The dog likes to swim."])

    async def fake_generate(*a, **kw):
        call_log["generate"] += 1
        return next(generate_seq)

    async def fake_validate(sentence, repos):
        # Always passes KET validation — duplicate is the ONLY reason to retry.
        return ValidationResult(ok=True, words_used=["the", "cat", "likes"], non_ket_words=[])

    monkeypatch.setattr(agent_module, "generate_sentence", fake_generate)
    monkeypatch.setattr(agent_module, "validate_sentence", fake_validate)

    llm = _mock_llm(intent_resp=None, sentence_text="ignored")
    graph = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})
    graph.agent.config.validate_retry_limit = 2

    # Seed recent_sentences with the duplicate so the first draft collides.
    graph.agent._recent_sentences.append("The cat likes to dance.")

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "dedup"}},
    )
    ai_msg = result["messages"][-1].content
    # The kid must see the fresh sentence, not the duplicate.
    assert "dog" in ai_msg, "duplicate must trigger regen; kid must see the fresh sentence"
    assert "The cat likes to dance." not in ai_msg
    assert call_log["generate"] == 2, "initial draft + one regen after duplicate detection"


@pytest.mark.asyncio
async def test_generate_node_passes_non_ket_words_to_regen(setup, monkeypatch):
    """When a regen is triggered by >1 non-KET words, the next generate call
    must list those words in `avoid_non_ket_words`. Without this, the LLM
    keeps producing the same non-KET word on every retry (e.g., "blocks" /
    "tower" for target "build")."""
    from flow.ket_partner import agent as agent_module
    from flow.ket_partner.sentence_validator import ValidationResult

    # Record the kwarg on every call. The first call should have an empty
    # list; the second call must contain the first attempt's non-KET words.
    captured_kwargs: list = []

    async def fake_generate(*a, **kw):
        # Snapshot the list — agent.py mutates it across retries, so the
        # reference would otherwise show the final state on every entry.
        captured_kwargs.append(list(kw.get("avoid_non_ket_words") or []))
        # Always returns the same body — validate_sentence is what flags it.
        return "Let us build a tall tower with blocks."

    async def fake_validate(sentence, repos):
        # Always reports the same two non-KET words so the retry loop keeps
        # regenerating with an ever-growing avoid list.
        return ValidationResult(
            ok=False,
            words_used=["let", "us", "build", "a", "tall"],
            non_ket_words=["tower", "blocks"],
        )

    monkeypatch.setattr(agent_module, "generate_sentence", fake_generate)
    monkeypatch.setattr(agent_module, "validate_sentence", fake_validate)

    llm = _mock_llm(intent_resp=None, sentence_text="ignored")
    graph = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})
    graph.agent.config.validate_retry_limit = 2

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "non-ket-avoid"}},
    )

    # Initial call has no prior non-KET words to avoid.
    assert captured_kwargs[0] == [], (
        f"first generate call must start with empty avoid list, got {captured_kwargs[0]!r}"
    )
    # The retry call must contain both non-KET words from the first attempt.
    assert "tower" in captured_kwargs[1] and "blocks" in captured_kwargs[1], (
        f"regen call must list prior non-KET words, got {captured_kwargs[1]!r}"
    )


@pytest.mark.asyncio
async def test_generate_node_passes_duplicate_sentence_to_regen(setup, monkeypatch):
    """When a regen is triggered by an exact-match duplicate, the next
    generate call must surface the offending sentence verbatim as
    `duplicate_sentence` so the LLM gets an explicit "do not output that
    exact sentence" callout — not just the soft avoid_sentences list."""
    from flow.ket_partner import agent as agent_module
    from flow.ket_partner.sentence_validator import ValidationResult

    captured_kwargs: list = []
    duplicate = "The cat likes to dance."
    # First call returns the duplicate; second call returns a fresh sentence.
    generate_seq = iter([duplicate, "The dog likes to swim."])

    async def fake_generate(*a, **kw):
        captured_kwargs.append(kw.get("duplicate_sentence", ""))
        return next(generate_seq)

    async def fake_validate(sentence, repos):
        # Always passes KET validation — duplicate is the ONLY reason to retry.
        return ValidationResult(ok=True, words_used=["the", "cat", "likes"], non_ket_words=[])

    monkeypatch.setattr(agent_module, "generate_sentence", fake_generate)
    monkeypatch.setattr(agent_module, "validate_sentence", fake_validate)

    llm = _mock_llm(intent_resp=None, sentence_text="ignored")
    graph = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})
    graph.agent.config.validate_retry_limit = 2

    # Seed recent_sentences with the duplicate so the first draft collides.
    graph.agent._recent_sentences.append(duplicate)

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "dup-kwarg"}},
    )

    # Initial call has no duplicate to call out (validation hasn't run yet).
    assert captured_kwargs[0] == "", (
        f"first generate call must have empty duplicate_sentence, got {captured_kwargs[0]!r}"
    )
    # The retry call must surface the offending sentence verbatim.
    assert captured_kwargs[1] == duplicate, (
        f"regen call must pass duplicate_sentence={duplicate!r}, got {captured_kwargs[1]!r}"
    )


@pytest.mark.asyncio
async def test_generate_node_scaffolding_passes_last_n_sentences(setup, monkeypatch):
    """The `recent_scaffolding` argument passed to generate_sentence must
    reflect the last N SENTENCES' worth of words (flattened), not just the
    last N words of one sentence (the pre-fix bug)."""
    from flow.ket_partner import agent as agent_module
    from flow.ket_partner.sentence_validator import ValidationResult

    captured_args = {}

    async def fake_generate(*a, **kw):
        captured_args["recent_scaffolding"] = kw.get("recent_scaffolding")
        captured_args["avoid_sentences"] = kw.get("avoid_sentences")
        return "fresh sentence words here"

    async def fake_validate(sentence, repos):
        return ValidationResult(ok=True, words_used=["fresh", "sentence", "words"], non_ket_words=[])

    monkeypatch.setattr(agent_module, "generate_sentence", fake_generate)
    monkeypatch.setattr(agent_module, "validate_sentence", fake_validate)

    llm = _mock_llm(intent_resp=None, sentence_text="ignored")
    graph = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})
    graph.agent.config.validate_retry_limit = 1
    graph.agent.config.variety.recent_window = 3

    # Seed three prior sentences' worth of words.
    graph.agent._recent_scaffolding = [["the", "cat", "runs"], ["the", "dog", "jumps"], ["a", "fish", "swims"]]
    graph.agent._recent_sentences = ["The cat runs.", "The dog jumps.", "A fish swims."]

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "window"}},
    )
    # All 9 words from the 3 prior sentences must reach the prompt.
    assert set(captured_args["recent_scaffolding"]) == {"the", "cat", "runs", "dog", "jumps", "a", "fish", "swims"}, (
        "recent_scaffolding must include ALL words from the last N sentences, "
        "not just the last N words of the most recent sentence"
    )
    assert set(captured_args["avoid_sentences"]) == {"The cat runs.", "The dog jumps.", "A fish swims."}


@pytest.mark.asyncio
async def test_single_non_ket_word_accepted_with_annotation(setup, monkeypatch):
    """Acceptance policy: a sentence with exactly ONE non-KET word is
    accepted (no regen) and that word's context meaning is appended so
    the kid can still translate the sentence.
    """
    from flow.ket_partner import agent as agent_module
    from flow.ket_partner.sentence_naturalness import NaturalnessResult
    from flow.ket_partner.sentence_validator import ValidationResult

    async def fake_generate(*a, **kw):
        return "The cat splashes in the water."

    async def fake_validate(sentence, repos):
        # 1 non-KET word (splashes); rest are KET.
        return ValidationResult(
            ok=False,
            words_used=["the", "cat", "in", "the", "water"],
            non_ket_words=["splashes"],
        )

    async def fake_naturalness(llm, sentence, age=8):
        return NaturalnessResult(ok=True, reason="")

    lookup_calls = {"count": 0}

    async def fake_lookup_meanings(llm, sentence, words):
        lookup_calls["count"] += 1
        # Should be called once with the single non-KET word.
        assert words == ["splashes"]
        return [{"word": "splashes", "meaning": "溅水"}]

    monkeypatch.setattr(agent_module, "generate_sentence", fake_generate)
    monkeypatch.setattr(agent_module, "validate_sentence", fake_validate)
    monkeypatch.setattr(agent_module, "check_naturalness", fake_naturalness)
    monkeypatch.setattr(agent_module, "lookup_word_meanings", fake_lookup_meanings)

    llm = _mock_llm(intent_resp=None, sentence_text="ignored")
    graph = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "ann-1"}},
    )
    ai_msg = result["messages"][-1].content
    # Kid sees the original sentence (with the non-KET word).
    assert "splashes" in ai_msg, "single non-KET word must be accepted as-is"
    # And the annotation appears below.
    assert "splashes 的意思是：溅水" in ai_msg, "non-KET word must be annotated with context meaning"
    assert lookup_calls["count"] == 1, "lookup_word_meanings must be called once on accept path"


@pytest.mark.asyncio
async def test_many_non_ket_after_exhaustion_accepts_with_all_annotations(setup, monkeypatch):
    """After retries exhaust with >1 non-KET words still present, the agent
    must accept the sentence and annotate ALL non-KET words so the kid has
    a chance to translate.
    """
    from flow.ket_partner import agent as agent_module
    from flow.ket_partner.sentence_validator import ValidationResult

    async def fake_generate(*a, **kw):
        return "alpha bravo the cat"

    async def fake_validate(sentence, repos):
        # Always >1 non-KET — forces the retry loop to exhaust.
        return ValidationResult(
            ok=False, words_used=["the", "cat"], non_ket_words=["alpha", "bravo"],
        )

    async def fake_lookup_meanings(llm, sentence, words):
        return [{"word": w, "meaning": f"<{w}的释义>"} for w in words]

    monkeypatch.setattr(agent_module, "generate_sentence", fake_generate)
    monkeypatch.setattr(agent_module, "validate_sentence", fake_validate)
    monkeypatch.setattr(agent_module, "lookup_word_meanings", fake_lookup_meanings)

    llm = _mock_llm(intent_resp=None, sentence_text="ignored")
    graph = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})
    graph.agent.config.validate_retry_limit = 2

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "ann-many"}},
    )
    ai_msg = result["messages"][-1].content
    # Both non-KET words must be annotated.
    assert "alpha 的意思是：<alpha的释义>" in ai_msg, "first non-KET word must be annotated"
    assert "bravo 的意思是：<bravo的释义>" in ai_msg, "second non-KET word must be annotated"


@pytest.mark.asyncio
async def test_all_ket_sentence_has_no_annotations(setup, monkeypatch):
    """A clean all-KET sentence must NOT trigger the lookup or render any
    annotation lines."""
    from flow.ket_partner import agent as agent_module
    from flow.ket_partner.sentence_naturalness import NaturalnessResult
    from flow.ket_partner.sentence_validator import ValidationResult

    async def fake_generate(*a, **kw):
        return "The cat sleeps."

    async def fake_validate(sentence, repos):
        return ValidationResult(ok=True, words_used=["the", "cat", "sleeps"], non_ket_words=[])

    async def fake_naturalness(llm, sentence, age=8):
        return NaturalnessResult(ok=True, reason="")

    lookup_calls = {"count": 0}

    async def fake_lookup(*a, **kw):
        lookup_calls["count"] += 1
        return []

    monkeypatch.setattr(agent_module, "generate_sentence", fake_generate)
    monkeypatch.setattr(agent_module, "validate_sentence", fake_validate)
    monkeypatch.setattr(agent_module, "check_naturalness", fake_naturalness)
    monkeypatch.setattr(agent_module, "lookup_word_meanings", fake_lookup)

    llm = _mock_llm(intent_resp=None, sentence_text="ignored")
    graph = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "ann-none"}},
    )
    ai_msg = result["messages"][-1].content
    assert "的意思是" not in ai_msg, "no annotation lines for all-KET sentence"
    assert lookup_calls["count"] == 0, "lookup must NOT run when there are no non-KET words"


@pytest.mark.asyncio
async def test_restart_does_not_leak_prior_session_unfinished_sentence(setup):
    """Regression: a kid who exits mid-sentence must NOT see the explanation
    of that unfinished sentence when they restart. The session_start marker
    written on REPL startup causes last_ai_message() to return None, so
    init_state skips restoration, route_after_init goes straight to
    select_target_word, and classify_intent never runs against the stale
    sentence.
    """
    # Session 1: simulate one full turn — kid says "hi", AI emits a sentence.
    llm1 = _mock_llm(intent_resp=None, sentence_text="The cat is on the bed.")
    agent1 = await build_agent(llm_flash=llm1, llm_smart=llm1, repos=setup, info={"nickname_kid": "t", "age": 8})
    await agent1.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "s1"}},
    )
    # Confirm session 1 left an AI row that COULD be restored.
    prior = await setup.log.last_ai_message()
    assert prior is not None and "cat" in prior["content"]

    # REPL exits and restarts — main.py (via autonomous) writes the marker.
    await setup.log.append_session_start()
    # Now last_ai_message must return None — the prior AI is "before marker".
    assert await setup.log.last_ai_message() is None, (
        "session_start marker must hide AI rows from prior sessions"
    )

    # Session 2: kid's first input "hi" — without the fix, classify_intent
    # would run against the stale sentence and likely default to translation,
    # leaking the answer via "正确翻译" / "你的翻译有误".
    llm2 = _mock_llm(
        intent_resp=IntentClassification(intent="translation", asked_word=None),
        sentence_text="The dog is on the bed.",
    )
    agent2 = await build_agent(llm_flash=llm2, llm_smart=llm2, repos=setup, info={"nickname_kid": "t", "age": 8})
    result = await agent2.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "s2"}},
    )
    ai_msg = result["messages"][-1].content
    # The session-2 sentence must be the freshly generated one (dog), and
    # must NOT include any evaluation/explanation of the prior session's
    # sentence (cat).
    assert "dog" in ai_msg, "session 2 must show a freshly generated sentence"
    assert "正确翻译" not in ai_msg, "must NOT evaluate the prior session's unfinished sentence"
    assert "你的翻译有误" not in ai_msg, "must NOT evaluate the prior session's unfinished sentence"
    assert "猫" not in ai_msg, "must NOT leak the Chinese meaning of the prior session's target word"


@pytest.mark.asyncio
async def test_naturalness_fail_triggers_regen_with_hint(setup, monkeypatch):
    """Regression: 'The cold ice cream makes my nose move.' passes KET
    validation but is semantically nonsensical. The retry loop must call
    check_naturalness after KET+dedup pass, and on rejection, regenerate
    with the rejection reason fed back into the prompt.
    """
    from flow.ket_partner import agent as agent_module
    from flow.ket_partner.sentence_naturalness import NaturalnessResult
    from flow.ket_partner.sentence_validator import ValidationResult

    # First draft is nonsensical; second (regen) is natural.
    generate_seq = iter([
        "The cold ice cream makes my nose move.",
        "The cold ice cream makes my teeth hurt.",
    ])

    captured_hints = []

    async def fake_generate(*a, **kw):
        # Record the naturalness_hint so we can verify the reason was fed back.
        captured_hints.append(kw.get("naturalness_hint", ""))
        return next(generate_seq)

    async def fake_validate(sentence, repos):
        return ValidationResult(ok=True, words_used=["the", "cold", "ice", "cream", "makes", "my", "nose", "move"], non_ket_words=[])

    # First call rejects (nonsense), second accepts (natural).
    nat_seq = iter([
        NaturalnessResult(ok=False, reason="ice cream does not make noses move"),
        NaturalnessResult(ok=True, reason=""),
    ])

    async def fake_naturalness(llm, sentence, age=8):
        return next(nat_seq)

    monkeypatch.setattr(agent_module, "generate_sentence", fake_generate)
    monkeypatch.setattr(agent_module, "validate_sentence", fake_validate)
    monkeypatch.setattr(agent_module, "check_naturalness", fake_naturalness)

    llm = _mock_llm(intent_resp=None, sentence_text="ignored")
    graph = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})
    graph.agent.config.validate_retry_limit = 3

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "nat"}},
    )
    ai_msg = result["messages"][-1].content

    # The kid must see the natural regenerated sentence, not the nonsense one.
    assert "teeth" in ai_msg, "naturalness rejection must trigger regen with a better sentence"
    assert "nose move" not in ai_msg, "the nonsensical first draft must not be shown"

    # The hint must have been propagated to the second generate_sentence call.
    assert len(captured_hints) >= 2, "generate_sentence must be called twice (initial + regen)"
    assert captured_hints[1] == "ice cream does not make noses move", (
        "the naturalness rejection reason must be fed back as naturalness_hint"
    )


@pytest.mark.asyncio
async def test_naturalness_check_skipped_when_ket_validation_fails(setup, monkeypatch):
    """The naturalness LLM call is expensive — it must only run on candidates
    that survived the cheap KET+dedup gates. When KET validation fails, the
    retry path is the existing rewrite/regen branch and check_naturalness
    must NOT be invoked.
    """
    from flow.ket_partner import agent as agent_module
    from flow.ket_partner.sentence_validator import ValidationResult

    naturalness_calls = {"count": 0}

    async def fake_generate(*a, **kw):
        return "alpha bravo charlie"  # all non-KET

    async def fake_validate(sentence, repos):
        # Always fails KET — never passes the cheap gate.
        return ValidationResult(ok=False, words_used=[], non_ket_words=["alpha", "bravo", "charlie"])

    async def fake_naturalness(llm, sentence, age=8):
        naturalness_calls["count"] += 1
        return NaturalnessResult(ok=True, reason="")

    monkeypatch.setattr(agent_module, "generate_sentence", fake_generate)
    monkeypatch.setattr(agent_module, "validate_sentence", fake_validate)
    monkeypatch.setattr(agent_module, "check_naturalness", fake_naturalness)

    llm = _mock_llm(intent_resp=None, sentence_text="ignored")
    graph = await build_agent(llm_flash=llm, llm_smart=llm, repos=setup, info={"nickname_kid": "t", "age": 8})
    graph.agent.config.validate_retry_limit = 2

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "skip-nat"}},
    )
    assert naturalness_calls["count"] == 0, (
        "check_naturalness must NOT be called when KET validation fails (cost gate)"
    )


@pytest.mark.asyncio
async def test_generate_node_marks_target_distinct_from_scaffolding(setup, monkeypatch):
    """Target word must end up with status='learning'; scaffolding-only words
    in the same sentence must end up with status='exposed'. Without the
    is_target flag, all words would be 'learning' and the refill pool would
    fill up with passive words — defeating the new design.

    Note: target-word selection uses ORDER BY RANDOM() in production
    (vocab_selector._pick_new_word → db.words_in_topic_without_stats), so we
    monkeypatch _pick_new_word to deterministically return 'cat'. The mock
    sentence "The cat is on the bed." includes both 'cat' (target) and 'bed'
    (scaffolding) so both branches of the assertion are exercised."""
    from flow.ket_partner import vocab_selector

    async def fake_pick_new_word(repos, profile):
        return "cat"

    monkeypatch.setattr(vocab_selector, "_pick_new_word", fake_pick_new_word)

    llm = _mock_llm(
        intent_resp=None,
        sentence_text="The cat is on the bed.",
    )
    agent = await build_agent(
        llm_flash=llm,
        llm_smart=llm,
        repos=setup,
        info={"nickname_kid": "t", "age": 8},
    )
    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "target-mark"}},
    )

    # 'cat' is the target_word (forced by the monkeypatch above).
    cat = await setup.stats.get("cat")
    assert cat is not None, "target word must have a stats row"
    assert cat["status"] == "learning", (
        f"target word must be 'learning' (got {cat['status']!r})"
    )
    # Scaffolding words from the sentence — none of them were the target.
    for word in ["bed"]:
        row = await setup.stats.get(word)
        assert row is not None, f"scaffolding word {word!r} must have a stats row"
        assert row["status"] == "exposed", (
            f"scaffolding word {word!r} must be 'exposed' (got {row['status']!r})"
        )


@pytest.mark.asyncio
async def test_generate_node_handles_multi_word_target(temp_db_path, monkeypatch):
    """Regression: validator's [A-Za-z']+ tokenizer can't recognize multi-word
    or alphanumeric targets like 'MP3 player' — only the trailing 'player'
    lands in words_used (the 'MP3' token is silently dropped as a proper noun).
    The agent must add the target itself so stats tracking marks it as
    'learning' and downstream filters (which use last_sentence_words) still
    see the right lexical unit."""
    from flow.ket_partner import agent as agent_module
    from flow.ket_partner import vocab_selector
    from flow.ket_partner.sentence_naturalness import NaturalnessResult

    csv_text = (
        "word,part_of_speech,topic\n"
        "MP3 player,n,Technology\n"
        "player,n,Sport\n"
        "she,pron,\n"
        "listen,v,Action\n"
        "to,prep,\n"
        "music,n,Art\n"
        "on,prep,\n"
        "her,det,\n"
        "new,adj,\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = f.name
    repos = await init_db(temp_db_path, csv_path=csv_path)

    sentence = "She listens to music on her new MP3 player."

    async def fake_generate(*a, **kw):
        return sentence

    async def fake_pick_new_word(repos, profile):
        return "MP3 player"

    async def fake_naturalness(*a, **kw):
        return NaturalnessResult(ok=True, reason="")

    monkeypatch.setattr(agent_module, "generate_sentence", fake_generate)
    monkeypatch.setattr(agent_module, "check_naturalness", fake_naturalness)
    monkeypatch.setattr(vocab_selector, "_pick_new_word", fake_pick_new_word)

    llm = _mock_llm(intent_resp=None)
    agent = await build_agent(
        llm_flash=llm,
        llm_smart=llm,
        repos=repos,
        info={"nickname_kid": "t", "age": 8},
    )
    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "mp3-target"}},
    )

    # The target phrase must be tracked as 'learning' — without the fix the
    # validator only sees 'player' and the is_target check never fires.
    target_stats = await repos.stats.get("MP3 player")
    assert target_stats is not None, "multi-word target must have a stats row"
    assert target_stats["status"] == "learning", (
        f"target must be status='learning' (got {target_stats['status']!r})"
    )
    # The target's trailing constituent must NOT be tracked as scaffolding.
    # In this sentence "player" only appears as part of "MP3 player", so
    # tracking it standalone would double-count the same lexical unit.
    player_stats = await repos.stats.get("player")
    assert player_stats is None, (
        f"'player' must not have a stats row (it's part of the target phrase); "
        f"got {player_stats!r}"
    )
    await repos.close()


# ---------------------------------------------------------------------------
# Task 5 (exposed-status plan): mastered decay + passive graduation +
# learning_count exclusion. These tests lock in the design's load-bearing
# claims end-to-end through apply_delta — the same API nodes.apply_mastery_updates
# uses. No production code changes; these verify existing behavior.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mastered_word_absorbs_one_wrong_answer(setup):
    """宽容 policy: mastery 3→2 keeps 'mastered' status. Single mistake
    must not flapping the word back to 'learning'."""
    repos = setup
    # Drive 'cat' to mastery=3 via target exposure + 2 correct deltas
    await repos.stats.apply_delta("cat", delta=1, exposed=True, is_target=True)
    await repos.stats.apply_delta("cat", delta=1, is_target=True)
    await repos.stats.apply_delta("cat", delta=1, is_target=True)
    assert (await repos.stats.get("cat"))["status"] == "mastered"

    # One wrong answer drops mastery to 2 — status must stay mastered
    await repos.stats.apply_delta("cat", delta=-1)
    cat = await repos.stats.get("cat")
    assert cat["mastery_score"] == 2
    assert cat["status"] == "mastered"


@pytest.mark.asyncio
async def test_mastered_word_demotes_after_two_consecutive_wrong(setup):
    """3→2 (absorbed) → 1 demotes to 'learning'. Per spec: 连续答错掉到1."""
    repos = setup
    await repos.stats.apply_delta("cat", delta=1, exposed=True, is_target=True)
    await repos.stats.apply_delta("cat", delta=1, is_target=True)
    await repos.stats.apply_delta("cat", delta=1, is_target=True)
    await repos.stats.apply_delta("cat", delta=-1)  # 3→2, stays mastered
    await repos.stats.apply_delta("cat", delta=-1)  # 2→1, demotes
    cat = await repos.stats.get("cat")
    assert cat["mastery_score"] == 1
    assert cat["status"] == "learning"


@pytest.mark.asyncio
async def test_passive_exposed_word_can_graduate_to_mastered(setup):
    """A word that has never been target can still reach 'mastered' via
    correct translations during other target practices. This is the user's
    '不耽误被动曝光的词自然达到掌握' goal."""
    repos = setup
    # 'cat' appears as scaffolding 3 times, kid translates correctly each time
    await repos.stats.apply_delta("cat", delta=1, exposed=True)  # exposed + correct
    assert (await repos.stats.get("cat"))["status"] == "exposed"
    await repos.stats.apply_delta("cat", delta=1)
    assert (await repos.stats.get("cat"))["status"] == "exposed"
    await repos.stats.apply_delta("cat", delta=1)
    # mastery hits 3 → graduates to 'mastered' regardless of target history
    assert (await repos.stats.get("cat"))["status"] == "mastered"


@pytest.mark.asyncio
async def test_learning_count_excludes_exposed_words(setup):
    """Refill-mode counting must skip 'exposed' words — only target-exposed
    words count toward the learning pool. This is the core fix that prevents
    scaffolding from prematurely triggering high_watermark."""
    repos = setup
    # 3 scaffolding-only exposures
    await repos.stats.increment_exposed("alpha")
    await repos.stats.increment_exposed("bravo")
    await repos.stats.increment_exposed("charlie")
    # 1 target exposure
    await repos.stats.increment_exposed("target_word", is_target=True)
    # learning_count must count only the target word
    assert await repos.stats.learning_count() == 1
