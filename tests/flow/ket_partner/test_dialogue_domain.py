from unittest.mock import AsyncMock, MagicMock

import pytest

from flow.ket_partner.dialogue_domain import (
    IntentClassification,
    ProfileSummary,
    TranslationEval,
    WrongWord,
    classify_intent,
    evaluate_translation,
    format_output_text,
    run_profile_summary,
)
from src.persistence.bootstrap import init_db
from src.persistence.repos import Repos

# ===========================================================================
# Input Classifier Tests
# ===========================================================================

def _make_classifier_llm(return_value: IntentClassification):
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=return_value)
    llm.with_structured_output = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_classify_translation():
    expected = IntentClassification(intent="translation", asked_word=None)
    llm = _make_classifier_llm(expected)
    result = await classify_intent(llm, "The cat is big.", "那只猫很大")
    assert result.intent == "translation"
    assert result.asked_word is None
    llm.with_structured_output.return_value.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_classify_idk():
    expected = IntentClassification(intent="idk", asked_word=None)
    llm = _make_classifier_llm(expected)
    result = await classify_intent(llm, "The cat is big.", "我不会")
    assert result.intent == "idk"
    llm.with_structured_output.return_value.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_classify_asks_meaning_extracts_word():
    expected = IntentClassification(intent="asks_meaning", asked_word="cat")
    llm = _make_classifier_llm(expected)
    result = await classify_intent(llm, "The cat is big.", "cat 是什么意思")
    assert result.intent == "asks_meaning"
    assert result.asked_word == "cat"
    llm.with_structured_output.return_value.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_classify_fallback_on_llm_error():
    """LLM 调用抛 openai.APIError 时,fallback 返回 translation,且 mock 确实被调用过。"""
    from openai import APIError

    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(
        side_effect=APIError(
            message="llm down",
            request=MagicMock(),
            body=None,
        )
    )
    llm.with_structured_output = MagicMock(return_value=bound)
    result = await classify_intent(llm, "The cat is big.", "那只猫很大")
    assert result.intent == "translation"
    llm.with_structured_output.assert_called_once_with(IntentClassification, method="function_calling")
    bound.ainvoke.assert_awaited_once()


# ===========================================================================
# Profile Summarizer Tests
# ===========================================================================

def _make_summary_llm(return_value):
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=return_value)
    llm.with_structured_output = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_summary_updates_profile(temp_db_path):
    db = await init_db(temp_db_path, csv_path=None)
    repos = Repos.for_user(db, "default")
    try:
        await repos.log.append("ai", "The cat is big.", ["cat"], [{"word": "cat", "context": ""}], turn_id=1)
        await repos.log.append("user", "那只狗很大", [], None, turn_id=1)
        llm = _make_summary_llm(ProfileSummary(
            weakness_words=["cat"],
            dialogue_strategy="cat 反复译错,需用图示强化。",
        ))
        await run_profile_summary(llm, repos)
        profile = await repos.profile.get()
        assert "cat" in profile["weakness_words"]
        assert "图示" in profile["dialogue_strategy"]
        llm.with_structured_output.return_value.ainvoke.assert_awaited_once()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_summary_fallback_on_error(temp_db_path):
    """LLM 调用抛 openai.APIError 时,fallback 保留旧 profile,且 mock 确实被调用过。"""
    from openai import APIError

    db = await init_db(temp_db_path, csv_path=None)
    repos = Repos.for_user(db, "default")
    try:
        await repos.profile.update(weakness_words=["old"], dialogue_strategy="old strat")
        llm = MagicMock()
        bound = MagicMock()
        bound.ainvoke = AsyncMock(
            side_effect=APIError(
                message="fail",
                request=MagicMock(),
                body=None,
            )
        )
        llm.with_structured_output = MagicMock(return_value=bound)
        await run_profile_summary(llm, repos)
        profile = await repos.profile.get()
        assert profile["weakness_words"] == ["old"]
        assert profile["dialogue_strategy"] == "old strat"
        llm.with_structured_output.assert_called_once_with(ProfileSummary, method="function_calling")
        bound.ainvoke.assert_awaited_once()
    finally:
        await db.close()


# ===========================================================================
# Translation Evaluator Tests
# ===========================================================================

def _make_eval_llm(return_value: TranslationEval):
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=return_value)
    llm.with_structured_output = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_evaluate_finds_wrong_word():
    expected = TranslationEval(
        correct_translation="那只猫很大",
        wrong_words=[WrongWord(word="cat", kid_translation="狗", correct_translation="猫")],
    )
    llm = _make_eval_llm(expected)
    result = await evaluate_translation(
        llm,
        sentence="The cat is big.",
        words=["the", "cat", "is", "big"],
        target="cat",
        kid_input="那只狗很大",
    )
    assert result.correct_translation == "那只猫很大"
    assert len(result.wrong_words) == 1
    assert result.wrong_words[0].word == "cat"
    assert result.wrong_words[0].correct_translation == "猫"
    assert result.wrong_words[0].kid_translation == "狗"
    llm.with_structured_output.return_value.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_evaluate_no_wrong_words():
    expected = TranslationEval(correct_translation="那只猫很大", wrong_words=[])
    llm = _make_eval_llm(expected)
    result = await evaluate_translation(
        llm,
        sentence="The cat is big.",
        words=["the", "cat", "is", "big"],
        target="cat",
        kid_input="那只猫很大",
    )
    assert result.wrong_words == []
    assert result.correct_translation == "那只猫很大"
    llm.with_structured_output.return_value.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_evaluate_fallback_on_error():
    """LLM 调用抛 openai.APIError 时,fallback 返回空 wrong_words,且 mock 确实被调用过。"""
    from openai import APIError

    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(
        side_effect=APIError(
            message="fail",
            request=MagicMock(),
            body=None,
        )
    )
    llm.with_structured_output = MagicMock(return_value=bound)
    result = await evaluate_translation(
        llm,
        sentence="The cat is big.",
        words=["the", "cat", "is", "big"],
        target="cat",
        kid_input="那只猫很大",
    )
    assert result.wrong_words == []
    assert result.correct_translation == ""
    llm.with_structured_output.assert_called_once_with(TranslationEval, method="function_calling")
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_evaluate_threads_target_context_into_prompt():
    expected = TranslationEval(correct_translation="聪明的孩子", wrong_words=[])
    llm = _make_eval_llm(expected)
    await evaluate_translation(
        llm,
        sentence="The smart kid.",
        words=["the", "smart", "kid"],
        target="smart",
        target_context="clever",
        kid_input="聪明的孩子",
    )
    bound = llm.with_structured_output.return_value
    messages = bound.ainvoke.call_args.args[0]
    target_msg = next(m for m in messages if "Target word being tested" in m.content)
    assert "smart" in target_msg.content
    assert "clever" in target_msg.content
    bound.ainvoke.assert_awaited_once()


# ===========================================================================
# Output Format Tests
# ===========================================================================

def test_format_output_translation_no_wrong():
    state = {
        "intent": "translation",
        "wrong_words": [],
        "last_target_word": "cat",
    }
    text = format_output_text(state, new_sentence="The dog runs.")
    assert "The dog runs." in text


def test_format_output_translation_with_wrong():
    state = {
        "intent": "translation",
        "sentence_translation": "那只猫在跑",
        "wrong_words": [{
            "word": "cat",
            "kid_translation": "狗",
            "correct_translation": "猫",
        }],
        "last_target_word": "cat",
    }
    text = format_output_text(state, new_sentence="The dog runs.")
    assert "正确翻译：那只猫在跑" in text
    assert "你的翻译有误:" in text
    assert "cat" in text
    assert "猫" in text
    assert "cat 的意思是：猫" in text
    assert "The dog runs." in text


def test_format_output_translation_with_omitted_word():
    state = {
        "intent": "translation",
        "sentence_translation": "那只有趣的猫在我的床上休息",
        "wrong_words": [{
            "word": "my",
            "kid_translation": "",
            "correct_translation": "我的",
        }],
        "last_target_word": "cat",
    }
    text = format_output_text(state, new_sentence="The bird sings.")
    assert "你的翻译有误:" in text
    assert "my 的意思是：我的" in text


def test_format_output_idk():
    state = {
        "intent": "idk",
        "sentence_translation": "那只猫在跑",
        "last_target_word": "cat",
    }
    text = format_output_text(state, new_sentence="The dog runs.")
    assert "正确翻译：那只猫在跑" in text
    assert "The dog runs." in text


def test_format_output_first_turn_no_feedback():
    state = {
        "intent": None,
    }
    text = format_output_text(state, new_sentence="The dog runs.")
    assert "The dog runs." in text
