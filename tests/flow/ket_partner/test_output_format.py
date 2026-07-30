from flow.ket_partner.output_format import format_output_text


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
    # Full correct translation must appear FIRST.
    assert "正确翻译：那只猫在跑" in text
    assert "你的翻译有误:" in text
    assert "cat" in text
    assert "猫" in text
    # Per-word correction line: "cat 的意思是：猫"
    assert "cat 的意思是：猫" in text
    assert "The dog runs." in text


def test_format_output_translation_with_omitted_word():
    """Regression: when the kid OMITS a word entirely (kid_translation is
    empty), the wrong_words entry must still be rendered. Otherwise
    "你的翻译有误:" shows with no items below it — confusing."""
    state = {
        "intent": "translation",
        "sentence_translation": "那只有趣的猫在我的床上休息",
        "wrong_words": [{
            "word": "my",
            "kid_translation": "",  # kid omitted it
            "correct_translation": "我的",
        }],
        "last_target_word": "cat",
    }
    text = format_output_text(state, new_sentence="The bird sings.")
    assert "你的翻译有误:" in text
    assert "my 的意思是：我的" in text, "omitted word must still be rendered"


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
