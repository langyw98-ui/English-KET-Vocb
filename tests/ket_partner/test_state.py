from flow.ket_partner.state import BTPKetState


def test_state_has_required_fields():
    state: BTPKetState = {
        "messages": [],
        "intent": None,
        "asked_word": None,
        "wrong_words": None,
        "sentence_translation": None,
        "overall_correct": None,
        "asked_word_meaning": None,
        "target_word": None,
        "target_context": None,
        "last_target_word": None,
        "last_target_context": None,
        "last_sentence_words": None,
        "topic": None,
        "profile_strategy": None,
        "profile_weakness": None,
        "last_english_sentence": None,
        "_exposure_recorded": None,
        "non_ket_annotations": None,
    }
    assert state["intent"] is None
    assert state["target_context"] is None
    assert state["last_target_context"] is None
    # TypedDict isn't runtime-enforced; verify the type's __annotations__
    # actually carry the new fields so a missing declaration can't hide.
    annotations = BTPKetState.__annotations__
    assert "target_context" in annotations
    assert "last_target_context" in annotations
