from flow.ket_partner.state import BTPKetState


def test_state_has_required_fields():
    state: BTPKetState = {
        "messages": [],
        "intent": None,
        "asked_word": None,
        "wrong_words": None,
        "sentence_translation": None,
        "asked_word_meaning": None,
        "target_word": None,
        "last_target_word": None,
        "last_sentence_words": None,
        "topic": None,
        "profile_strategy": None,
        "profile_weakness": None,
        "last_english_sentence": None,
        "_exposure_recorded": None,
    }
    assert state["intent"] is None
