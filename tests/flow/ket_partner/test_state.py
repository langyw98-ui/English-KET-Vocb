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


def test_ket_intent_matches_classification_schema():
    """state.KetIntent 字面量必须与 dialogue_domain.IntentClassification 的 Literal 完全一致,
    否则 LLM 返回的 intent 值无法被类型系统校验,路由可能静默失败。"""
    from typing import get_args

    from flow.ket_partner.dialogue_domain import IntentClassification
    from flow.ket_partner.state import KetIntent

    state_literals = set(get_args(KetIntent))
    schema_field = IntentClassification.model_fields["intent"]
    schema_literals = set(schema_field.annotation.__args__)
    assert state_literals == schema_literals, (
        f"KetIntent 与 IntentClassification.intent 不一致: "
        f"state={state_literals}, schema={schema_literals}"
    )
