
from flow.ket_partner.state import BTPKetState


def route_by_intent(state: BTPKetState) -> str:
    intent = state.get("intent")
    if intent in ("translation", "idk"):
        return "select_target_word"
    if intent == "asks_meaning":
        return "explain_meaning"
    if intent == "off_topic":
        return "redirect_to_translate"
    if intent == "non_compliant":
        return "compliance_redirect"
    return "select_target_word"


def route_after_init(state: BTPKetState) -> str:
    if state.get("last_english_sentence") is None:
        return "select_target_word"
    return "classify_intent"
