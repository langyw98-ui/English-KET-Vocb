from typing import Dict, List, Optional, TypedDict

from langchain.messages import AnyMessage


class BTPKetState(TypedDict):
    messages: List[AnyMessage]
    intent: Optional[str]
    asked_word: Optional[str]
    wrong_words: Optional[List[str]]
    correct_meanings: Optional[Dict[str, str]]
    target_word_meaning: Optional[str]
    asked_word_meaning: Optional[str]
    # Populated by evaluate_translation_node when the kid mistranslates a
    # preposition / spatial particle (in/on/at/etc.). Format:
    # [{"word": "in", "kid_translation": "上", "correct_translation": "里",
    #   "contrast": "in=里, on=上"}]
    function_word_errors: Optional[List[Dict[str, str]]]
    target_word: Optional[str]
    last_target_word: Optional[str]
    last_sentence_words: Optional[List[str]]
    topic: Optional[str]
    profile_strategy: Optional[str]
    profile_weakness: Optional[str]
    last_english_sentence: Optional[str]
    # Flag set by generate_sentence_node after it has incremented
    # exposed_count for the NEW sentence's words. persist_turn_node reads
    # this so non-generate turns (asks_meaning/idk/off_topic/non_compliant)
    # do NOT re-increment exposure for the prior sentence's words.
    _exposure_recorded: Optional[bool]
