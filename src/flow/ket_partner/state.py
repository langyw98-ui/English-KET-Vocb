from typing import Any, Dict, List, Optional, TypedDict

from langchain.messages import AnyMessage


class BTPKetState(TypedDict):
    messages: List[AnyMessage]
    intent: Optional[str]
    asked_word: Optional[str]
    wrong_words: Optional[List[str]]
    correct_meanings: Optional[Dict[str, str]]
    target_word_meaning: Optional[str]
    asked_word_meaning: Optional[str]
    target_word: Optional[str]
    last_target_word: Optional[str]
    last_sentence_words: Optional[List[str]]
    topic: Optional[str]
    profile_strategy: Optional[str]
    profile_weakness: Optional[str]
    last_english_sentence: Optional[str]
