from typing import TypedDict

from langchain_core.messages import AnyMessage


class BTPKetState(TypedDict):
    messages: list[AnyMessage]
    intent: str | None
    asked_word: str | None
    # Unified list of words the kid mistranslated. Populated by
    # evaluate_translation_node. Filtered to words that appear in the
    # displayed sentence (KET canonical + non-KET tokens) so LLM
    # hallucinations get dropped. Non-KET entries reach display for
    # correction feedback but never reach vocab_stats — apply_mastery_updates
    # iterates last_sentence_words (KET subset) only.
    # Format: [{"word": "eat", "kid_translation": "在",
    #           "correct_translation": "吃", "contrast": "..."}]
    wrong_words: list[dict[str, str]] | None
    # Full Chinese translation of the last English sentence. Populated by
    # either evaluate_translation_node (translation intent) or
    # lookup_target_meaning_node (idk intent) so format_output_text can
    # render "正确翻译：..." uniformly.
    sentence_translation: str | None
    # Sentence-level verdict from the translation evaluator. True when the
    # kid's translation faithfully conveys the whole English sentence. The
    # per-word wrong_words list catches misaligned/omitted words; this flag
    # catches STRUCTURAL errors no single word owns — most commonly the kid
    # adding content that isn't in the English original (e.g. "玩球" for
    # "play") or distorting the sentence's overall meaning. None on
    # non-translation intents; default True on the evaluator side so a
    # missing/old field never falsely punishes.
    overall_correct: bool | None
    asked_word_meaning: str | None
    target_word: str | None
    # Context of the current target word. Only target words carry real
    # context; scaffolding words always use context="" (Spec §4.3).
    target_context: str | None
    last_target_word: str | None
    # Context of the previous turn's target. Used by evaluate_translation_node
    # to thread context into the evaluator prompt (Spec §7.4).
    last_target_context: str | None
    last_sentence_words: list[str] | None
    topic: str | None
    profile_strategy: str | None
    profile_weakness: str | None
    last_english_sentence: str | None
    # Flag set by generate_sentence_node after it has incremented
    # exposed_count for the NEW sentence's words. persist_turn_node reads
    # this so non-generate turns (asks_meaning/idk/off_topic/non_compliant)
    # do NOT re-increment exposure for the prior sentence's words.
    _exposure_recorded: bool | None
    # Non-KET-word annotations for the displayed sentence. Populated by
    # generate_sentence_node when the accepted sentence still contains
    # ≤1 (or, after retries, more) non-KET words. Format:
    # [{"word": "sledding", "meaning": "滑雪橇"}, ...]
    # format_output_text renders each as "<word> 的意思是：<meaning>" after
    # the sentence so the kid can still translate even with unknown words.
    non_ket_annotations: list[dict[str, str]] | None
