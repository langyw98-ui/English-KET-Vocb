from typing import Literal, TypedDict

from langchain_core.messages import AnyMessage

KetIntent = Literal["translation", "asks_meaning", "idk", "off_topic", "non_compliant"]


class BTPKetState(TypedDict):
    """
    BTP Ket Partner 核心对话状态图

    字段 Single-Writer / Multi-Writer 声明(按实际代码,nodes 已合并入 KETPartnerAgent):

    - messages: 仅 init_state(截断超 10 条时)、format_output_node、
      explain_meaning_node、redirect_to_translate_node、compliance_redirect_node
      在追加 AI 回复时写;其他位置只读
    - intent: 仅 classify_intent_node 在路由阶段写;其他位置只读
    - asked_word: 仅 classify_intent_node 在解析查词意图时写;其他位置只读
    - wrong_words: 仅 evaluate_translation_node 写;其他位置只读
    - sentence_translation: 仅 evaluate_translation_node(translation 路径)
      与 lookup_target_meaning_node(idk 路径)写;其他位置只读
    - overall_correct: 仅 evaluate_translation_node 写;其他位置只读
    - asked_word_meaning: 仅 lookup_asked_meaning_node 写;其他位置只读
    - target_word: init_state(从上一轮 AI 消息恢复时)与
      select_target_word_node、generate_sentence_node(换词时)写;其他位置只读
    - target_context: 同 target_word,init_state / select_target_word_node /
      generate_sentence_node 写;其他位置只读
    - last_target_word: 仅 init_state(从上一轮 AI 消息恢复)写;
      persist_turn_node 仅读取用于落库,不写状态;其他位置只读
    - last_target_context: 同 last_target_word,仅 init_state 写;
      persist_turn_node 仅读取;其他位置只读
    - last_sentence_words: init_state(从上一轮 AI 消息恢复)与
      generate_sentence_node 写;其他位置只读
    - topic: 仅 init_state 从 profile 加载时写;其他位置只读
    - profile_strategy: 仅 init_state 从 DB profile 加载时写;其他位置只读
    - profile_weakness: 同 profile_strategy,仅 init_state 写;其他位置只读
    - last_english_sentence: init_state(从上一轮 AI 消息恢复)与
      generate_sentence_node 写;其他位置只读
    - _exposure_recorded: 仅 generate_sentence_node 标记,persist_turn_node 读取;
      其他位置只读
    - non_ket_annotations: 仅 generate_sentence_node 写;其他位置只读
    """

    messages: list[AnyMessage]
    intent: KetIntent | None
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
