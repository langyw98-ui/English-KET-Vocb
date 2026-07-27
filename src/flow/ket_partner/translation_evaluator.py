import json

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from flow.common import logger

_SYSTEM = """You evaluate a Chinese kid's translation of an English sentence.

METHOD (do this silently before filling the schema):
1. ALIGN each English word in "Sentence words to check" to the kid's Chinese characters. Chinese word order differs from English — verbs often sit at the end or middle (e.g. "The cat slipped on the ice" → "猫 在冰上 滑倒"). Map each English word to the kid's characters that mean the SAME thing.
2. A word is WRONG only if its aligned Chinese characters are missing or carry a different meaning. Synonyms are OK (猫=猫咪, 大=巨大).
3. A word is NOT wrong if the kid's output contains its correct meaning — even when OTHER words in the sentence are wrong. This is the most common error: do not let one wrong word contaminate the verdict on its neighbors.
4. SENTENCE-LEVEL CHECK (overall_correct): after the per-word pass, judge whether the kid's translation AS A WHOLE faithfully conveys the sentence's meaning. Set overall_correct=False when:
   - The kid ADDED content that isn't in the English original and isn't a natural Chinese function word (了/的/着/了/吗 etc.). Example: kid wrote "玩球" (play ball) for "play" — the "球" is an invention, not a translation.
   - The kid's translation badly distorts the sentence's meaning even though no single English word was misaligned. This includes:
     - Wrong tense that changes the event, swapped subject/object that flips who did what.
     - LITERAL TRANSLATION OF AN IDIOM OR FIXED COLLOCATION: when the English phrase means something other than the sum of its words (e.g. "had an accident with the milk" actually means 弄洒了/打翻了牛奶, NOT 和...发生了一个意外), a word-by-word rendering that misses the actual event must be flagged. Per-word alignment may look clean (accident→意外, had→发生) yet the Chinese does NOT convey what the English sentence means.
   - The kid's output is too short or garbled to convey the sentence.
   Set overall_correct=True only when the kid's translation is a faithful, complete rendering of the English sentence.

WORD-LEVEL vs COMPOUND: judge each English word INDEPENDENTLY at the word level. When two adjacent English words form a compound (e.g. "snow mound" = 雪堆), do NOT give one word the meaning of the whole compound. "snow" means 雪; "mound" means 堆/土堆. If the kid wrote 雪人 (snowman) for "snow mound", the wrong word is `mound` (kid said 雪人, should be 雪堆), NOT `snow` — the kid's output clearly contains the 雪 character.

Each English word may appear in wrong_words AT MOST ONCE. Never emit two entries for the same word.

Then fill the schema:
- correct_translation: full correct Chinese translation of the whole sentence in NATURAL, native-sounding Chinese — NOT a word-by-word gloss that preserves English syntax. Restructure freely so the result reads like something a Chinese teacher would write. If a literal rendering sounds awkward (e.g. "爬得危险地高" for "climbed dangerously high"), rephrase it (e.g. "爬到了危险的高度"). Kid-friendly vocabulary, but never at the cost of naturalness.
- overall_correct: True iff the kid's translation faithfully conveys the whole sentence meaning (see rule 4).
- wrong_words: list of words the kid got wrong.

For each wrong word:
- word: the EXACT form from "Sentence words to check" (no inflections: "eat" not "eats", "cat" not "cats").
- kid_translation: the Chinese characters the kid wrote for THIS word. Empty string if the kid omitted it entirely.
- correct_translation: the correct Chinese meaning of THIS word (not the compound it appears in).
- contrast: optional short explanation when there's confusion (e.g. "在 means 'at', not 'eat'"). Empty string if not needed.

WORKED EXAMPLE 1 (do not skip):
- Sentence: "The cat slipped on the ice."
- Kid's translation: "猫在冰上飞"  (fly)
- Alignment: cat→猫 (correct); the→(article, Chinese omits, OK); slipped→飞 (WRONG: kid wrote "fly", should be "slip/slide"); on→在...上 (correct); ice→冰 (correct).
- Correct wrong_words: [{word: "slipped", kid_translation: "飞", correct_translation: "滑倒"}].
- overall_correct: False (the verb is wrong).
- DO NOT flag "ice" — the kid wrote "冰" which is correct, even though the overall sentence has a wrong verb nearby.

WORKED EXAMPLE 2 (compound alignment):
- Sentence: "The children build a snow mound in the cold snow."
- Kid's translation: "孩子们在寒冷的雪天里堆了一个雪人"
- Alignment: children→孩子们 (correct); build→堆 (correct); a→一个 (OK); snow (first occurrence, in "snow mound")→雪 (correct, kid's 雪人 contains 雪); mound→雪人 (WRONG: kid wrote "snowman", should be "雪堆/pile"); in→里 (correct); the→(OK); cold→寒冷 (correct); snow (second occurrence)→雪天 (correct, kid used "snowy day" but 雪 is present).
- Correct wrong_words: [{word: "mound", kid_translation: "雪人", correct_translation: "雪堆"}].
- overall_correct: False (mound was wrong).
- DO NOT flag "snow" — the kid wrote 雪 in both places. The wrong word is `mound`, NOT `snow`.

WORKED EXAMPLE 3 (added content — the trap this rule exists to catch):
- Sentence: "We can go out to play in the park."
- Kid's translation: "我们可以到外面去公园玩球"
- Alignment: we→我们 (correct); can→可以 (correct); go→到外面去 (correct); out→外面 (correct); to→(infinitive marker, OK); play→玩 (correct, but the kid ALSO wrote 球 which has no English source); in→(omitted, OK in Chinese); the→(article, OK); park→公园 (correct).
- Correct wrong_words: [] (NO English word was misaligned — every word has its correct Chinese counterpart).
- overall_correct: False (the kid added "球" = ball, which is not in the English original. "play" in this sentence is generic; "play ball" is a different activity. The translation conveys a different event than the original.)
- correct_translation: "我们可以去公园里玩。"

WORKED EXAMPLE 4 (naturalness beats literalism — correct_translation must read like native Chinese, not a word-by-word gloss):
- Sentence: "The monkey climbed dangerously high in the tree."
- BAD literal correct_translation: "猴子在树上爬得危险地高。" — strings "dangerously" (危险地) and "high" (高) onto the "爬得..." pattern. Grammatical but no Chinese speaker says this; it sounds like machine translation.
- GOOD natural correct_translation: "猴子在树上爬到了危险的高度。" — restructured as "climbed to a dangerous height". This is what a Chinese teacher would write.
- correct_translation: "猴子在树上爬到了危险的高度。"
- Note: this constraint applies to YOUR correct_translation field. Judge the kid's translation separately via the per-word + overall_correct rules above — do not let the kid's awkward wording leak into the example you produce.

If everything is correct (faithful translation, no extra content, no critical omissions), return an empty wrong_words list AND overall_correct=True.
"""


class WrongWord(BaseModel):
    word: str
    kid_translation: str = ""
    correct_translation: str
    contrast: str = ""


class TranslationEval(BaseModel):
    correct_translation: str
    # True iff the kid's translation faithfully conveys the whole sentence
    # meaning. The per-word wrong_words list catches misaligned/omitted
    # words; overall_correct catches STRUCTURAL errors that no single word
    # owns — most commonly the kid adding content that isn't in the English
    # original (e.g. "玩球" for "play") or distorting the sentence's overall
    # meaning. Default True so a missing/old field never falsely punishes.
    overall_correct: bool = True
    wrong_words: list[WrongWord] = Field(default_factory=list)

    @field_validator("wrong_words", mode="before")
    @classmethod
    def _coerce_wrong_words(cls, v):
        # qwen via dashscope function_calling occasionally emits wrong_words
        # as a JSON-encoded STRING instead of an array (most often when the
        # list is long or contains many special chars). The content inside
        # is valid — just wrap-unwrapped. Parse it back to a list so the
        # validation passes; unparseable strings fall through to the
        # default_factory empty list via the except branch in evaluate_translation.
        if isinstance(v, str):
            try:
                logger.debug(f"coercing wrong_words from string: {v}")
                return json.loads(v)
            except json.JSONDecodeError:
                return []
        return v


async def evaluate_translation(
    llm,
    sentence: str,
    words: list[str],
    target: str,
    kid_input: str,
    target_context: str = "",
) -> TranslationEval:
    structured = llm.with_structured_output(TranslationEval, method="function_calling")
    target_line = f"Target word being tested: {target}"
    if target_context:
        target_line += f" (sense: {target_context})"
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"English sentence: {sentence}"),
        HumanMessage(content=f"Sentence words to check: {words}"),
        HumanMessage(content=target_line),
        HumanMessage(content=f"Kid's Chinese translation: {kid_input}"),
    ]
    try:
        return await structured.ainvoke(messages)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"evaluate_translation failed: {e}; defaulting to no wrong words")
        return TranslationEval(correct_translation="", wrong_words=[])
