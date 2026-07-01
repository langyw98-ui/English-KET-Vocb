import json
from os.path import dirname, join

from pydantic import BaseModel


class VocabRefillConfig(BaseModel):
    low_watermark: int = 5
    high_watermark: int = 10
    interval_turns: int = 5


class SentenceConfig(BaseModel):
    min_words: int = 5
    max_words: int = 12
    rewrite_threshold: int = 2


class VarietyConfig(BaseModel):
    recent_window: int = 3


class SummaryConfig(BaseModel):
    interval_turns: int = 15


class StrugglingThreshold(BaseModel):
    wrong_count_min: int = 2
    exposed_count_min: int = 4


class KetConfig(BaseModel):
    vocab_refill: VocabRefillConfig = VocabRefillConfig()
    sentence: SentenceConfig = SentenceConfig()
    variety: VarietyConfig = VarietyConfig()
    summary: SummaryConfig = SummaryConfig()
    validate_retry_limit: int = 2
    struggling_threshold: StrugglingThreshold = StrugglingThreshold()


_CONFIG_PATH = join(dirname(__file__), "data", "config.json")


def load_config() -> KetConfig:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return KetConfig.model_validate(data)
