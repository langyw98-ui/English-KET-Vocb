import json
from os.path import dirname, join
from typing import Any

from pydantic import BaseModel


class VocabRefillConfig(BaseModel):
    low_watermark: int = 5
    high_watermark: int = 10
    interval_turns: int = 3


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

# 模块级一次性加载 JSON(同步,import 时执行)。
# 这样 load_config() 函数体只做 pydantic 校验,不触发 IO,
# 在 async 调用路径(graph.build_agent)中安全。
# 与 sentence_domain.py 模块级加载 function_words.json / lemmas.json 同模式。
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _CONFIG_DATA: Any = json.load(f)


def load_config() -> KetConfig:
    """返回新的 KetConfig 实例(每次调用都 model_validate 一遍)。"""
    return KetConfig.model_validate(_CONFIG_DATA)
