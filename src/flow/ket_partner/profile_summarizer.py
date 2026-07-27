
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from flow.common import logger
from flow.ket_partner.db import Repos

_SYSTEM = """你分析一个小朋友的 KET 词汇学习历史,增量更新学习画像。
- 学习聚焦,不分析兴趣 / 性格 / 情感
- 输出全部中文
- weakness_words: 持续挣扎的词或词类(从历史中提取)
- dialogue_strategy: 2-3 句具体可执行的建议(哪些词加强 / 哪些解释方式有效)
- 增量更新: 如果旧的 weakness_words 仍准确,保留;否则替换
"""


class ProfileSummary(BaseModel):
    weakness_words: list[str] = Field(default_factory=list)
    dialogue_strategy: str = ""


async def run_profile_summary(llm, repos: Repos) -> None:
    profile = await repos.profile.get()
    recent = await repos.log.recent(limit=20)
    log_text = "\n".join(f"[{r['role']}] {r['content']}" for r in recent)

    structured = llm.with_structured_output(ProfileSummary, method="function_calling")
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"当前 weakness: {profile['weakness_words']}"),
        HumanMessage(content=f"当前 strategy: {profile['dialogue_strategy']}"),
        HumanMessage(content=f"最近对话:\n{log_text}"),
    ]
    try:
        summary = await structured.ainvoke(messages)
        await repos.profile.update(
            weakness_words=summary.weakness_words,
            dialogue_strategy=summary.dialogue_strategy,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"profile summary failed: {e}; keeping old profile")
