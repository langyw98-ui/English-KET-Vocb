from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    ai_reply: str
    turn_id: int


class ReportResponse(BaseModel):
    mastered_count: int
    learning_count: int
    struggling_count: int
    used_count: int
    unused_count: int
    total_words: int


class ReportWord(BaseModel):
    word: str
    context: str = ""
    mastery_score: int
    exposed_count: int
    correct_count: int
    wrong_count: int
    status: str


class ReportCategoryResponse(BaseModel):
    category: str
    page: int
    page_size: int
    total: int
    total_pages: int
    words: list[ReportWord]


class MessageOut(BaseModel):
    role: str
    content: str
    turn_id: int | None = None
    created_at: datetime


class LlmStatusResponse(BaseModel):
    state: Literal["red", "green"]
    masked_key: str | None
    last_error: str | None

