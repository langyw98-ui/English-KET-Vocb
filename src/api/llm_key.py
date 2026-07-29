import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Ensure src is in sys.path for from flow.common import ...
src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from flow.common import dashscope_api_key  # noqa: E402


@dataclass
class LlmKeyStatus:
    """LLM 可用性状态容器。

    共享可变状态,单一写入者契约(CLAUDE.md §三.2、§三.3):

    - last_error & last_error_updated_at: 仅 routes/chat.py 在 chat 鉴权失败或成功时写,
      记录状态变更时间戳以解决并发/交错竞态问题。其他位置只读。

    并发写入语义: guard 用写入时间(completion time, time.time())比较,即"最后完成的那次 chat 胜出",
    确保成功产出 AI 回复的请求能及时清空错误标记,真实反映系统的最新可用状态。
    """

    last_error: str | None = None
    last_error_updated_at: float | None = None

    def set_error(self, error: str, timestamp: float | None = None) -> None:
        ts = timestamp or time.time()
        if self.last_error_updated_at is None or ts >= self.last_error_updated_at:
            self.last_error = error
            self.last_error_updated_at = ts

    def clear_error(self, timestamp: float | None = None) -> None:
        ts = timestamp or time.time()
        if self.last_error_updated_at is None or ts >= self.last_error_updated_at:
            self.last_error = None
            self.last_error_updated_at = ts

    @property
    def state(self) -> Literal["red", "green"]:
        key = _read_current_key()
        if not key or not key.strip():
            return "red"
        if self.last_error is not None:
            return "red"
        return "green"


def _read_current_key() -> str:
    """从 src/flow/common.py 读取已解析的 dashscope key(已 strip)。"""
    return str(dashscope_api_key.strip())


def mask_key(key: str) -> str | None:
    """格式化 key 为掩码形式:

    - 空 / 纯空白 → None
    - 长度 < 8   → "***XX"(末 2 位)
    - 长度 ≥ 8   → "XXXX***XXXX"(前 4 + 后 4)
    """
    if not key or not key.strip():
        return None
    k = key.strip()
    if len(k) < 8:
        return f"***{k[-2:]}"
    return f"{k[:4]}***{k[-4:]}"
