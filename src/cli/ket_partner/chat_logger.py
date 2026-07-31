from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import TextIO


class ChatLogger:
    def __init__(self, log_dir: str = "storage/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._fp: TextIO | None = None
        self._session: Path | None = None

    def _next_index(self) -> int:
        existing = list(self.log_dir.glob("chat_log_*.txt"))
        indices = []
        for f in existing:
            try:
                idx = int(f.stem.split("_")[-1])
                indices.append(idx)
            except ValueError:
                continue
        return (max(indices) + 1) if indices else 1

    def start_session(self, nickname: str) -> None:
        idx = self._next_index()
        path = self.log_dir / f"chat_log_{idx:04d}.txt"
        self._session = path
        self._fp = path.open("w", encoding="utf-8")
        self._fp.write("=" * 60 + "\n")
        self._fp.write(f"Chat Log - Session {idx:04d}\n")
        self._fp.write(f"Nickname: {nickname}\n")
        self._fp.write(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        self._fp.write("=" * 60 + "\n\n")
        self._fp.flush()

    def log_turn(self, turn_id: int, role: str, content: str) -> None:
        if self._fp is None:
            return
        self._fp.write(f"[turn {turn_id:>4} - {role:>4}] {content}\n")
        self._fp.flush()

    def close_session(self) -> None:
        if self._fp is None:
            return
        try:
            self._fp.write("\n" + "-" * 60 + "\n")
            self._fp.write(f"Session ended: {datetime.now():%H:%M:%S}\n")
        finally:
            self._fp.close()
            self._fp = None

    def __enter__(self) -> "ChatLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # 兜底关闭:无论 with 块内是否抛异常,都尝试关闭当前 session。
        # close_session 内部已判 _fp is None,二次调用安全。
        self.close_session()
