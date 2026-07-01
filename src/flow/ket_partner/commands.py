from typing import Optional

from flow.common import logger
from flow.ket_partner.config import load_config
from flow.ket_partner.db import init_db
from flow.ket_partner.exporter import export_learning_report


class ExitLoop(Exception):
    pass


class CommandHandler:
    SUPPORTED = {
        "/exportstats": "导出学习状态报告",
        "/exit":        "退出练习",
        "/quit":        "退出练习",
        "/help":        "显示命令列表",
    }

    def __init__(self, db_path: str, chat_logger):
        self.db_path = db_path
        self.chat_logger = chat_logger

    async def handle(self, user_input: str) -> None:
        cmd = user_input.strip().split()[0]
        if cmd == "/exit" or cmd == "/quit":
            raise ExitLoop()
        elif cmd == "/help":
            self._print_help()
        elif cmd == "/exportstats":
            await self._export_stats()
        else:
            print(f"未知命令: {cmd}。输入 /help 查看支持的命令。")

    def _print_help(self) -> None:
        for cmd, desc in self.SUPPORTED.items():
            print(f"  {cmd:<15} {desc}")

    async def _export_stats(self) -> None:
        from datetime import datetime
        cfg = load_config()
        repos = await init_db(self.db_path, csv_path=None)
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"learning_report_{stamp}.md"
            await export_learning_report(output_path, repos, cfg)
            print(f"已导出学习报告到 {output_path}")
        finally:
            await repos._db.close()
