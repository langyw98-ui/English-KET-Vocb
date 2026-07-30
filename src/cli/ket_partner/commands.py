# src/cli/ket_partner/commands.py
"""CLI command dispatch. /exportstats now uses the injected Repos (no separate
init_db, no private _db access).
"""
from datetime import datetime

from flow.ket_partner.config import load_config
from flow.ket_partner.persistence import KETPartnerRepos
from src.reporting.ket_partner.exporter import export_learning_report


class ExitLoop(Exception):
    """/exit or /quit raises this to break the main loop."""


class CommandHandler:
    SUPPORTED: dict[str, str] = {
        "/exportstats": "导出学习状态报告",
        "/exit":        "退出练习",
        "/quit":        "退出练习",
        "/help":        "显示命令列表",
    }

    def __init__(self, repos: KETPartnerRepos, chat_logger: object) -> None:
        self.repos = repos
        self.chat_logger = chat_logger

    async def handle(self, user_input: str) -> None:
        cmd = user_input.strip().split()[0]
        if cmd in ("/exit", "/quit"):
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
        """Use injected repos. No init_db, no private _db access."""
        cfg = load_config()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"learning_report_{stamp}.md"
        await export_learning_report(output_path, self.repos, cfg)
        print(f"已导出学习报告到 {output_path}")
