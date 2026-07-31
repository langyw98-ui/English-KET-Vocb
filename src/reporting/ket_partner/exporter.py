# src/reporting/ket_partner/exporter.py
"""Report export orchestration. Replaces flow/ket_partner/exporter.py.

Key changes vs old:
- _fetch_all_stats via repos.stats._db.execute → await repos.stats.list_all_with_vocab()
- inline _classify → src.reporting.ket_partner.categories.group_by_category
- inline _render_markdown → src.reporting.ket_partner.markdown.render_markdown
- repos: Repos → KETPartnerRepos Protocol

Single-writer notes (CLAUDE.md §三):
- output_path file: only export_learning_report writes (single call site per request)
"""
import asyncio
from pathlib import Path

from flow.common import logger
from flow.ket_partner.config import KetConfig
from flow.ket_partner.persistence import KETPartnerRepos
from src.reporting.ket_partner.categories import group_by_category
from src.reporting.ket_partner.markdown import render_markdown


def _write_report_file(path: Path, content: str) -> None:
    """Sync file writer wrapped by asyncio.to_thread in async callers.

    §7.2: async functions must not perform sync IO; offload to thread pool.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


async def export_learning_report(
    output_path: str = "storage/reports/learning_report.md",
    repos: KETPartnerRepos | None = None,
    cfg: KetConfig | None = None,
    fmt: str = "markdown",
) -> str:
    """Pull stats → group → render → write file. Returns output_path."""
    if fmt != "markdown":
        raise ValueError(f"unsupported format: {fmt}")
    if repos is None or cfg is None:
        raise ValueError("repos and cfg must be provided")
    out_p = Path(output_path)
    if out_p.parent == Path("."):
        out_p = Path("storage/reports") / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    content = await render_report_text(repos, cfg)
    await asyncio.to_thread(_write_report_file, out_p, content)
    logger.info(f"Exported learning report to {out_p}")
    return str(out_p)


async def render_report_text(
    repos: KETPartnerRepos,
    cfg: KetConfig,
) -> str:
    """In-memory render — for API /report or unit tests.

    Pulls profile + all vocab rows, groups into 5 categories, renders
    markdown. No file IO — caller decides where to write.
    """
    profile = await repos.profile.get()
    rows = await repos.stats.list_all_with_vocab()
    rows_by_category = group_by_category(rows, cfg)
    return render_markdown(profile, rows_by_category)
