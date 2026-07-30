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
from flow.common import logger
from flow.ket_partner.config import KetConfig
from flow.ket_partner.persistence import KETPartnerRepos
from src.reporting.ket_partner.categories import group_by_category
from src.reporting.ket_partner.markdown import render_markdown


async def export_learning_report(
    output_path: str,
    repos: KETPartnerRepos,
    cfg: KetConfig,
    fmt: str = "markdown",
) -> str:
    """Pull stats → group → render → write file. Returns output_path."""
    if fmt != "markdown":
        raise ValueError(f"unsupported format: {fmt}")
    content = await render_report_text(repos, cfg)
    with open(output_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        f.write(content)
    logger.info(f"Exported learning report to {output_path}")
    return output_path


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
