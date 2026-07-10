from typing import List

from flow.common import logger
from flow.ket_partner.config import KetConfig
from flow.ket_partner.db import Repos


def _fmt_word(word: str, context: str) -> str:
    """Spec §10: render as 'word(context)' when context is non-empty,
    otherwise plain 'word'. No separate context column."""
    return f"{word}({context})" if context else word


async def _fetch_all_stats(repos: Repos) -> List[dict]:
    async with repos.stats._db.execute(
        "SELECT v.word, v.context, v.pos, s.exposed_count, s.correct_count, "
        "s.wrong_count, s.mastery_score, s.status "
        "FROM ket_vocabulary v "
        "LEFT JOIN vocab_stats s ON v.word = s.word AND v.context = s.context "
        "ORDER BY v.word, v.context"
    ) as cur:
        rows = await cur.fetchall()
    return [
        {
            "word": r[0],
            "context": r[1] or "",
            "pos": r[2],
            "exposed_count": r[3] or 0,
            "correct_count": r[4] or 0,
            "wrong_count": r[5] or 0,
            "mastery_score": r[6] or 0,
            "status": r[7] or "new",
        }
        for r in rows
    ]


def _classify(row: dict, cfg: KetConfig) -> str:
    if row["exposed_count"] == 0:
        return "unused"
    if row["status"] == "mastered":
        return "mastered"
    if row["status"] == "learning":
        return "learning"
    if (row["wrong_count"] >= cfg.struggling_threshold.wrong_count_min
        or (row["exposed_count"] >= cfg.struggling_threshold.exposed_count_min
            and row["mastery_score"] == 0)):
        return "struggling"
    return "used"


def _render_markdown(profile: dict, rows: List[dict], cfg: KetConfig) -> str:
    mastered = [r for r in rows if _classify(r, cfg) == "mastered"]
    learning = [r for r in rows if _classify(r, cfg) == "learning"]
    used = [r for r in rows if _classify(r, cfg) == "used"]
    unused = [r for r in rows if _classify(r, cfg) == "unused"]
    struggling = [r for r in rows if _classify(r, cfg) == "struggling"]
    lines = [
        f"# 学习报告 - {profile.get('nickname') or '小朋友'}",
        f"总轮数: {profile.get('total_turns', 0)}",
        "",
        f"## 正在学习 ({len(learning)} 项)",
        "| word | pos | exposed | correct | wrong | mastery |",
        "|---|---|---|---|---|---|",
    ]
    for r in learning:
        w = _fmt_word(r["word"], r["context"])
        lines.append(
            f"| {w} | {r['pos']} | {r['exposed_count']} | "
            f"{r['correct_count']} | {r['wrong_count']} | {r['mastery_score']} |"
        )
    lines.append("")
    lines.append(f"## 已掌握 ({len(mastered)} 项)")
    lines.append("| word | pos | exposed | correct | wrong | mastery |")
    lines.append("|---|---|---|---|---|---|")
    for r in mastered:
        w = _fmt_word(r["word"], r["context"])
        lines.append(
            f"| {w} | {r['pos']} | {r['exposed_count']} | "
            f"{r['correct_count']} | {r['wrong_count']} | {r['mastery_score']} |"
        )
    lines.append("")
    lines.append(f"## 已使用 ({len(used)} 项)")
    lines.append("| word | pos | exposed | correct | wrong | mastery |")
    lines.append("|---|---|---|---|---|---|")
    for r in used:
        w = _fmt_word(r["word"], r["context"])
        lines.append(
            f"| {w} | {r['pos']} | {r['exposed_count']} | "
            f"{r['correct_count']} | {r['wrong_count']} | {r['mastery_score']} |"
        )
    lines.append("")
    lines.append(f"## 未使用 ({len(unused)} 项)")
    lines.append("| word | pos |")
    lines.append("|---|---|")
    for r in unused:
        w = _fmt_word(r["word"], r["context"])
        lines.append(f"| {w} | {r['pos']} |")
    lines.append("")
    lines.append(f"## 学习困难 ({len(struggling)} 项)")
    lines.append("| word | pos | exposed | correct | wrong | mastery |")
    lines.append("|---|---|---|---|---|---|")
    for r in struggling:
        w = _fmt_word(r["word"], r["context"])
        lines.append(
            f"| {w} | {r['pos']} | {r['exposed_count']} | "
            f"{r['correct_count']} | {r['wrong_count']} | {r['mastery_score']} |"
        )
    return "\n".join(lines)


async def export_learning_report(
    output_path: str,
    repos: Repos,
    cfg: KetConfig,
    fmt: str = "markdown",
) -> str:
    profile = await repos.profile.get()
    rows = await _fetch_all_stats(repos)
    if fmt == "markdown":
        content = _render_markdown(profile, rows, cfg)
    else:
        raise ValueError(f"unsupported format: {fmt}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Exported learning report to {output_path}")
    return output_path
