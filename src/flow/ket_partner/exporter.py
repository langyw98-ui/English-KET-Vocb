from typing import List

from flow.common import logger
from flow.ket_partner.config import KetConfig
from flow.ket_partner.db import Repos


async def _fetch_all_stats(repos: Repos) -> List[dict]:
    async with repos.stats._db.execute(
        "SELECT v.word, v.pos, s.exposed_count, s.correct_count, s.wrong_count, "
        "s.mastery_score, s.status FROM ket_vocabulary v "
        "LEFT JOIN vocab_stats s ON v.word = s.word "
        "ORDER BY v.word"
    ) as cur:
        rows = await cur.fetchall()
    return [
        {
            "word": r[0],
            "pos": r[1],
            "exposed_count": r[2] or 0,
            "correct_count": r[3] or 0,
            "wrong_count": r[4] or 0,
            "mastery_score": r[5] or 0,
            "status": r[6] or "new",
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
        f"## 正在学习 ({len(learning)} 词)",
        "| word | pos | exposed | correct | wrong | mastery |",
        "|---|---|---|---|---|---|",
    ]
    for r in learning:
        lines.append(
            f"| {r['word']} | {r['pos']} | {r['exposed_count']} | "
            f"{r['correct_count']} | {r['wrong_count']} | {r['mastery_score']} |"
        )
    lines.append("")
    lines.append(f"## 已掌握 ({len(mastered)} 词)")
    lines.append("| word | pos | exposed | correct | wrong | mastery |")
    lines.append("|---|---|---|---|---|---|")
    for r in mastered:
        lines.append(
            f"| {r['word']} | {r['pos']} | {r['exposed_count']} | "
            f"{r['correct_count']} | {r['wrong_count']} | {r['mastery_score']} |"
        )
    lines.append("")
    lines.append(f"## 已使用 ({len(used)} 词)")
    lines.append("| word | pos | exposed | correct | wrong | mastery |")
    lines.append("|---|---|---|---|---|---|")
    for r in used:
        lines.append(
            f"| {r['word']} | {r['pos']} | {r['exposed_count']} | "
            f"{r['correct_count']} | {r['wrong_count']} | {r['mastery_score']} |"
        )
    lines.append("")
    lines.append(f"## 未使用 ({len(unused)} 词)")
    lines.append("| word | pos |")
    lines.append("|---|---|")
    for r in unused:
        lines.append(f"| {r['word']} | {r['pos']} |")
    lines.append("")
    lines.append(f"## 学习困难 ({len(struggling)} 词)")
    lines.append("| word | pos | exposed | correct | wrong | mastery |")
    lines.append("|---|---|---|---|---|---|")
    for r in struggling:
        lines.append(
            f"| {r['word']} | {r['pos']} | {r['exposed_count']} | "
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
