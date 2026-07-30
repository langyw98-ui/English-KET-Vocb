"""Markdown rendering helpers for the learning report. Pure functions."""


def fmt_word(word: str, context: str) -> str:
    """Render 'word(context)' when context non-empty, else plain 'word'."""
    return f"{word}({context})" if context else word


def _render_table(rows: list[dict], with_stats: bool) -> list[str]:
    lines: list[str] = []
    if with_stats:
        lines.append("| word | pos | exposed | correct | wrong | mastery |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            w = fmt_word(r["word"], r["context"])
            lines.append(
                f"| {w} | {r['pos']} | {r['exposed_count']} | "
                f"{r['correct_count']} | {r['wrong_count']} | {r['mastery_score']} |"
            )
    else:
        lines.append("| word | pos |")
        lines.append("|---|---|")
        for r in rows:
            w = fmt_word(r["word"], r["context"])
            lines.append(f"| {w} | {r['pos']} |")
    return lines


def render_markdown(
    profile: dict,
    rows_by_category: dict[str, list[dict]],
) -> str:
    """Render report from pre-bucketed rows. Categories module owns the
    classification; this function only renders.
    """
    mastered = rows_by_category["mastered"]
    learning = rows_by_category["learning"]
    used = rows_by_category["used"]
    unused = rows_by_category["unused"]
    struggling = rows_by_category["struggling"]

    lines: list[str] = [
        f"# 学习报告 - {profile.get('nickname') or '小朋友'}",
        f"总轮数: {profile.get('total_turns', 0)}",
        "",
        f"## 正在学习 ({len(learning)} 项)",
    ]
    lines.extend(_render_table(learning, with_stats=True))
    lines.append("")
    lines.append(f"## 已掌握 ({len(mastered)} 项)")
    lines.extend(_render_table(mastered, with_stats=True))
    lines.append("")
    lines.append(f"## 已使用 ({len(used)} 项)")
    lines.extend(_render_table(used, with_stats=True))
    lines.append("")
    lines.append(f"## 未使用 ({len(unused)} 项)")
    lines.extend(_render_table(unused, with_stats=False))
    lines.append("")
    lines.append(f"## 学习困难 ({len(struggling)} 项)")
    lines.extend(_render_table(struggling, with_stats=True))
    return "\n".join(lines)
