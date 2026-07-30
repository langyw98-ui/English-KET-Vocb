"""Single source of truth for the 5 report categories.

Option B: replaces the old StatsRepo._category_where_sql / count_by_category
/ list_by_category SQL methods. CLI + API /report share this module, so
category rules stay in sync.
"""
from typing import Literal

from flow.ket_partner.config import KetConfig

Category = Literal["mastered", "learning", "struggling", "used", "unused"]

CATEGORIES: tuple[str, ...] = ("mastered", "learning", "struggling", "used", "unused")


def classify_row(
    row: dict,
    struggling_wc_min: int,
    struggling_ec_min: int,
) -> Category:
    """Pure classification rule. Order matters — earlier branches win."""
    if row["exposed_count"] == 0:
        return "unused"
    if row["status"] == "mastered":
        return "mastered"
    if row["status"] == "learning":
        return "learning"
    if (
        row["wrong_count"] >= struggling_wc_min
        or (row["exposed_count"] >= struggling_ec_min and row["mastery_score"] == 0)
    ):
        return "struggling"
    return "used"


def classify(row: dict, cfg: KetConfig) -> Category:
    """Convenience wrapper using cfg thresholds."""
    return classify_row(
        row,
        struggling_wc_min=cfg.struggling_threshold.wrong_count_min,
        struggling_ec_min=cfg.struggling_threshold.exposed_count_min,
    )


def group_by_category(
    rows: list[dict],
    cfg: KetConfig,
) -> dict[str, list[dict]]:
    """Bucket rows into the 5 categories in one pass."""
    bucket: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for r in rows:
        bucket[classify(r, cfg)].append(r)
    return bucket
