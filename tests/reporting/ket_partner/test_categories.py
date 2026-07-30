# tests/reporting/ket_partner/test_categories.py
from unittest.mock import MagicMock

from src.reporting.ket_partner.categories import (
    CATEGORIES,
    classify,
    classify_row,
    group_by_category,
)


def _cfg(wc_min=2, ec_min=5):
    cfg = MagicMock()
    cfg.struggling_threshold.wrong_count_min = wc_min
    cfg.struggling_threshold.exposed_count_min = ec_min
    return cfg


def test_classify_row_unused_when_unexposed():
    row = {"exposed_count": 0, "status": "new", "wrong_count": 0, "mastery_score": 0}
    assert classify_row(row, struggling_wc_min=2, struggling_ec_min=5) == "unused"


def test_classify_row_mastered():
    row = {"exposed_count": 5, "status": "mastered", "wrong_count": 0, "mastery_score": 2}
    assert classify_row(row, struggling_wc_min=2, struggling_ec_min=5) == "mastered"


def test_classify_row_learning():
    row = {"exposed_count": 3, "status": "learning", "wrong_count": 1, "mastery_score": 1}
    assert classify_row(row, struggling_wc_min=2, struggling_ec_min=5) == "learning"


def test_classify_row_struggling_by_wrong_count():
    row = {"exposed_count": 3, "status": "new", "wrong_count": 2, "mastery_score": 1}
    assert classify_row(row, struggling_wc_min=2, struggling_ec_min=5) == "struggling"


def test_classify_row_struggling_by_exposed_with_zero_mastery():
    row = {"exposed_count": 5, "status": "new", "wrong_count": 1, "mastery_score": 0}
    assert classify_row(row, struggling_wc_min=2, struggling_ec_min=5) == "struggling"


def test_classify_row_used_when_below_struggling_thresholds():
    row = {"exposed_count": 3, "status": "new", "wrong_count": 1, "mastery_score": 1}
    assert classify_row(row, struggling_wc_min=2, struggling_ec_min=5) == "used"


def test_classify_uses_cfg_thresholds():
    row = {"exposed_count": 3, "status": "new", "wrong_count": 1, "mastery_score": 1}
    assert classify(row, _cfg(wc_min=2, ec_min=5)) == "used"
    # Raise the bar so this row is struggling
    assert classify(row, _cfg(wc_min=1, ec_min=5)) == "struggling"


def test_group_by_category_returns_all_five_buckets():
    rows = [
        {"word": "a", "exposed_count": 0, "status": "new", "wrong_count": 0, "mastery_score": 0},
        {"word": "b", "exposed_count": 5, "status": "mastered", "wrong_count": 0, "mastery_score": 2},
        {"word": "c", "exposed_count": 3, "status": "learning", "wrong_count": 1, "mastery_score": 1},
        {"word": "d", "exposed_count": 3, "status": "new", "wrong_count": 2, "mastery_score": 1},
        {"word": "e", "exposed_count": 3, "status": "new", "wrong_count": 1, "mastery_score": 1},
    ]
    bucket = group_by_category(rows, _cfg())
    assert set(bucket.keys()) == set(CATEGORIES)
    assert len(bucket["unused"]) == 1
    assert len(bucket["mastered"]) == 1
    assert len(bucket["learning"]) == 1
    assert len(bucket["struggling"]) == 1
    assert len(bucket["used"]) == 1
