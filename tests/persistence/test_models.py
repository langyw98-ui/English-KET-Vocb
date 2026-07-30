# tests/persistence/test_models.py
from src.persistence.models import MASTERY_CAP, WordRef, derive_status


def test_derive_status_mastered_at_score_cap():
    assert derive_status("learning", 2) == "mastered"
    assert derive_status("exposed", 2) == "mastered"
    assert derive_status("mastered", 5) == "mastered"


def test_derive_status_mastered_demotes_at_score_1_or_below():
    assert derive_status("mastered", 1) == "learning"
    assert derive_status("mastered", 0) == "learning"


def test_derive_status_is_target_promotes_to_learning():
    assert derive_status("exposed", 0, is_target=True) == "learning"
    assert derive_status("exposed", 1, is_target=True) == "learning"
    assert derive_status(None, 0, is_target=True) == "learning"


def test_derive_status_new_row_not_target_is_exposed():
    assert derive_status(None, 0) == "exposed"


def test_derive_status_preserves_exposed_and_learning_below_cap():
    assert derive_status("exposed", 1) == "exposed"
    assert derive_status("exposed", 2) == "mastered"
    assert derive_status("learning", 1) == "learning"
    assert derive_status("learning", 2) == "mastered"


def test_mastery_cap_is_two():
    assert MASTERY_CAP == 2


def test_wordref_defaults_empty_context():
    ref = WordRef(word="cat")
    assert ref.word == "cat"
    assert ref.context == ""
