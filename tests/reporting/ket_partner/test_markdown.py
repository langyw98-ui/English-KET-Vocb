# tests/reporting/ket_partner/test_markdown.py
from src.reporting.ket_partner.markdown import fmt_word, render_markdown


def test_fmt_word_no_context():
    assert fmt_word("cat", "") == "cat"


def test_fmt_word_with_context():
    assert fmt_word("bank", "Finance") == "bank(Finance)"


def test_render_markdown_renders_all_sections():
    profile = {"nickname": "小明", "total_turns": 42}
    rows_by_category = {
        "mastered": [{"word": "cat", "context": "", "pos": "n", "exposed_count": 5,
                      "correct_count": 5, "wrong_count": 0, "mastery_score": 2}],
        "learning": [{"word": "dog", "context": "", "pos": "n", "exposed_count": 3,
                      "correct_count": 1, "wrong_count": 1, "mastery_score": 1}],
        "struggling": [],
        "used": [{"word": "the", "context": "", "pos": "det", "exposed_count": 10,
                  "correct_count": 8, "wrong_count": 2, "mastery_score": 1}],
        "unused": [{"word": "ghost", "context": "", "pos": "n",
                    "exposed_count": 0, "correct_count": 0,
                    "wrong_count": 0, "mastery_score": 0}],
    }
    out = render_markdown(profile, rows_by_category)
    assert "学习报告 - 小明" in out
    assert "总轮数: 42" in out
    assert "正在学习 (1 项)" in out
    assert "已掌握 (1 项)" in out
    assert "已使用 (1 项)" in out
    assert "未使用 (1 项)" in out
    assert "学习困难 (0 项)" in out
    assert "cat" in out and "dog" in out


def test_render_markdown_handles_empty_buckets():
    profile = {"nickname": None, "total_turns": 0}
    empty = {c: [] for c in ("mastered", "learning", "struggling", "used", "unused")}
    out = render_markdown(profile, empty)
    assert "学习报告 - 小朋友" in out  # None nickname → '小朋友'
