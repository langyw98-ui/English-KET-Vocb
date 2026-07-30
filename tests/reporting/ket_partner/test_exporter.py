import pytest

from flow.ket_partner.config import load_config
from src.persistence.bootstrap import init_db
from src.persistence.repos import Repos
from src.reporting.ket_partner.exporter import export_learning_report


@pytest.fixture
async def repos(temp_db_path):
    csv_text = "word,part_of_speech,topic,context\ncat,n,Animals,\ndog,n,Animals,\nbird,n,Animals,\nfish,n,Animals,\nfox,n,Animals,\n"
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = f.name
    db = await init_db(temp_db_path, csv_path=csv_path)
    r = Repos.for_user(db, "default")
    await r.stats.apply_delta("cat", delta=3, exposed=True)   # → mastered (caps at MASTERY_CAP)
    await r.stats.apply_delta("dog", delta=1, exposed=True)   # → used (exposed, not learning/mastered/struggling)
    # fox: target exposure with sub-cap mastery → status='learning'
    await r.stats.apply_delta("fox", delta=1, exposed=True, is_target=True)
    # fish: exposed 5x, all wrong, mastery stuck at 0 → struggling
    for _ in range(5):
        await r.stats.apply_delta("fish", delta=-1, exposed=True)
    # bird intentionally left with no stats → unused
    yield r
    await db.close()


@pytest.mark.asyncio
async def test_export_markdown(repos, tmp_path):
    out = tmp_path / "report.md"
    cfg = load_config()
    await export_learning_report(str(out), repos, cfg)
    content = out.read_text(encoding="utf-8")
    # 5-table structure with "项" counts (was "词" pre-migration).
    assert "## 正在学习 (1 项)" in content
    assert "## 已掌握 (1 项)" in content
    assert "## 已使用 (1 项)" in content
    assert "## 未使用 (1 项)" in content
    assert "## 学习困难 (1 项)" in content
    # Each word lands in its expected table.
    assert "cat" in content
    assert "dog" in content
    assert "bird" in content
    assert "fish" in content
    assert "fox" in content


@pytest.mark.asyncio
async def test_export_renders_word_with_context(repos, tmp_path):
    """Spec §10: when a word has a non-empty context, the word column shows
    'word(context)'; otherwise just 'word'. No separate context column."""
    # Add a multi-sense row alongside cat (already in fixture at default sense).
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('cat', 'animal', 'n', 0)"
    )
    # Touch the (cat, animal) row so it lands in a non-unused bucket.
    await repos.stats.apply_delta("cat", context="animal", delta=1, exposed=True)
    await repos.vocab._db.commit()
    out = tmp_path / "report.md"
    cfg = load_config()
    await export_learning_report(str(out), repos, cfg)
    content = out.read_text(encoding="utf-8")
    # Both rows appear: "cat" for default, "cat(animal)" for the specific sense.
    assert "| cat |" in content or "| cat " in content
    assert "cat(animal)" in content


@pytest.mark.asyncio
async def test_render_report_text_returns_markdown_without_writing(temp_db_path):
    """Spec §6: render_report_text is the in-memory variant for API /report."""
    from src.persistence.bootstrap import init_db
    from src.persistence.repos import Repos
    from src.reporting.ket_partner.exporter import render_report_text

    db = await init_db(temp_db_path, csv_path=None)
    repos = Repos.for_user(db, "default")
    cfg = load_config()
    text = await render_report_text(repos, cfg)
    assert "学习报告" in text
    await repos.close()
