import pytest

from flow.ket_partner.config import load_config
from flow.ket_partner.db import init_db
from flow.ket_partner.exporter import export_learning_report


@pytest.fixture
async def repos(temp_db_path):
    csv_text = "word,part_of_speech,topic\ncat,n,Animals\ndog,n,Animals\nbird,n,Animals\nfish,n,Animals\nfox,n,Animals\n"
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = f.name
    r = await init_db(temp_db_path, csv_path=csv_path)
    await r.stats.apply_delta("cat", delta=3, exposed=True)   # → mastered (caps at MASTERY_CAP)
    await r.stats.apply_delta("dog", delta=1, exposed=True)   # → used (exposed, not learning/mastered/struggling)
    # fox: target exposure with sub-cap mastery → status='learning'
    await r.stats.apply_delta("fox", delta=1, exposed=True, is_target=True)
    # fish: exposed 5x, all wrong, mastery stuck at 0 → struggling
    for _ in range(5):
        await r.stats.apply_delta("fish", delta=-1, exposed=True)
    # bird intentionally left with no stats → unused
    yield r
    await r.close()


@pytest.mark.asyncio
async def test_export_markdown(repos, tmp_path):
    out = tmp_path / "report.md"
    cfg = load_config()
    result = await export_learning_report(str(out), repos, cfg)
    content = out.read_text(encoding="utf-8")
    # 5-table structure with counts in headers.
    assert "## 正在学习 (1 词)" in content
    assert "## 已掌握 (1 词)" in content
    assert "## 已使用 (1 词)" in content
    assert "## 未使用 (1 词)" in content
    assert "## 学习困难 (1 词)" in content
    # Each word lands in its expected table.
    assert "cat" in content
    assert "dog" in content
    assert "bird" in content
    assert "fish" in content
    assert "fox" in content
