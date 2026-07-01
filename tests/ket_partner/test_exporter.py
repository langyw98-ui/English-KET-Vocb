import pytest

from flow.ket_partner.config import load_config
from flow.ket_partner.db import init_db
from flow.ket_partner.exporter import export_learning_report


@pytest.fixture
async def repos(temp_db_path):
    csv_text = "word,part_of_speech,topic\ncat,n,Animals\ndog,n,Animals\n"
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = f.name
    r = await init_db(temp_db_path, csv_path=csv_path)
    await r.stats.apply_delta("cat", delta=3, exposed=True)
    await r.stats.apply_delta("dog", delta=1, exposed=True)
    yield r
    await r.close()


@pytest.mark.asyncio
async def test_export_markdown(repos, tmp_path):
    out = tmp_path / "report.md"
    cfg = load_config()
    result = await export_learning_report(str(out), repos, cfg)
    content = out.read_text(encoding="utf-8")
    assert "已掌握" in content
    assert "cat" in content
    assert "正在学习" in content
    assert "dog" in content
