import tempfile

import pytest

from flow.ket_partner.db import init_db
from flow.ket_partner.sentence_validator import validate_sentence


@pytest.fixture
async def repos(temp_db_path):
    csv_text = (
        "word,part_of_speech,topic\n"
        "cat,n,Animals\n"
        "dog,n,Animals\n"
        "big,adj,\n"
        "small,adj,\n"
        "is,v,\n"
        "the,det,\n"
        "a,det,\n"
        "on,prep,\n"
        "bed,n,\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = f.name
    r = await init_db(temp_db_path, csv_path=csv_path)
    yield r
    await r.close()


@pytest.mark.asyncio
async def test_validate_all_ket_words_passes(repos):
    result = await validate_sentence("The cat is on the bed.", repos)
    assert result.ok is True
    assert "cat" in result.words_used


@pytest.mark.asyncio
async def test_validate_non_ket_word_fails(repos):
    result = await validate_sentence("The cat is on the elephant.", repos)
    assert result.ok is False
    assert "elephant" in result.non_ket_words


@pytest.mark.asyncio
async def test_validate_lemma_reduces_to_ket_root(repos):
    result = await validate_sentence("The cats are on the bed.", repos)
    # 'cats' lemmatizes to 'cat' which is KET
    assert result.ok is True
