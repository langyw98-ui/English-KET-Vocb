import tempfile

import pytest

from flow.ket_partner.db import init_db
from flow.ket_partner.sentence_validator import validate_sentence


@pytest.fixture
async def repos(temp_db_path):
    csv_text = (
        "word,part_of_speech,topic,context\n"
        "cat,n,Animals,\n"
        "dog,n,Animals,\n"
        "big,adj,,\n"
        "small,adj,,\n"
        "happy,adj,,\n"
        "is,v,,\n"
        "am,v,,\n"
        "are,v,,\n"
        "go,v,Action,\n"
        "I,pron,,\n"
        "the,det,,\n"
        "a,det,,\n"
        "on,prep,,\n"
        "bed,n,,\n"
        "hat,n,Clothing,\n"
        "cake,n,Food,\n"
        "wear,v,Clothing,\n"
        "watch,v,Action,\n"
        "walk,v,Action,\n"
        "run,v,Action,\n"
        "bake,v,Food,\n"
        "bake,v,Food,\n"
        "story,n,Reading,\n"
        "try,v,Action,\n"
        "large,adj,Size,\n"
        "make,v,Action,\n"
        "rain,n,Weather,\n"
        "grass,n,Nature,\n"
        "wet,adj,,\n"
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


@pytest.mark.asyncio
async def test_validate_handles_verb_inflections(repos):
    """Regular verb tense inflections must reduce to the KET root.
    Without this, generated sentences like 'wears' / 'walked' / 'running'
    were misclassified as non-KET.
    """
    # 3rd-person singular -s
    r1 = await validate_sentence("The cat wears a hat.", repos)
    assert r1.ok is True, f"wears should reduce to wear; got non_ket={r1.non_ket_words}"
    assert "wear" in r1.words_used
    # past tense -ed
    r2 = await validate_sentence("The dog walked.", repos)
    assert r2.ok is True
    assert "walk" in r2.words_used
    # -ed where verb ends in 'e' (drop only d): baked → bake
    r3 = await validate_sentence("The big cake baked.", repos)
    assert r3.ok is True, f"baked should reduce to bake; got non_ket={r3.non_ket_words}"
    assert "bake" in r3.words_used
    # gerund -ing with doubled consonant: running → run
    r4 = await validate_sentence("The dog is running.", repos)
    assert r4.ok is True, f"running should reduce to run; got non_ket={r4.non_ket_words}"
    assert "run" in r4.words_used
    # -ing where verb ends in 'e': baking → bake
    r5 = await validate_sentence("The big cake is baking.", repos)
    assert r5.ok is True, f"baking should reduce to bake; got non_ket={r5.non_ket_words}"
    assert "bake" in r5.words_used


@pytest.mark.asyncio
async def test_validate_handles_noun_plurals(repos):
    # -ies → y: stories → story
    r1 = await validate_sentence("The happy stories.", repos)
    assert r1.ok is True, f"stories should reduce to story; got non_ket={r1.non_ket_words}"
    assert "story" in r1.words_used
    # -es: watches → watch
    r2 = await validate_sentence("The dog watches.", repos)
    assert r2.ok is True, f"watches should reduce to watch; got non_ket={r2.non_ket_words}"
    assert "watch" in r2.words_used


@pytest.mark.asyncio
async def test_validate_handles_comparatives_and_superlatives(repos):
    # -ier → y: happier → happy
    r1 = await validate_sentence("The happier dog walked.", repos)
    assert r1.ok is True, f"happier should reduce to happy; got non_ket={r1.non_ket_words}"
    assert "happy" in r1.words_used
    # -er: bigger → big
    r2 = await validate_sentence("The bigger dog walked.", repos)
    assert r2.ok is True
    assert "big" in r2.words_used
    # -er where adj ends in 'e': larger → large
    r3 = await validate_sentence("The larger dog walked.", repos)
    assert r3.ok is True, f"larger should reduce to large; got non_ket={r3.non_ket_words}"
    assert "large" in r3.words_used
    # -iest → y: happiest → happy
    r4 = await validate_sentence("The happiest dog walked.", repos)
    assert r4.ok is True
    assert "happy" in r4.words_used
    # -est: biggest → big
    r5 = await validate_sentence("The biggest dog walked.", repos)
    assert r5.ok is True
    assert "big" in r5.words_used


@pytest.mark.asyncio
async def test_validate_verb_s_form_after_e_verb(repos):
    """Regression: "makes" ends in "es" so the -es plural rule stripped both
    letters giving "mak" (not a word). The -s branch was unreachable due to
    `elif`, so "makes" was misclassified as non-KET even though "make" is in
    the vocab. Same shape applies to likes/bakes/hopes/etc."""
    r = await validate_sentence("The rain makes the grass wet.", repos)
    assert r.ok is True, f"makes should reduce to make; got non_ket={r.non_ket_words}"
    assert "make" in r.words_used


@pytest.mark.asyncio
async def test_validate_lemma_overrides_rule_based(repos):
    """Explicit lemma entries in lemmas.json (irregular forms) must take
    precedence over rule-based suffix stripping."""
    # 'went' is in lemmas.json → 'go'. Rule-based stripping would give 'wt' or similar.
    r = await validate_sentence("The dog went big.", repos)
    assert "go" in r.words_used, "irregular lemma 'went→go' must take precedence"


@pytest.mark.asyncio
async def test_validate_pronoun_I_case_insensitive(repos):
    """The pronoun 'I' is stored in KET vocab with its canonical capital
    spelling. A sentence starting with 'I' must validate, and the canonical
    form 'I' (not lowercase 'i') must be recorded in words_used so mastery
    tracking reconciles with target-word selection (which uses canonical form)."""
    r = await validate_sentence("I am big.", repos)
    assert r.ok is True, f"'I' must match the vocab row stored as 'I'; got non_ket={r.non_ket_words}"
    assert "I" in r.words_used, "canonical form 'I' must be recorded, not lowercase 'i'"


@pytest.mark.asyncio
async def test_validator_recognizes_word_with_only_specific_contexts_as_ket(repos):
    """Spec §8.2: smart has only (smart, clever) and (smart, stylish) —
    no (smart, '') row. The validator must still recognize 'smart' as KET
    by using get_ket_word_any_context. Without this, sentences containing
    smart get force-regenerated or annotated as non-KET."""
    # Inject the smart rows directly (fixture above doesn't include them).
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('smart', 'clever', 'adj', 0), ('smart', 'stylish', 'adj', 0)"
    )
    await repos.vocab._db.commit()
    r = await validate_sentence("The smart dog runs.", repos)
    assert r.ok is True, f"smart must be KET; got non_ket={r.non_ket_words}"
    assert "smart" in r.words_used
