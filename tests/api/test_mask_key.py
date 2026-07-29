from src.api.llm_key import mask_key


def test_mask_key_normal() -> None:
    assert mask_key("sk-abcdefghijklmno") == "sk-a***lmno"


def test_mask_key_short() -> None:
    assert mask_key("abc") == "***bc"


def test_mask_key_empty() -> None:
    assert mask_key("") is None


def test_mask_key_whitespace() -> None:
    assert mask_key("   ") is None


def test_mask_key_boundary_8() -> None:
    assert mask_key("abcdefgh") == "abcd***efgh"


def test_mask_key_boundary_7() -> None:
    assert mask_key("abcdefg") == "***fg"


def test_mask_key_strips_whitespace() -> None:
    assert mask_key("  sk-abcdefghijklmno  ") == "sk-a***lmno"
