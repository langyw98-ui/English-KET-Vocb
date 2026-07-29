import pytest

from src.api.llm_key import LlmKeyStatus


def test_state_green_when_key_present_and_no_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.api.llm_key._read_current_key", lambda: "sk-xxx")
    status = LlmKeyStatus()
    assert status.state == "green"


def test_state_red_when_key_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.api.llm_key._read_current_key", lambda: "")
    status = LlmKeyStatus()
    assert status.state == "red"


def test_state_red_when_error_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.api.llm_key._read_current_key", lambda: "sk-xxx")
    status = LlmKeyStatus()
    status.set_error("auth error")
    assert status.state == "red"


def test_state_green_after_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.api.llm_key._read_current_key", lambda: "sk-xxx")
    status = LlmKeyStatus()
    status.set_error("auth error")
    status.clear_error()
    assert status.state == "green"


def test_state_ignores_whitespace_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.api.llm_key._read_current_key", lambda: "   ")
    status = LlmKeyStatus()
    assert status.state == "red"


def test_set_error_with_newer_timestamp_overwrites() -> None:
    status = LlmKeyStatus()
    status.set_error("old error", timestamp=100.0)
    status.set_error("new error", timestamp=200.0)
    assert status.last_error == "new error"
    assert status.last_error_updated_at == 200.0


def test_set_error_with_older_timestamp_does_not_overwrite() -> None:
    status = LlmKeyStatus()
    status.set_error("first error", timestamp=200.0)
    status.set_error("second error", timestamp=100.0)
    assert status.last_error == "first error"
    assert status.last_error_updated_at == 200.0


def test_clear_error_with_older_timestamp_does_not_clear() -> None:
    status = LlmKeyStatus()
    status.set_error("some error", timestamp=200.0)
    status.clear_error(timestamp=100.0)
    assert status.last_error == "some error"
    assert status.last_error_updated_at == 200.0


def test_clear_error_with_newer_timestamp_clears() -> None:
    status = LlmKeyStatus()
    status.set_error("some error", timestamp=100.0)
    status.clear_error(timestamp=200.0)
    assert status.last_error is None
    assert status.last_error_updated_at == 200.0
