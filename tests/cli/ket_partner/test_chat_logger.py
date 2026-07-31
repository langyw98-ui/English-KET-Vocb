import pytest

from src.cli.ket_partner.chat_logger import ChatLogger


def test_chat_logger_creates_file(tmp_path):
    logger = ChatLogger(log_dir=str(tmp_path))
    logger.start_session("宝贝")
    logger.log_turn(1, "user", "hello")
    logger.log_turn(1, "AI", "Hi there!")
    logger.close_session()
    files = list(tmp_path.glob("chat_log_*.txt"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "宝贝" in content
    assert "hello" in content
    assert "Hi there!" in content


def test_chat_logger_auto_increments(tmp_path):
    a = ChatLogger(log_dir=str(tmp_path))
    a.start_session("A")
    a.close_session()
    b = ChatLogger(log_dir=str(tmp_path))
    b.start_session("B")
    b.close_session()
    files = sorted(tmp_path.glob("chat_log_*.txt"))
    assert len(files) == 2


def test_chat_logger_exit_closes_session_on_exception(tmp_path):
    """with 块内抛异常时,__exit__ 仍应调 close_session,释放文件 handle。"""
    log_dir = str(tmp_path / "logs")
    logger = ChatLogger(log_dir=log_dir)
    logger.start_session("test")

    fp_ref = logger._fp
    assert fp_ref is not None

    with pytest.raises(ValueError):
        with logger:
            raise ValueError("simulated failure")

    # __exit__ 应已关闭 fp
    assert fp_ref.closed is True
    assert logger._fp is None


def test_chat_logger_exit_idempotent(tmp_path):
    """__exit__ 多次调用不重复关 fp(避免 ValueError: I/O operation on closed file)。"""
    logger = ChatLogger(log_dir=str(tmp_path / "logs"))
    logger.start_session("test")
    with logger:
        pass
    # 二次 close 不应抛异常
    logger.close_session()
