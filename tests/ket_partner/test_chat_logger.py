from flow.ket_partner.chat_logger import ChatLogger


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
