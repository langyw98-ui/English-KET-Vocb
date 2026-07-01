import pytest

from flow.ket_partner.commands import CommandHandler, ExitLoop


@pytest.mark.asyncio
async def test_exit_command_raises_exitloop():
    handler = CommandHandler(db_path="dummy", chat_logger=None)
    with pytest.raises(ExitLoop):
        await handler.handle("/exit")


@pytest.mark.asyncio
async def test_quit_command_raises_exitloop():
    handler = CommandHandler(db_path="dummy", chat_logger=None)
    with pytest.raises(ExitLoop):
        await handler.handle("/quit")


@pytest.mark.asyncio
async def test_help_command_returns_text(capsys):
    handler = CommandHandler(db_path="dummy", chat_logger=None)
    await handler.handle("/help")
    captured = capsys.readouterr()
    assert "exportstats" in captured.out
    assert "exit" in captured.out
