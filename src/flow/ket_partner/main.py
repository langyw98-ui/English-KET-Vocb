import asyncio
import os
from os.path import dirname, join

from langchain.messages import HumanMessage

from flow.agent import memory
from flow.common import logger
from flow.ket_partner.agent import autonomous
from flow.ket_partner.chat_logger import ChatLogger
from flow.ket_partner.commands import CommandHandler, ExitLoop

DEFAULT_DB = "ket_partner.db"
DEFAULT_CSV = join(dirname(__file__), "..", "..", "..", "data", "KET_vocabulary.csv")


async def main():
    info = {
        "nickname_kid": os.environ.get("KID_NICKNAME", "宝贝"),
        "age": int(os.environ.get("KID_AGE", "8")),
    }
    db_path = os.environ.get("KET_DB_PATH", DEFAULT_DB)
    csv_path = DEFAULT_CSV if os.path.exists(DEFAULT_CSV) else None

    agent = await autonomous(info, db_path=db_path, csv_path=csv_path)
    chat_logger = ChatLogger(log_dir="logs/chat")
    chat_logger.start_session(info["nickname_kid"])
    cmd_handler = CommandHandler(db_path, chat_logger)

    messages = []
    turn_id = 1
    try:
        while True:
            user_input = await asyncio.to_thread(input, "用户输入: ")
            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                try:
                    await cmd_handler.handle(user_input)
                except ExitLoop:
                    break
                continue

            messages.append(HumanMessage(content=user_input))
            response = await agent.ainvoke(
                {"messages": messages[-5:]},
                config={"configurable": {"thread_id": "main"}},
            )
            ai_reply = response["messages"][-1].content
            messages.append(response["messages"][-1])

            chat_logger.log_turn(turn_id, "user", user_input)
            chat_logger.log_turn(turn_id, "AI", ai_reply)
            print(f"AI: {ai_reply}\n")
            turn_id += 1
    finally:
        # Drain any in-flight background summary task before the loop closes.
        # Without this, a task scheduled on the turn that triggered /exit is
        # silently dropped (I3).
        agent_instance = getattr(agent, "agent", None)
        if agent_instance is not None:
            try:
                await agent_instance.aclose()
            except Exception as e:
                logger.warning(f"agent.aclose() failed during shutdown: {e}")
        chat_logger.close_session()


if __name__ == "__main__":
    asyncio.run(main())
