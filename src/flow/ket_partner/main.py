import asyncio
import os
from os.path import dirname, join

from langchain.messages import HumanMessage

from flow.common import llm_flash, llm_max, logger
from flow.ket_partner.agent import build_agent
from flow.ket_partner.chat_logger import ChatLogger
from flow.ket_partner.commands import CommandHandler, ExitLoop
from flow.ket_partner.db import Repos, init_db

DEFAULT_DB = "ket_partner.db"
DEFAULT_CSV = join(dirname(__file__), "..", "..", "..", "data", "KET_vocabulary.csv")


async def main():
    info = {
        "nickname_kid": os.environ.get("KID_NICKNAME", "宝贝"),
        "age": int(os.environ.get("KID_AGE", "8")),
    }
    db_path = os.environ.get("KET_DB_PATH", DEFAULT_DB)
    csv_path = DEFAULT_CSV if os.path.exists(DEFAULT_CSV) else None

    db = await init_db(
        db_path,
        csv_path=csv_path,
        default_nickname=info["nickname_kid"],
        default_age=info["age"],
    )
    repos = Repos.for_user(db, "default")
    await repos.log.append_session_start()
    agent = await build_agent(llm_flash, llm_max, db)

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
            config = {
                "configurable": {
                    "thread_id": "main",
                    "user_id": "default",
                    "repos": repos,
                    "user_info": {"nickname": info["nickname_kid"], "age": info["age"]},
                }
            }
            response = await agent.ainvoke(
                {"messages": messages[-5:]},
                config=config,
            )
            ai_reply = response["messages"][-1].content
            messages.append(response["messages"][-1])

            chat_logger.log_turn(turn_id, "user", user_input)
            chat_logger.log_turn(turn_id, "AI", ai_reply)
            print(f"AI: {ai_reply}\n")
            turn_id += 1
    finally:
        agent_instance = getattr(agent, "agent", None)
        if agent_instance is not None:
            try:
                await agent_instance.aclose()
            except Exception as e:
                logger.warning(f"agent.aclose() failed during shutdown: {e}")
        await db.close()
        chat_logger.close_session()


if __name__ == "__main__":
    asyncio.run(main())
