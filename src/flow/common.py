from os import environ
from httpx import AsyncClient
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI

if environ.get("PYTEST_VERSION") is not None:
    from logging import getLogger

    logger = getLogger()
    IS_RUNNING_IN_PYTEST = True
else:
    from loguru import logger

    IS_RUNNING_IN_PYTEST = False

extra_params = {"enable_thinking": False}

dashscope_api_key = environ.get("DASHSCOPE_API_KEY", "")
llm_plus = ChatOpenAI(
    api_key=dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.6-plus",
    client=AsyncOpenAI(
        api_key=dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        http_client=AsyncClient(),
    ),
    temperature=0,
    extra_body=extra_params,
)

llm_max = ChatOpenAI(
    api_key=dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.6-max-preview",
    client=AsyncOpenAI(
        api_key=dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        http_client=AsyncClient(),
    ),
    temperature=0,
    extra_body=extra_params,
)

llm_flash = ChatOpenAI(
    api_key=dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.6-flash",
    client=AsyncOpenAI(
        api_key=dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        http_client=AsyncClient(),
    ),
    temperature=0.8,
    top_p=0.8,
    extra_body={"enable_thinking": False},
)

doubao_api = environ.get("DOUBAO_API_KEY", "")
llm_doubao = ChatOpenAI(
    api_key=doubao_api,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # model="doubao-seed-1-6-251015",
    # model="doubao-seed-1-6-lite-251015",
    model="doubao-seed-1-6-flash-250828",
    # async_client=AsyncClient(),
    client=AsyncOpenAI(
        api_key=doubao_api,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        http_client=AsyncClient(),
    ),
    temperature=0.8,
    top_p=0.8,
)
