from logging import getLogger
from os import environ
from pathlib import Path

import yaml
from dotenv import load_dotenv
from httpx import AsyncClient
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic import SecretStr

load_dotenv()

logger = getLogger("ket_partner")

if environ.get("PYTEST_VERSION") is not None:
    IS_RUNNING_IN_PYTEST = True
else:
    IS_RUNNING_IN_PYTEST = False

extra_params = {"enable_thinking": False}


def _resolve_dashscope_api_key() -> str:
    # 1. 优先读取环境变量
    for env_var in ["DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"]:
        val = environ.get(env_var)
        if val and val.strip():
            return val.strip()

    # 2. 读取用户家目录下的 ~/.config/pet/config.yaml 配置文件
    pet_config = Path.home() / ".config" / "pet" / "config.yaml"
    if pet_config.exists():
        try:
            with open(pet_config, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                keys_to_check = [
                    "dashscope_api_key",
                    "daskscope_api_key",
                    "DASHSCOPE_API_KEY",
                    "AKE_LLM_LOW_API_KEY",
                    "AKE_LLM_MID_API_KEY",
                    "AKE_LLM_HIGH_API_KEY",
                    "api_key",
                ]
                for k in keys_to_check:
                    val = data.get(k)
                    if val and isinstance(val, str) and val.strip():
                        return val.strip()
                for k, v in data.items():
                    if "api_key" in str(k).lower() and isinstance(v, str) and v.strip():
                        return v.strip()
        except (yaml.YAMLError, OSError) as e:
            logger.warning(f"Failed to load API key from {pet_config}: {e}", exc_info=True)

    return ""


dashscope_api_key = _resolve_dashscope_api_key()
llm_plus = ChatOpenAI(
    api_key=SecretStr(dashscope_api_key or "placeholder"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.6-plus",
    client=AsyncOpenAI(
        api_key=dashscope_api_key or "placeholder",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        http_client=AsyncClient(),
    ),
    temperature=0,
    extra_body=extra_params,
)

llm_max = ChatOpenAI(
    api_key=SecretStr(dashscope_api_key or "placeholder"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.6-max-preview",
    client=AsyncOpenAI(
        api_key=dashscope_api_key or "placeholder",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        http_client=AsyncClient(),
    ),
    temperature=0,
    extra_body=extra_params,
)

llm_flash = ChatOpenAI(
    api_key=SecretStr(dashscope_api_key or "placeholder"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.6-flash",
    client=AsyncOpenAI(
        api_key=dashscope_api_key or "placeholder",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        http_client=AsyncClient(),
    ),
    temperature=0.8,
    top_p=0.8,
    extra_body={"enable_thinking": False},
)

doubao_api = environ.get("DOUBAO_API_KEY", "")
llm_doubao = ChatOpenAI(
    api_key=SecretStr(doubao_api or "placeholder"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # model="doubao-seed-1-6-251015",
    # model="doubao-seed-1-6-lite-251015",
    model="doubao-seed-1-6-flash-250828",
    # async_client=AsyncClient(),
    client=AsyncOpenAI(
        api_key=doubao_api or "placeholder",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        http_client=AsyncClient(),
    ),
    temperature=0.8,
    top_p=0.8,
)
