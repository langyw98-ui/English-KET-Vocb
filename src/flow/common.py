from logging import getLogger
from os import environ
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml
from dotenv import load_dotenv
from httpx import AsyncClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic import SecretStr

load_dotenv()

logger = getLogger("ket_partner")

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
    extra_body=extra_params,
)


@runtime_checkable
class LlmService(Protocol):
    """LLM 服务抽象。业务代码依赖此 Protocol,便于测试注入 mock。

    Implementations: DashScopeLlmService(默认),未来可加 MockLlmService 等。

    Attributes:
        smart: 慢思考模型(BaseChatModel),用于难度判断等需要稳定输出的场景。
        flash: 快思考模型(BaseChatModel),用于轻量分类、解析等场景。
    """

    smart: BaseChatModel
    flash: BaseChatModel


class DashScopeLlmService(LlmService):
    """DashScope(Qwen 兼容模式)具体实现。封装 ChatOpenAI 实例化细节。

    每次 __init__ 创建独立的 ChatOpenAI + AsyncOpenAI + AsyncClient,
    避免不同 service 实例之间共享底层 HTTP client。
    """

    def __init__(self) -> None:
        api_key = _resolve_dashscope_api_key()
        # 复用 llm_max / llm_flash 的实例化契约:相同 base_url、相同 extra_body、
        # 相同 placeholder fallback;但每次调用都新建独立 client/http_client。
        self.smart = _build_dashscope_chat(
            api_key=api_key,
            model="qwen3.6-max-preview",
            temperature=0,
        )
        self.flash = _build_dashscope_chat(
            api_key=api_key,
            model="qwen3.6-flash",
            temperature=0.8,
            top_p=0.8,
        )


def _build_dashscope_chat(
    *,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float | None = None,
) -> ChatOpenAI:
    """工厂函数:每次调用新建一个独立 DashScope ChatOpenAI 实例。

    抽取自 llm_max / llm_flash 的共享构造逻辑,避免复制粘贴导致字段漂移。
    保持与模块级 llm_max / llm_flash 完全一致的实例化参数契约。
    """
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    kwargs: dict[str, Any] = {
        "api_key": SecretStr(api_key or "placeholder"),
        "base_url": base_url,
        "model": model,
        "client": AsyncOpenAI(
            api_key=api_key or "placeholder",
            base_url=base_url,
            http_client=AsyncClient(),
        ),
        "temperature": temperature,
        "extra_body": extra_params,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p
    return ChatOpenAI(**kwargs)


default_llm_service: LlmService = DashScopeLlmService()
