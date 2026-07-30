"""LlmService Protocol + DashScopeLlmService 测试。"""
from unittest.mock import MagicMock

from langchain_core.language_models.chat_models import BaseChatModel


def test_default_llm_service_exposes_smart_and_flash():
    """default_llm_service 单例必须暴露 smart 和 flash 两个 BaseChatModel。"""
    from flow.common import default_llm_service

    assert isinstance(default_llm_service.smart, BaseChatModel)
    assert isinstance(default_llm_service.flash, BaseChatModel)


def test_dashscope_llm_service_creates_independent_instances():
    """DashScopeLlmService 每次实例化创建独立 ChatOpenAI(避免共享 client)。"""
    from flow.common import DashScopeLlmService

    svc1 = DashScopeLlmService()
    svc2 = DashScopeLlmService()
    assert svc1.smart is not svc2.smart
    assert svc1.flash is not svc2.flash


def test_llm_service_protocol_accepts_mock():
    """LlmService Protocol 必须能接受任意含 smart/flash 属性的对象(便于 DI mock)。"""
    from flow.common import LlmService

    mock_svc = MagicMock()
    mock_svc.smart = MagicMock(spec=BaseChatModel)
    mock_svc.flash = MagicMock(spec=BaseChatModel)
    # Protocol 是结构性子类型,只要属性匹配就通过
    assert isinstance(mock_svc, LlmService)
