"""KETPartnerAgent 与 LlmService DI 集成测试。

Scope: 验证 KETPartnerAgent 通过 LlmService Protocol 接收 LLM 实例,
并通过 @property 暴露 llm_smart/llm_flash,而非直接存储模型实例。
这是 Phase 2 重构的核心契约——业务代码必须依赖 LlmService 抽象,
不绕过它直接持有 BaseChatModel。
"""
from unittest.mock import MagicMock, patch

from langchain_core.language_models.chat_models import BaseChatModel


def test_agent_exposes_llm_smart_and_flash_via_property():
    """KETPartnerAgent 必须通过 @property 暴露 llm_smart/llm_flash,
    不能直接存 self.llm_smart(避免绕开 LlmService 抽象)。"""
    from flow.ket_partner.agent import KETPartnerAgent
    from flow.ket_partner.config import KetConfig

    mock_svc = MagicMock()
    mock_svc.smart = MagicMock(spec=BaseChatModel)
    mock_svc.flash = MagicMock(spec=BaseChatModel)

    agent = KETPartnerAgent(mock_svc, KetConfig())

    # property 应该返回 llm_service.smart/flash
    assert agent.llm_smart is mock_svc.smart
    assert agent.llm_flash is mock_svc.flash
    # 内部存的是 service,不是直接的 llm
    assert agent._llm_service is mock_svc


async def test_build_agent_uses_injected_llm_service():
    """build_agent 接受 llm_service 参数,不应使用模块级 default_llm_service。"""
    from flow.ket_partner.graph import build_agent

    mock_svc = MagicMock()
    mock_svc.smart = MagicMock(spec=BaseChatModel)
    mock_svc.flash = MagicMock(spec=BaseChatModel)

    with patch("flow.ket_partner.graph.default_llm_service") as mock_default:
        # 让 default 抛异常,确保它没被调用
        mock_default.side_effect = AssertionError("不应使用 default_llm_service")
        graph = await build_agent(llm_service=mock_svc)

    # graph.agent 是 build_agent 内挂上的(KETPartnerAgent 实例)
    inner_agent = getattr(graph, "agent", None)
    assert inner_agent is not None
    assert inner_agent._llm_service is mock_svc
