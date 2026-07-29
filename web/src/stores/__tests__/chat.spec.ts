import { setActivePinia, createPinia } from 'pinia'
import { describe, beforeEach, it, expect, vi } from 'vitest'
import { useChatStore } from '../chat'
import * as clientModule from '../../api/client'

vi.mock('../../api/client', async () => {
  const actual = await vi.importActual<typeof import('../../api/client')>('../../api/client')
  return {
    ...actual,
    api: vi.fn(),
  }
})

describe('chat store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('maps 503 error correctly and pops optimistic message', async () => {
    const apiMock = vi.mocked(clientModule.api)
    apiMock.mockRejectedValueOnce(new clientModule.ApiError(503, 'LLM not configured'))
    apiMock.mockResolvedValueOnce({ state: 'red', masked_key: null, last_error: null })

    const chatStore = useChatStore()
    await expect(chatStore.send('hello')).rejects.toThrow()

    expect(chatStore.error).toBe('LLM 未配置,请联系管理员')
    expect(chatStore.messages.length).toBe(0)
  })

  it('maps 401 error correctly', async () => {
    const apiMock = vi.mocked(clientModule.api)
    apiMock.mockRejectedValueOnce(new clientModule.ApiError(401, 'LLM auth failed'))
    apiMock.mockResolvedValueOnce({ state: 'red', masked_key: 'sk-a***lmno', last_error: 'API key 无效或无权限' })

    const chatStore = useChatStore()
    await expect(chatStore.send('hello')).rejects.toThrow()

    expect(chatStore.error).toBe('API key 异常,详情见右上角状态')
  })

  it('refreshes llm status on finally', async () => {
    const apiMock = vi.mocked(clientModule.api)
    apiMock.mockResolvedValueOnce({ ai_reply: 'hi', turn_id: 1 })
    apiMock.mockResolvedValueOnce({ state: 'green', masked_key: 'sk-a***lmno', last_error: null })

    const chatStore = useChatStore()
    await chatStore.send('hello')

    expect(apiMock).toHaveBeenCalledWith('/api/llm/status')
  })
})
