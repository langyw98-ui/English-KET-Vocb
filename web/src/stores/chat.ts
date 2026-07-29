import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api, ApiError } from '../api/client'
import type { ChatResponse, Message } from '../api/types'
import { useLlmKeyStore } from './llmKey'

function mapChatError(e: unknown): string {
  if (e instanceof ApiError) {
    switch (e.status) {
      case 503:
        return 'LLM 未配置,请联系管理员'
      case 401:
        return 'API key 异常,详情见右上角状态'
      case 504:
        return '请求超时,请重新发送'
      case 502:
        return '网络异常,请稍后重新发送'
      case 429:
        return '请求过于频繁,请稍后再试'
    }
  }
  return e instanceof Error ? e.message : String(e)
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const sending = ref(false)
  const error = ref<string | null>(null)
  const llmKeyStore = useLlmKeyStore()

  async function load() {
    messages.value = await api<Message[]>('/api/messages?limit=15')
  }

  async function send(text: string) {
    sending.value = true
    error.value = null
    const optimistic: Message = {
      role: 'user',
      content: text,
      turn_id: null,
      created_at: new Date().toISOString()
    }
    messages.value.push(optimistic)
    try {
      const res = await api<ChatResponse>('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ text })
      })
      optimistic.turn_id = res.turn_id
      messages.value.push({
        role: 'ai',
        content: res.ai_reply,
        turn_id: res.turn_id,
        created_at: new Date().toISOString()
      })
    } catch (e: unknown) {
      error.value = mapChatError(e)
      messages.value.pop()
      throw e
    } finally {
      sending.value = false
      await llmKeyStore.loadStatus().catch(e => console.warn('refresh llm status failed', e))
    }
  }

  return { messages, sending, error, load, send }
})
