import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '../api/client'
import type { ChatResponse, Message } from '../api/types'

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const sending = ref(false)
  const error = ref<string | null>(null)

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
      error.value = e instanceof Error ? e.message : String(e)
      messages.value.pop()
    } finally {
      sending.value = false
    }
  }

  return { messages, sending, error, load, send }
})
