import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '../api/client'

interface LlmStatus {
  state: 'red' | 'green'
  masked_key: string | null
  last_error: string | null
}

export const useLlmKeyStore = defineStore('llmKey', () => {
  const state = ref<'red' | 'green'>('red')
  const maskedKey = ref<string | null>(null)
  const lastError = ref<string | null>(null)
  const loaded = ref(false)
  const popoverOpen = ref(false)

  async function loadStatus() {
    try {
      const res = await api<LlmStatus>('/api/llm/status')
      state.value = res.state
      maskedKey.value = res.masked_key
      lastError.value = res.last_error
    } catch (e) {
      console.warn('loadStatus failed, keeping current state', e)
    } finally {
      loaded.value = true
    }
  }

  function openPopover() { popoverOpen.value = true }
  function closePopover() { popoverOpen.value = false }

  return { state, maskedKey, lastError, loaded, popoverOpen, loadStatus, openPopover, closePopover }
})
