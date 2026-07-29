import { mount } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import { describe, beforeEach, it, expect } from 'vitest'
import LlmStatusBadge from '../LlmStatusBadge.vue'
import { useLlmKeyStore } from '../../stores/llmKey'

describe('LlmStatusBadge.vue', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('renders green state correctly', () => {
    const store = useLlmKeyStore()
    store.state = 'green'
    const wrapper = mount(LlmStatusBadge)
    expect(wrapper.find('.dot.green').exists()).toBe(true)
    expect(wrapper.text()).toContain('LLM 可用')
  })

  it('renders red state and opens popover on click', async () => {
    const store = useLlmKeyStore()
    store.state = 'red'
    store.maskedKey = 'sk-a***lmno'
    store.lastError = 'API key 无效'

    const wrapper = mount(LlmStatusBadge)
    expect(wrapper.find('.dot.red').exists()).toBe(true)
    expect(wrapper.find('.popover').exists()).toBe(false)

    await wrapper.find('.llm-status-badge').trigger('click')
    expect(store.popoverOpen).toBe(true)
    expect(wrapper.find('.popover').exists()).toBe(true)
    expect(wrapper.find('.popover').text()).toContain('API key 无效')
  })
})
