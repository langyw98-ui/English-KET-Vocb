<template>
  <div class="llm-status-badge" ref="badgeRef" @click.stop="togglePopover">
    <span class="dot" :class="llmKeyStore.state"></span>
    <span class="label">{{ llmKeyStore.state === 'green' ? 'LLM 可用' : 'LLM 不可用' }}</span>

    <div v-if="llmKeyStore.popoverOpen" class="popover">
      <div class="popover-row">
        <span class="dot" :class="llmKeyStore.state"></span>
        <span>{{ llmKeyStore.state === 'green' ? 'LLM 可用' : 'LLM 不可用' }}</span>
      </div>
      <div class="popover-row">
        <span class="row-label">当前 key:</span>
        <code>{{ llmKeyStore.maskedKey ?? '未配置' }}</code>
      </div>
      <div v-if="llmKeyStore.lastError" class="popover-row error-row">
        <span class="row-label">错误原因:</span>
        <span>{{ llmKeyStore.lastError }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref } from 'vue'
import { useLlmKeyStore } from '../stores/llmKey'

const llmKeyStore = useLlmKeyStore()
const badgeRef = ref<HTMLElement | null>(null)

function togglePopover() {
  llmKeyStore.popoverOpen ? llmKeyStore.closePopover() : llmKeyStore.openPopover()
}

function handleDocumentClick(e: MouseEvent) {
  if (llmKeyStore.popoverOpen && badgeRef.value && !badgeRef.value.contains(e.target as Node)) {
    llmKeyStore.closePopover()
  }
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape' && llmKeyStore.popoverOpen) {
    llmKeyStore.closePopover()
  }
}

onMounted(() => {
  document.addEventListener('click', handleDocumentClick)
  document.addEventListener('keydown', handleKeydown)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', handleDocumentClick)
  document.removeEventListener('keydown', handleKeydown)
})
</script>

<style scoped>
.llm-status-badge {
  position: relative;
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem 0.75rem;
  border-radius: 9px;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 600;
  transition: background 0.2s;
}
.llm-status-badge:hover {
  background: #f1f5f9;
}
.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
}
.dot.green { background: #10b981; box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2); }
.dot.red { background: #ef4444; box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.2); }
.label { color: #475569; }

.popover {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  min-width: 240px;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 0.85rem 1rem;
  box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
  z-index: 200;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.popover-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
}
.popover-row code {
  background: #f1f5f9;
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  font-family: ui-monospace, 'SF Mono', Menlo, monospace;
  font-size: 0.8rem;
}
.row-label {
  color: #64748b;
  min-width: 70px;
}
.error-row {
  color: #991b1b;
}
</style>
