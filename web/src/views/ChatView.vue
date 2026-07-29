<template>
  <div class="chat-container">
    <!-- Header info banner -->
    <div class="chat-header">
      <div class="chat-title-group">
        <span class="avatar-icon">🤖</span>
        <div>
          <h2 class="chat-title">KET 英语学习搭子</h2>
          <p class="chat-subtitle">和我用英语或中文对话，一起积累词汇吧！</p>
        </div>
      </div>
    </div>

    <!-- Messages Scroll Area -->
    <div ref="messagesContainer" class="messages-list">
      <div v-if="chatStore.messages.length === 0 && !chatStore.sending" class="chat-welcome">
        <div class="welcome-card">
          <span class="welcome-emoji">👋</span>
          <h3>你好！我是你的 KET 英语搭子</h3>
          <p>试着跟我说几句话，例如：</p>
          <div class="sample-chips">
            <button class="chip" @click="sendSample('The cat slipped on the ice.')">
              The cat slipped on the ice.
            </button>
            <button class="chip" @click="sendSample('我想学习关于小动物的单词')">
              我想学习关于小动物的单词
            </button>
          </div>
        </div>
      </div>

      <div
        v-for="(msg, idx) in chatStore.messages"
        :key="idx"
        class="message-row"
        :class="msg.role"
      >
        <div class="message-avatar">
          <span v-if="msg.role === 'user'">🧒</span>
          <span v-else-if="msg.role === 'ai'">🤖</span>
          <span v-else>⚙️</span>
        </div>

        <div class="message-bubble-wrapper">
          <div class="message-bubble" :class="msg.role">
            <div class="message-content">{{ msg.content }}</div>
          </div>
          <div class="message-time">{{ formatTime(msg.created_at) }}</div>
        </div>
      </div>

      <!-- Pending / Sending Spinner Bubble -->
      <div v-if="chatStore.sending" class="message-row ai sending">
        <div class="message-avatar">🤖</div>
        <div class="message-bubble-wrapper">
          <div class="message-bubble ai typing">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
          </div>
        </div>
      </div>
    </div>

    <!-- Error Banner -->
    <div v-if="chatStore.error" class="error-banner">
      <span class="error-icon">⚠️</span>
      <span class="error-text">{{ chatStore.error }}</span>
      <button class="retry-btn" @click="handleRetry">重试</button>
    </div>

    <!-- Input Form Area -->
    <form class="input-area" @submit.prevent="handleSend">
      <input
        v-model="inputText"
        type="text"
        class="chat-input"
        placeholder="输入英语句子或对话内容..."
        :disabled="chatStore.sending"
        ref="inputRef"
      />
      <button
        type="submit"
        class="send-btn"
        :disabled="!inputText.trim() || chatStore.sending"
      >
        <span v-if="!chatStore.sending">发送 &rarr;</span>
        <span v-else class="btn-spinner"></span>
      </button>
    </form>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick, watch } from 'vue'
import { useChatStore } from '../stores/chat'

const chatStore = useChatStore()
const inputText = ref('')
const lastSentText = ref('')
const messagesContainer = ref<HTMLElement | null>(null)
const inputRef = ref<HTMLInputElement | null>(null)

function focusInput() {
  nextTick(() => {
    inputRef.value?.focus()
  })
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

function formatTime(iso: string): string {
  if (!iso) return ''
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}

async function handleSend() {
  const text = inputText.value.trim()
  if (!text || chatStore.sending) return

  lastSentText.value = text
  inputText.value = ''
  try {
    await chatStore.send(text)
  } finally {
    scrollToBottom()
    focusInput()
  }
}

async function sendSample(text: string) {
  inputText.value = text
  await handleSend()
}

async function handleRetry() {
  if (lastSentText.value && !chatStore.sending) {
    try {
      await chatStore.send(lastSentText.value)
    } finally {
      scrollToBottom()
      focusInput()
    }
  } else {
    chatStore.load()
    focusInput()
  }
}

watch(
  () => chatStore.sending,
  (sending) => {
    if (!sending) {
      focusInput()
    }
  }
)

watch(
  () => chatStore.messages.length,
  () => {
    scrollToBottom()
  }
)

onMounted(async () => {
  await chatStore.load()
  scrollToBottom()
  focusInput()
})
</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 120px);
  max-height: 780px;
  background: #ffffff;
  border-radius: 20px;
  box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
  border: 1px solid #e2e8f0;
  overflow: hidden;
}

.chat-header {
  padding: 1rem 1.5rem;
  background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
  border-bottom: 1px solid #e2e8f0;
}

.chat-title-group {
  display: flex;
  align-items: center;
  gap: 0.85rem;
}

.avatar-icon {
  font-size: 2rem;
  background: #ffffff;
  padding: 0.35rem 0.55rem;
  border-radius: 14px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

.chat-title {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 700;
  color: #0f172a;
}

.chat-subtitle {
  margin: 0.15rem 0 0 0;
  font-size: 0.85rem;
  color: #64748b;
}

.messages-list {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
  background: #fafafa;
}

.chat-welcome {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
}

.welcome-card {
  text-align: center;
  background: #ffffff;
  padding: 2rem;
  border-radius: 16px;
  border: 1px dashed #cbd5e1;
  max-width: 400px;
}

.welcome-emoji {
  font-size: 2.5rem;
}

.welcome-card h3 {
  margin: 0.75rem 0 0.5rem 0;
  color: #1e293b;
  font-size: 1.1rem;
}

.welcome-card p {
  color: #64748b;
  font-size: 0.875rem;
  margin-bottom: 1rem;
}

.sample-chips {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.chip {
  background: #f1f5f9;
  border: 1px solid #e2e8f0;
  padding: 0.5rem 0.85rem;
  border-radius: 8px;
  color: #2563eb;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
}

.chip:hover {
  background: #dbeafe;
  border-color: #93c5fd;
}

.message-row {
  display: flex;
  gap: 0.75rem;
  max-width: 82%;
}

.message-row.user {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.message-row.ai {
  align-self: flex-start;
}

.message-avatar {
  font-size: 1.4rem;
  width: 38px;
  height: 38px;
  border-radius: 50%;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 4px rgba(0,0,0,0.04);
  flex-shrink: 0;
}

.message-bubble-wrapper {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.message-row.user .message-bubble-wrapper {
  align-items: flex-end;
}

.message-bubble {
  padding: 0.85rem 1.15rem;
  border-radius: 16px;
  font-size: 0.95rem;
  line-height: 1.5;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
}

.message-bubble.user {
  background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
  color: #ffffff;
  border-bottom-right-radius: 4px;
}

.message-bubble.ai {
  background: #ffffff;
  color: #1e293b;
  border: 1px solid #e2e8f0;
  border-bottom-left-radius: 4px;
}

.message-content {
  white-space: pre-wrap;
  word-break: break-word;
}

.message-time {
  font-size: 0.72rem;
  color: #94a3b8;
  padding: 0 0.2rem;
}

.message-bubble.typing {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.85rem 1.25rem;
}

.dot {
  width: 7px;
  height: 7px;
  background: #94a3b8;
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out both;
}

.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

.error-banner {
  background: #fef2f2;
  border-top: 1px solid #fca5a5;
  padding: 0.65rem 1.25rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: #991b1b;
}

.error-text {
  flex: 1;
}

.retry-btn {
  background: #dc2626;
  color: #ffffff;
  border: none;
  padding: 0.25rem 0.75rem;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
}

.retry-btn:hover {
  background: #b91c1c;
}

.input-area {
  padding: 1rem 1.25rem;
  background: #ffffff;
  border-top: 1px solid #e2e8f0;
  display: flex;
  gap: 0.75rem;
}

.chat-input {
  flex: 1;
  padding: 0.75rem 1rem;
  border: 1px solid #cbd5e1;
  border-radius: 12px;
  font-size: 0.95rem;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.chat-input:focus {
  border-color: #2563eb;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
}

.chat-input:disabled {
  background: #f8fafc;
  cursor: not-allowed;
}

.send-btn {
  background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
  color: #ffffff;
  border: none;
  padding: 0.75rem 1.4rem;
  border-radius: 12px;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 90px;
}

.send-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
}

.send-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid #ffffff;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
