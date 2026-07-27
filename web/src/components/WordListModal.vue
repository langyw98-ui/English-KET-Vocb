<template>
  <Teleport to="body">
    <Transition name="modal-fade">
      <div v-if="reportStore.activeCategory" class="modal-backdrop" @click.self="handleClose">
        <div class="modal-container" role="dialog" aria-modal="true">
          <!-- Header -->
          <header class="modal-header">
            <div class="header-title-group">
              <span class="category-badge" :class="reportStore.activeCategory">
                {{ categoryName }}
              </span>
              <h2 class="modal-title">词汇列表</h2>
              <span class="count-pill">共 {{ reportStore.total }} 词</span>
            </div>
            <button class="close-btn" @click="handleClose" aria-label="关闭">
              &times;
            </button>
          </header>

          <!-- Loading State -->
          <div v-if="reportStore.loadingPage" class="modal-loading">
            <div class="spinner"></div>
            <span>加载词汇中...</span>
          </div>

          <!-- Empty State -->
          <div v-else-if="reportStore.words.length === 0" class="modal-empty">
            <div class="empty-icon">📖</div>
            <p>该分类下暂无词汇</p>
          </div>

          <!-- Word List Grid -->
          <div v-else class="modal-body">
            <div class="word-grid">
              <div
                v-for="item in reportStore.words"
                :key="item.word + '-' + item.context"
                class="word-card"
                :class="reportStore.activeCategory"
              >
                <div class="word-header">
                  <span class="word-text">{{ item.word }}</span>
                  <span class="status-badge" :class="item.status">{{ getStatusLabel(item.status) }}</span>
                </div>

                <div v-if="item.context" class="word-context">
                  <span class="context-label">上下文:</span> {{ item.context }}
                </div>

                <!-- Stats for exposed words -->
                <div v-if="reportStore.activeCategory !== 'unused' && item.exposed_count > 0" class="word-stats">
                  <div class="stat-item mastery">
                    <span class="stat-label">掌握度</span>
                    <span class="stat-value">{{ item.mastery_score }}</span>
                  </div>
                  <div class="stat-item exposed">
                    <span class="stat-label">出现</span>
                    <span class="stat-value">{{ item.exposed_count }}次</span>
                  </div>
                  <div class="stat-item correct">
                    <span class="stat-label">正确</span>
                    <span class="stat-value">{{ item.correct_count }}</span>
                  </div>
                  <div class="stat-item wrong">
                    <span class="stat-label">错误</span>
                    <span class="stat-value">{{ item.wrong_count }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Pagination Footer -->
          <footer v-if="reportStore.totalPages > 1" class="modal-footer">
            <button
              class="page-btn"
              :disabled="reportStore.page <= 1 || reportStore.loadingPage"
              @click="reportStore.prevPage()"
            >
              &larr; 上一页
            </button>

            <span class="page-info">
              第 {{ reportStore.page }} / {{ reportStore.totalPages }} 页
            </span>

            <button
              class="page-btn"
              :disabled="reportStore.page >= reportStore.totalPages || reportStore.loadingPage"
              @click="reportStore.nextPage()"
            >
              下一页 &rarr;
            </button>
          </footer>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import { useReportStore } from '../stores/report'

const reportStore = useReportStore()

const categoryNames: Record<string, string> = {
  mastered: '已掌握',
  learning: '正在学',
  struggling: '有困难',
  used: '接触过',
  unused: '未学',
}

const statusLabels: Record<string, string> = {
  mastered: '已掌握',
  learning: '正在学',
  struggling: '有困难',
  new: '未接触',
}

const categoryName = computed(() => {
  return reportStore.activeCategory
    ? categoryNames[reportStore.activeCategory] || reportStore.activeCategory
    : ''
})

function getStatusLabel(status: string): string {
  return statusLabels[status] || status
}

function handleClose() {
  reportStore.closeCategory()
}

function handleKeyDown(e: KeyboardEvent) {
  if (e.key === 'Escape' && reportStore.activeCategory) {
    handleClose()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeyDown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyDown)
})
</script>

<style scoped>
.modal-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(15, 23, 42, 0.65);
  backdrop-filter: blur(8px);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
}

.modal-container {
  background: #ffffff;
  width: 100%;
  max-width: 680px;
  max-height: 85vh;
  border-radius: 20px;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  animation: modalSlideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.modal-header {
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: linear-gradient(to right, #f8fafc, #f1f5f9);
}

.header-title-group {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.modal-title {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 700;
  color: #0f172a;
}

.category-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  font-size: 0.8125rem;
  font-weight: 600;
}

.category-badge.mastered {
  background: #dcfce7;
  color: #15803d;
}
.category-badge.learning {
  background: #dbeafe;
  color: #1d4ed8;
}
.category-badge.struggling {
  background: #fee2e2;
  color: #b91c1c;
}
.category-badge.used {
  background: #fef3c7;
  color: #b45309;
}
.category-badge.unused {
  background: #f1f5f9;
  color: #64748b;
}

.count-pill {
  font-size: 0.8125rem;
  color: #64748b;
  background: #ffffff;
  padding: 0.2rem 0.6rem;
  border-radius: 6px;
  border: 1px solid #cbd5e1;
}

.close-btn {
  background: transparent;
  border: none;
  font-size: 1.5rem;
  line-height: 1;
  color: #94a3b8;
  cursor: pointer;
  padding: 0.25rem 0.5rem;
  border-radius: 8px;
  transition: all 0.2s;
}

.close-btn:hover {
  background: #e2e8f0;
  color: #0f172a;
}

.modal-loading,
.modal-empty {
  padding: 3rem 1.5rem;
  text-align: center;
  color: #64748b;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
}

.empty-icon {
  font-size: 2.5rem;
}

.spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #e2e8f0;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.modal-body {
  padding: 1.25rem 1.5rem;
  overflow-y: auto;
  flex: 1;
}

.word-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1rem;
}

.word-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1rem;
  transition: transform 0.2s, box-shadow 0.2s;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.word-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
}

.word-card.mastered { border-left: 4px solid #22c55e; }
.word-card.learning { border-left: 4px solid #3b82f6; }
.word-card.struggling { border-left: 4px solid #ef4444; }
.word-card.used { border-left: 4px solid #f59e0b; }
.word-card.unused { border-left: 4px solid #94a3b8; }

.word-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.word-text {
  font-size: 1.125rem;
  font-weight: 700;
  color: #0f172a;
}

.status-badge {
  font-size: 0.75rem;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-weight: 600;
}
.status-badge.mastered { background: #dcfce7; color: #166534; }
.status-badge.learning { background: #dbeafe; color: #1e40af; }
.status-badge.struggling { background: #fee2e2; color: #991b1b; }
.status-badge.new { background: #f1f5f9; color: #475569; }

.word-context {
  font-size: 0.84rem;
  color: #475569;
  background: #f8fafc;
  padding: 0.4rem 0.6rem;
  border-radius: 6px;
  line-height: 1.4;
}

.context-label {
  font-weight: 600;
  color: #64748b;
}

.word-stats {
  display: flex;
  gap: 0.75rem;
  margin-top: 0.25rem;
  padding-top: 0.5rem;
  border-top: 1px dashed #e2e8f0;
  font-size: 0.75rem;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
  background: #f8fafc;
  padding: 0.25rem;
  border-radius: 6px;
}

.stat-label {
  color: #94a3b8;
  font-size: 0.7rem;
}

.stat-value {
  font-weight: 700;
  color: #1e293b;
}

.modal-footer {
  padding: 1rem 1.5rem;
  border-top: 1px solid #e2e8f0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #f8fafc;
}

.page-btn {
  padding: 0.5rem 1rem;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  background: #ffffff;
  color: #1e293b;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.page-btn:hover:not(:disabled) {
  background: #3b82f6;
  color: #ffffff;
  border-color: #3b82f6;
}

.page-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-info {
  font-size: 0.875rem;
  color: #64748b;
  font-weight: 500;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@keyframes modalSlideUp {
  from {
    opacity: 0;
    transform: translateY(20px) scale(0.98);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.25s ease;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}
</style>
