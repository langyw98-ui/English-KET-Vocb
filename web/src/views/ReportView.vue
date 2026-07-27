<template>
  <div class="report-container">
    <!-- Header -->
    <div class="report-header">
      <div>
        <h2 class="report-title">学习成果与词汇报告</h2>
        <p class="report-subtitle">实时统计词汇积累情况，点击下方卡片查看具体词汇</p>
      </div>
      <button class="refresh-btn" @click="reportStore.loadCounts()">
        <span>🔄 刷新数据</span>
      </button>
    </div>

    <!-- Total Vocabulary Banner -->
    <div class="total-banner">
      <div class="total-info">
        <span class="total-label">KET 核心词表总数</span>
        <span class="total-value">{{ reportStore.counts?.total_words ?? 0 }} <small>词</small></span>
      </div>
      <div class="total-decoration">📊</div>
    </div>

    <!-- Loading State -->
    <div v-if="!reportStore.counts" class="report-loading">
      <div class="spinner"></div>
      <p>正在加载报告数据...</p>
    </div>

    <!-- 5 Count Cards -->
    <div v-else class="cards-grid">
      <!-- Mastered Card -->
      <div
        class="stat-card mastered"
        @click="reportStore.openCategory('mastered')"
        role="button"
        tabindex="0"
      >
        <div class="card-top">
          <span class="card-icon">🏆</span>
          <span class="percentage-badge mastered">
            {{ getPercentage(reportStore.counts.mastered_count) }}
          </span>
        </div>
        <div class="card-count">{{ reportStore.counts.mastered_count }}</div>
        <div class="card-label">已掌握</div>
        <div class="card-hint">熟练运用的核心词汇 &rarr;</div>
      </div>

      <!-- Learning Card -->
      <div
        class="stat-card learning"
        @click="reportStore.openCategory('learning')"
        role="button"
        tabindex="0"
      >
        <div class="card-top">
          <span class="card-icon">🌱</span>
          <span class="percentage-badge learning">
            {{ getPercentage(reportStore.counts.learning_count) }}
          </span>
        </div>
        <div class="card-count">{{ reportStore.counts.learning_count }}</div>
        <div class="card-label">正在学</div>
        <div class="card-hint">逐步加深理解的词汇 &rarr;</div>
      </div>

      <!-- Struggling Card -->
      <div
        class="stat-card struggling"
        @click="reportStore.openCategory('struggling')"
        role="button"
        tabindex="0"
      >
        <div class="card-top">
          <span class="card-icon">💪</span>
          <span class="percentage-badge struggling">
            {{ getPercentage(reportStore.counts.struggling_count) }}
          </span>
        </div>
        <div class="card-count">{{ reportStore.counts.struggling_count }}</div>
        <div class="card-label">有困难</div>
        <div class="card-hint">易错或需重点巩固 &rarr;</div>
      </div>

      <!-- Used Card -->
      <div
        class="stat-card used"
        @click="reportStore.openCategory('used')"
        role="button"
        tabindex="0"
      >
        <div class="card-top">
          <span class="card-icon">👀</span>
          <span class="percentage-badge used">
            {{ getPercentage(reportStore.counts.used_count) }}
          </span>
        </div>
        <div class="card-count">{{ reportStore.counts.used_count }}</div>
        <div class="card-label">接触过</div>
        <div class="card-hint">对话中露过面的词汇 &rarr;</div>
      </div>

      <!-- Unused Card -->
      <div
        class="stat-card unused"
        @click="reportStore.openCategory('unused')"
        role="button"
        tabindex="0"
      >
        <div class="card-top">
          <span class="card-icon">📦</span>
          <span class="percentage-badge unused">
            {{ getPercentage(reportStore.counts.unused_count) }}
          </span>
        </div>
        <div class="card-count">{{ reportStore.counts.unused_count }}</div>
        <div class="card-label">未学</div>
        <div class="card-hint">等待解锁的新词汇 &rarr;</div>
      </div>
    </div>

    <!-- Word List Modal -->
    <WordListModal />
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useReportStore } from '../stores/report'
import WordListModal from '../components/WordListModal.vue'

const reportStore = useReportStore()

function getPercentage(count: number): string {
  const total = reportStore.counts?.total_words ?? 0
  if (total <= 0) return '0.0%'
  const pct = (count / total) * 100
  return `${pct.toFixed(1)}%`
}

onMounted(() => {
  reportStore.loadCounts()
})
</script>

<style scoped>
.report-container {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
}

.report-title {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 800;
  color: #0f172a;
}

.report-subtitle {
  margin: 0.25rem 0 0 0;
  font-size: 0.9rem;
  color: #64748b;
}

.refresh-btn {
  background: #ffffff;
  border: 1px solid #cbd5e1;
  padding: 0.5rem 1rem;
  border-radius: 10px;
  font-size: 0.875rem;
  font-weight: 600;
  color: #334155;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.refresh-btn:hover {
  background: #f8fafc;
  border-color: #94a3b8;
  color: #0f172a;
}

.total-banner {
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  color: #ffffff;
  border-radius: 16px;
  padding: 1.5rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 10px 20px -5px rgba(15, 23, 42, 0.25);
}

.total-label {
  display: block;
  font-size: 0.875rem;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 600;
}

.total-value {
  font-size: 2.25rem;
  font-weight: 800;
  color: #ffffff;
}

.total-value small {
  font-size: 1rem;
  font-weight: 500;
  color: #cbd5e1;
}

.total-decoration {
  font-size: 3rem;
  opacity: 0.8;
}

.report-loading {
  padding: 4rem;
  text-align: center;
  color: #64748b;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}

.spinner {
  width: 36px;
  height: 36px;
  border: 3px solid #e2e8f0;
  border-top-color: #2563eb;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1.25rem;
}

.stat-card {
  background: #ffffff;
  border-radius: 16px;
  padding: 1.25rem;
  border: 1px solid #e2e8f0;
  cursor: pointer;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
}

.stat-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.08);
}

.card-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-icon {
  font-size: 1.6rem;
}

.percentage-badge {
  font-size: 0.75rem;
  font-weight: 700;
  padding: 0.2rem 0.55rem;
  border-radius: 9999px;
}

.percentage-badge.mastered { background: #dcfce7; color: #15803d; }
.percentage-badge.learning { background: #dbeafe; color: #1d4ed8; }
.percentage-badge.struggling { background: #fee2e2; color: #b91c1c; }
.percentage-badge.used { background: #fef3c7; color: #b45309; }
.percentage-badge.unused { background: #f1f5f9; color: #64748b; }

.card-count {
  font-size: 2rem;
  font-weight: 800;
  color: #0f172a;
}

.card-label {
  font-size: 1rem;
  font-weight: 700;
  color: #334155;
}

.card-hint {
  font-size: 0.78rem;
  color: #94a3b8;
  margin-top: 0.25rem;
  transition: color 0.2s;
}

.stat-card:hover .card-hint {
  color: #2563eb;
}

.stat-card.mastered { border-top: 4px solid #22c55e; }
.stat-card.learning { border-top: 4px solid #3b82f6; }
.stat-card.struggling { border-top: 4px solid #ef4444; }
.stat-card.used { border-top: 4px solid #f59e0b; }
.stat-card.unused { border-top: 4px solid #94a3b8; }

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
