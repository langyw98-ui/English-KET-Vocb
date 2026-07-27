import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '../api/client'
import type { ReportCategoryResponse, ReportResponse, ReportWord } from '../api/types'

export const useReportStore = defineStore('report', () => {
  const counts = ref<ReportResponse | null>(null)
  const activeCategory = ref<string | null>(null)
  const words = ref<ReportWord[]>([])
  const page = ref(1)
  const pageSize = ref(100)
  const total = ref(0)
  const totalPages = ref(1)
  const loadingPage = ref(false)

  async function loadCounts() {
    counts.value = await api<ReportResponse>('/api/report')
  }

  async function openCategory(cat: string) {
    activeCategory.value = cat
    await loadPage(1)
  }

  async function loadPage(p: number) {
    if (!activeCategory.value) return
    loadingPage.value = true
    try {
      const res = await api<ReportCategoryResponse>(
        `/api/report/${activeCategory.value}?page=${p}&page_size=${pageSize.value}`
      )
      words.value = res.words
      page.value = res.page
      total.value = res.total
      totalPages.value = res.total_pages
    } finally {
      loadingPage.value = false
    }
  }

  function closeCategory() {
    activeCategory.value = null
    words.value = []
    page.value = 1
    total.value = 0
    totalPages.value = 1
  }

  async function nextPage() {
    if (page.value < totalPages.value) await loadPage(page.value + 1)
  }

  async function prevPage() {
    if (page.value > 1) await loadPage(page.value - 1)
  }

  return {
    counts,
    activeCategory,
    words,
    page,
    pageSize,
    total,
    totalPages,
    loadingPage,
    loadCounts,
    openCategory,
    loadPage,
    closeCategory,
    nextPage,
    prevPage,
  }
})
