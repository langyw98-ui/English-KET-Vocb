export interface ChatResponse {
  ai_reply: string
  turn_id: number
}

export interface Message {
  role: 'user' | 'ai' | 'system'
  content: string
  turn_id: number | null
  created_at: string
}

export interface ReportResponse {
  mastered_count: number
  learning_count: number
  struggling_count: number
  used_count: number
  unused_count: number
  total_words: number
}

export interface ReportWord {
  word: string
  context: string
  mastery_score: number
  exposed_count: number
  correct_count: number
  wrong_count: number
  status: string
}

export interface ReportCategoryResponse {
  category: string
  page: number
  page_size: number
  total: number
  total_pages: number
  words: ReportWord[]
}
