import { createRouter, createWebHistory } from 'vue-router'
import ChatView from './views/ChatView.vue'
import ReportView from './views/ReportView.vue'

const routes = [
  { path: '/', redirect: '/chat' },
  { path: '/chat', component: ChatView },
  { path: '/report', component: ReportView },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})
