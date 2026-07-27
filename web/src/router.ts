import { createRouter, createWebHistory } from 'vue-router'
import { defineComponent, h } from 'vue'

const ChatPlaceholder = defineComponent({
  name: 'ChatViewPlaceholder',
  render() {
    return h('div', 'Chat View Placeholder')
  }
})

const ReportPlaceholder = defineComponent({
  name: 'ReportViewPlaceholder',
  render() {
    return h('div', 'Report View Placeholder')
  }
})

const routes = [
  { path: '/', redirect: '/chat' },
  { path: '/chat', component: ChatPlaceholder },
  { path: '/report', component: ReportPlaceholder },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})
