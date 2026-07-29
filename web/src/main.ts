import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import { router } from './router'
import { useLlmKeyStore } from './stores/llmKey'

const app = createApp(App)
const pinia = createPinia()
app.use(pinia)
app.use(router)

app.mount('#app')

const llmKey = useLlmKeyStore(pinia)
llmKey.loadStatus()
