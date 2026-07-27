<template>
  <div id="app-container">
    <header class="navbar">
      <h1 class="title">KET 英语搭子</h1>
      <nav class="nav-links">
        <RouterLink to="/chat" class="nav-item">对话</RouterLink>
        <RouterLink to="/report" class="nav-item">报告</RouterLink>
      </nav>
    </header>
    <main class="content">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<style scoped>
#app-container {
  max-width: 800px;
  margin: 0 auto;
  font-family: sans-serif;
}
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-bottom: 1px solid #ccc;
}
.nav-links {
  display: flex;
  gap: 1rem;
}
.nav-item {
  text-decoration: none;
  color: #333;
  font-weight: bold;
}
.nav-item.router-link-active {
  color: #42b983;
}
.content {
  padding: 1rem;
}
</style>
