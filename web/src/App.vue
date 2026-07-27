<template>
  <div id="app-wrapper">
    <!-- Navbar Header -->
    <header class="app-header">
      <div class="header-inner">
        <div class="brand">
          <div class="brand-logo">💬</div>
          <div class="brand-text">
            <h1 class="brand-title">KET 英语搭子</h1>
            <span class="brand-tag">Smart Partner</span>
          </div>
        </div>

        <nav class="nav-tabs" aria-label="Main Navigation">
          <RouterLink to="/chat" class="nav-tab">
            <span class="tab-icon">💬</span>
            <span class="tab-label">对话</span>
          </RouterLink>
          <RouterLink to="/report" class="nav-tab">
            <span class="tab-icon">📊</span>
            <span class="tab-label">报告</span>
          </RouterLink>
        </nav>
      </div>
    </header>

    <!-- Main Container -->
    <main class="app-main">
      <div class="main-content">
        <RouterView />
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<style>
/* Global CSS reset & base styles */
*, *::before, *::after {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
  background-color: #f1f5f9;
  color: #0f172a;
  -webkit-font-smoothing: antialiased;
}

#app-wrapper {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid #e2e8f0;
  box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.03);
}

.header-inner {
  max-width: 960px;
  margin: 0 auto;
  padding: 0.75rem 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.brand-logo {
  font-size: 1.5rem;
  background: linear-gradient(135deg, #2563eb, #3b82f6);
  width: 40px;
  height: 40px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25);
}

.brand-title {
  margin: 0;
  font-size: 1.2rem;
  font-weight: 800;
  color: #0f172a;
  letter-spacing: -0.02em;
}

.brand-tag {
  font-size: 0.7rem;
  color: #64748b;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.nav-tabs {
  display: flex;
  background: #f1f5f9;
  padding: 0.25rem;
  border-radius: 12px;
  gap: 0.25rem;
}

.nav-tab {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.45rem 1.1rem;
  border-radius: 9px;
  text-decoration: none;
  font-size: 0.9rem;
  font-weight: 600;
  color: #64748b;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

.nav-tab:hover {
  color: #0f172a;
}

.nav-tab.router-link-active {
  background: #ffffff;
  color: #2563eb;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.tab-icon {
  font-size: 1rem;
}

.app-main {
  flex: 1;
  padding: 1.5rem;
  max-width: 960px;
  margin: 0 auto;
  width: 100%;
}

.main-content {
  width: 100%;
}
</style>
