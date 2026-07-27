const BASE = ''

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    let detail = text
    try {
      const json = JSON.parse(text)
      detail = json.detail || text
    } catch {}
    throw new Error(`API ${res.status}: ${detail}`)
  }
  return res.json()
}
