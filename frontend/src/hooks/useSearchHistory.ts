// frontend/src/hooks/useSearchHistory.ts
import { useState, useCallback } from 'react'
import type { SearchMode } from '@/lib/api'

const STORAGE_KEY = 'vsearch:history'
const MAX_HISTORY = 10

export interface HistoryEntry {
  query: string
  mode: SearchMode
  timestamp: number
}

function load(): HistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
  } catch {
    return []
  }
}

function save(entries: HistoryEntry[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
}

export function useSearchHistory() {
  const [history, setHistory] = useState<HistoryEntry[]>(load)

  const push = useCallback((query: string, mode: SearchMode) => {
    if (!query.trim()) return
    setHistory((prev) => {
      const filtered = prev.filter((e) => !(e.query === query && e.mode === mode))
      const next = [{ query, mode, timestamp: Date.now() }, ...filtered].slice(0, MAX_HISTORY)
      save(next)
      return next
    })
  }, [])

  const remove = useCallback((query: string, mode: SearchMode) => {
    setHistory((prev) => {
      const next = prev.filter((e) => !(e.query === query && e.mode === mode))
      save(next)
      return next
    })
  }, [])

  return { history, push, remove }
}
