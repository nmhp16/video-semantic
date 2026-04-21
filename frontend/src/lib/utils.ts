import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function formatTimeRange(start: number, end: number): string {
  return `${formatTime(start)} – ${formatTime(end)}`
}

export function scorePercent(score: number): number {
  // FAISS inner-product scores for normalized embeddings are in [-1, 1]
  // Convert to 0–100 percentage
  return Math.round(Math.max(0, Math.min(1, (score + 1) / 2)) * 100)
}

export function isYouTubeId(videoId: string): boolean {
  return /^[a-zA-Z0-9_-]{11}$/.test(videoId)
}

export function youtubeUrl(videoId: string, startSeconds?: number): string {
  const t = startSeconds !== undefined ? `&t=${Math.floor(startSeconds)}` : ''
  return `https://www.youtube.com/watch?v=${videoId}${t}`
}

export function truncate(str: string, maxLen: number): string {
  if (str.length <= maxLen) return str
  return str.slice(0, maxLen).trimEnd() + '…'
}
