import { isYouTubeId } from './utils'

const _titleCache: Record<string, string> = {}

export function ytThumbnail(videoId: string): string {
  return `https://img.youtube.com/vi/${videoId}/hqdefault.jpg`
}

export async function fetchVideoTitle(videoId: string): Promise<string> {
  if (_titleCache[videoId]) return _titleCache[videoId]
  if (!isYouTubeId(videoId)) return videoId
  try {
    const res = await fetch(
      `https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`
    )
    if (res.ok) {
      const data = await res.json()
      _titleCache[videoId] = data.title as string
      return data.title as string
    }
  } catch {}
  _titleCache[videoId] = videoId
  return videoId
}

export function getCachedTitle(videoId: string): string {
  return _titleCache[videoId] ?? videoId
}
