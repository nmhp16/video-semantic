const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

export interface VideoMeta {
  video_id: string
  has_text_search: boolean
  has_visual_search: boolean
  has_action_search: boolean
}

export interface UnifiedSearchHit {
  start: number
  end: number
  score: number
  text?: string
  frame?: string
  objects?: string[]
  caption?: string
  video_id: string
}

export interface UnifiedSearchResponse {
  video_id: string | null
  mode: 'text' | 'visual' | 'action'
  hits: UnifiedSearchHit[]
  info: Record<string, unknown>
}

export interface IngestResponse {
  success: boolean
  video_id: string
  status: 'completed' | 'processing' | 'failed' | 'unknown'
  error?: string
}

export type SearchMode = 'text' | 'visual' | 'action'
export type SearchScope = 'video' | 'global'

export interface UnifiedSearchRequest {
  video_url?: string
  video_id?: string
  query?: string
  mode: SearchMode
  k?: number
  ingest_if_needed?: boolean
  scope?: SearchScope
  videos?: string[]
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, init)
  if (!res.ok) {
    const body = await res.text()
    throw new Error(body || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  baseUrl: BASE_URL,

  getVideos(): Promise<{ videos: VideoMeta[] }> {
    return request('/videos')
  },

  ingest(video_url: string, video_id?: string): Promise<IngestResponse> {
    return request('/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_url, video_id }),
    })
  },

  getIngestStatus(video_id: string): Promise<IngestResponse> {
    return request(`/ingest/status/${encodeURIComponent(video_id)}`)
  },

  query(req: UnifiedSearchRequest): Promise<UnifiedSearchResponse> {
    return request('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
  },

  frameUrl(framePath: string): string {
    if (framePath.startsWith('http')) return framePath   // Supabase public URL
    const match = framePath.match(/(?:data\/)?frames\/(.+)/)
    const rel = match ? match[1] : framePath
    return `${BASE_URL}/frames/${rel}`
  },

}
