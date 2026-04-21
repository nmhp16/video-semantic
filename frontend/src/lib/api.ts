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
  mode: 'text' | 'visual' | 'action' | 'action_chain'
  hits: UnifiedSearchHit[]
  info: Record<string, unknown>
}

export interface IngestResponse {
  success: boolean
  message: string
  video_id: string
  status: 'completed' | 'already_exists'
}

export type SearchMode = 'text' | 'visual' | 'action' | 'action_chain'
export type SearchScope = 'video' | 'global'

export interface UnifiedSearchRequest {
  video_url?: string
  video_id?: string
  query?: string
  mode: SearchMode
  k?: number
  filter_objects?: string
  steps?: string[]
  max_gap?: number
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

  query(req: UnifiedSearchRequest): Promise<UnifiedSearchResponse> {
    return request('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
  },

  frameUrl(framePath: string): string {
    // frames stored as e.g. "data/frames/video_id/frame_0042.jpg"
    // strip leading "data/frames/" or "frames/" prefix
    const match = framePath.match(/(?:data\/)?frames\/(.+)/)
    const rel = match ? match[1] : framePath
    return `${BASE_URL}/frames/${rel}`
  },

  mediaUrl(videoId: string): string {
    return `${BASE_URL}/media/${videoId}.mp4`
  },
}
