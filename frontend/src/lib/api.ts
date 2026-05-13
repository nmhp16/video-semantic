// frontend/src/lib/api.ts
const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

export interface VideoMeta {
  video_id: string
  title: string | null
  source_url: string | null
  has_text_search: boolean
  has_visual_search: boolean
  has_action_search: boolean
  has_xclip_action: boolean
  thumbnail_url?: string | null
  top_objects: string[]
}

export interface ScoreRange {
  min: number
  max: number
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
  score_range: ScoreRange | null
}

export interface IngestJobResponse {
  job_id: string | null
  video_id: string
  status: 'queued' | 'already_exists'
  message?: string
}

export interface JobStatusResponse {
  job_id: string
  video_id: string
  status: 'queued' | 'running' | 'done' | 'error'
  stage: string
  error: string | null
}

export interface IngestResponse {
  success: boolean
  message: string
  video_id: string
  status: 'completed' | 'already_exists'
}

export type SearchMode = 'text' | 'visual' | 'action'
export type SearchScope = 'video' | 'global'

export interface UnifiedSearchRequest {
  video_url?: string
  video_id?: string
  query?: string
  mode: SearchMode
  k?: number
  filter_objects?: string
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

  ingest(video_url: string, video_id?: string): Promise<IngestJobResponse> {
    return request('/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_url, video_id }),
    })
  },

  ingestStatus(jobId: string): Promise<JobStatusResponse> {
    return request(`/ingest/status/${encodeURIComponent(jobId)}`)
  },

  buildContexts(videoIds?: string[]): Promise<{ success: boolean; message: string }> {
    return request('/build_contexts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(videoIds ?? null),
    })
  },

  query(req: UnifiedSearchRequest): Promise<UnifiedSearchResponse> {
    return request('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
  },

  deleteVideo(videoId: string): Promise<{ success: boolean; video_id: string }> {
    return request(`/videos/${encodeURIComponent(videoId)}`, { method: 'DELETE' })
  },

  assetUrl(path: string): string {
    return path.startsWith('http') ? path : `${BASE_URL}${path}`
  },

  frameUrl(framePath: string): string {
    const match = framePath.match(/(?:data\/)?frames\/(.+)/)
    const rel = match ? match[1] : framePath
    return `${BASE_URL}/frames/${rel}`
  },

  mediaUrl(videoId: string): string {
    return `${BASE_URL}/media/${videoId}.mp4`
  },
}
