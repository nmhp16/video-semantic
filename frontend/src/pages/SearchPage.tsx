import { useState, useCallback, useEffect, useRef } from 'react'
import { Search, Plus } from 'lucide-react'
import { ModeSelector } from '@/components/ModeSelector'
import { FilterPanel } from '@/components/FilterPanel'
import { ResultCard } from '@/components/ResultCard'
import { IngestModal } from '@/components/IngestModal'
import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import { api } from '@/lib/api'
import type { SearchMode, SearchScope, UnifiedSearchHit, VideoMeta } from '@/lib/api'

const MODE_PLACEHOLDER: Record<SearchMode, string> = {
  text:   'e.g. "how does photosynthesis work" — finds the moment it was explained',
  visual: 'e.g. "person holding a whiteboard marker" — finds that frame',
  action: 'e.g. "someone opens a door and walks in" — finds that activity',
}

const MODE_EMPTY_HINT: Record<SearchMode, string> = {
  text:   'Searches the spoken transcript using semantic meaning — finds the moment even if the exact words differ.',
  visual: 'Searches every frame by visual description — finds scenes, objects, and settings.',
  action: 'Searches sliding windows of frames — finds activities and events as they unfold.',
}

export function SearchPage() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('text')
  const [scope, setScope] = useState<SearchScope>('global')
  const [selectedVideo, setSelectedVideo] = useState('')
  const [videos, setVideos] = useState<VideoMeta[]>([])
  const [hits, setHits] = useState<UnifiedSearchHit[]>([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [ingestOpen, setIngestOpen] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const fetchVideos = useCallback(async () => {
    try {
      const res = await api.getVideos()
      setVideos(res.videos)
      if (res.videos.length > 0 && !selectedVideo) {
        setSelectedVideo(res.videos[0].video_id)
      }
    } catch {
      // backend may not be running
    }
  }, [selectedVideo])

  useEffect(() => {
    fetchVideos()
  }, [fetchVideos])

  const handleSearch = useCallback(async () => {
    const q = query.trim()
    if (!q) return
    if (scope === 'video' && !selectedVideo) return

    setLoading(true)
    setError(null)
    setSearched(true)

    try {
      const res = await api.query({
        query: q,
        mode,
        scope,
        video_id: scope === 'video' ? selectedVideo : undefined,
        k: 50,
        ingest_if_needed: false,
      })
      setHits(res.hits)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed')
      setHits([])
    } finally {
      setLoading(false)
    }
  }, [query, mode, scope, selectedVideo])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch()
  }

  const handleModeChange = (m: SearchMode) => {
    setMode(m)
    setHits([])
    setSearched(false)
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/7 flex-shrink-0">
        <div>
          <h1 className="text-lg font-semibold text-vs-text">Semantic Video Search</h1>
          <p className="text-xs text-vs-muted">{videos.length} video{videos.length !== 1 ? 's' : ''} indexed — search by meaning, not keywords</p>
        </div>
        <Button variant="default" size="sm" onClick={() => setIngestOpen(true)}>
          <Plus className="h-4 w-4" />
          Add Video
        </Button>
      </div>

      {/* Search area */}
      <div className="flex-shrink-0 px-6 pt-5 pb-4 space-y-3 border-b border-white/7">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-vs-muted pointer-events-none" />
            <input
              ref={inputRef}
              className="w-full h-11 pl-10 pr-4 rounded-lg border border-white/8 bg-vs-surface-2 text-sm text-vs-text placeholder:text-vs-muted focus:outline-none focus:border-vs-accent/50 focus:ring-1 focus:ring-vs-accent/20 transition-colors"
              placeholder={MODE_PLACEHOLDER[mode]}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              autoFocus
            />
          </div>
          <Button
            variant="default"
            size="md"
            className="h-11 px-5"
            onClick={handleSearch}
            disabled={loading || !query.trim()}
          >
            {loading ? <Spinner size="sm" /> : <Search className="h-4 w-4" />}
            Search
          </Button>
        </div>

        <ModeSelector value={mode} onChange={handleModeChange} />

        <FilterPanel
          videos={videos}
          selectedVideo={selectedVideo}
          onVideoChange={setSelectedVideo}
          scope={scope}
          onScopeChange={setScope}
        />
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto px-6 py-5">
        {loading && (
          <div className="flex flex-col items-center justify-center py-24 gap-4">
            <Spinner size="lg" />
            <p className="text-sm text-vs-muted">Searching…</p>
          </div>
        )}

        {error && !loading && (
          <div className="rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3">
            <p className="text-sm font-medium text-red-400">Error</p>
            <p className="text-xs text-vs-muted mt-1">{error}</p>
          </div>
        )}

        {!loading && !error && searched && hits.length === 0 && (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <div className="h-14 w-14 rounded-2xl bg-vs-surface-2 border border-white/7 flex items-center justify-center mb-4">
              <Search className="h-6 w-6 text-vs-subtle" />
            </div>
            <h3 className="text-base font-semibold text-vs-text mb-1">No results</h3>
            <p className="text-sm text-vs-muted">Try a different query or switch search mode.</p>
          </div>
        )}

        {!loading && !error && !searched && (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <div className="h-14 w-14 rounded-2xl bg-gradient-accent flex items-center justify-center mb-4 shadow-glow">
              <Search className="h-6 w-6 text-white" />
            </div>
            <h3 className="text-base font-semibold text-vs-text mb-2">Find any moment in a video</h3>
            <p className="text-sm text-vs-muted max-w-sm leading-relaxed">
              {MODE_EMPTY_HINT[mode]}
            </p>
            {videos.length === 0 && (
              <button
                onClick={() => setIngestOpen(true)}
                className="mt-5 text-sm text-vs-accent-light hover:text-vs-accent transition-colors underline underline-offset-2"
              >
                Add a YouTube video to get started →
              </button>
            )}
          </div>
        )}

        {!loading && hits.length > 0 && (
          <div>
            <p className="text-xs text-vs-muted mb-4">
              {hits.length} moment{hits.length !== 1 ? 's' : ''} found
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {hits.map((hit, i) => (
                <ResultCard key={`${hit.video_id}-${hit.start}-${i}`} hit={hit} index={i} />
              ))}
            </div>
          </div>
        )}
      </div>

      <IngestModal
        open={ingestOpen}
        onOpenChange={setIngestOpen}
        onSuccess={() => fetchVideos()}
      />
    </div>
  )
}
