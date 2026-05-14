import { useCallback, useEffect, useRef, useState } from 'react'
import { Plus, Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import { FilterPanel } from '@/components/FilterPanel'
import { IngestModal } from '@/components/IngestModal'
import { ModeSelector } from '@/components/ModeSelector'
import { ResultCard } from '@/components/ResultCard'
import { api } from '@/lib/api'
import type {
  SearchMode,
  SearchScope,
  UnifiedSearchHit,
  VideoMeta,
  ScoreRange,
} from '@/lib/api'
import { useSearchHistory } from '@/hooks/useSearchHistory'

const PLACEHOLDERS: Record<SearchMode, string> = {
  auto: 'Describe anything — scene, action, or object…',
  text: 'Search transcripts…',
  visual: 'Describe a scene or object…',
  action: 'Describe an action or activity…',
}

export function SearchPage() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('auto')
  const [scope, setScope] = useState<SearchScope>('video')
  const [selectedVideo, setSelectedVideo] = useState('')
  const [filterObjects, setFilterObjects] = useState('')
  const [videos, setVideos] = useState<VideoMeta[]>([])
  const [hits, setHits] = useState<UnifiedSearchHit[]>([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [ingestOpen, setIngestOpen] = useState(false)
  const [libraryError, setLibraryError] = useState<string | null>(null)
  const [scoreRange, setScoreRange] = useState<ScoreRange | null>(null)
  const { history, push: pushHistory, remove: removeHistory } = useSearchHistory()
  const [inputFocused, setInputFocused] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const searchAbortRef = useRef<AbortController | null>(null)

  const videoTitles = Object.fromEntries(
    videos.filter((v) => v.title).map((v) => [v.video_id, v.title!])
  )

  const objectSuggestions = scope === 'video'
    ? (videos.find((v) => v.video_id === selectedVideo)?.top_objects ?? [])
    : [...new Set(videos.flatMap((v) => v.top_objects ?? []))].slice(0, 20)

  const fetchVideos = useCallback(async () => {
    try {
      const res = await api.getVideos()
      setVideos(res.videos)
      setSelectedVideo((prev) => prev || res.videos[0]?.video_id || '')
      setLibraryError(null)
    } catch (e) {
      setLibraryError(
        e instanceof Error ? e.message : 'Could not reach backend',
      )
    }
  }, [])

  useEffect(() => {
    fetchVideos()
  }, [fetchVideos])

  // Keyboard shortcut: Cmd/Ctrl+K to focus search
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const handleSearch = useCallback(async () => {
    const q = query.trim()
    if (!q) return
    if (scope === 'video' && !selectedVideo) return

    searchAbortRef.current?.abort()
    const controller = new AbortController()
    searchAbortRef.current = controller

    setLoading(true)
    setError(null)
    setSearched(true)

    try {
      const res = await api.query({
        query: q,
        mode,
        scope,
        video_id: scope === 'video' ? selectedVideo : undefined,
        filter_objects: filterObjects.trim() || undefined,
        k: 24,
        ingest_if_needed: false,
      }, controller.signal)
      setHits(res.hits)
      setScoreRange(res.score_range ?? null)
      pushHistory(q, mode)
    } catch (e) {
      if (e instanceof Error && e.message === 'canceled') return
      setError(e instanceof Error ? e.message : 'Search failed')
      setHits([])
    } finally {
      setLoading(false)
    }
  }, [query, mode, scope, selectedVideo, filterObjects, pushHistory])


  const handleModeChange = (m: SearchMode) => {
    setMode(m)
    setHits([])
    setSearched(false)
  }

  return (
    <div className="flex flex-col py-6 gap-5">
      {/* Header */}
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-fg tracking-tight">Search</h1>
          <p className="mt-0.5 text-xs text-muted">
            {videos.length} video{videos.length !== 1 ? 's' : ''} indexed
          </p>
        </div>
        <Button variant="primary" size="sm" onClick={() => setIngestOpen(true)}>
          <Plus className="h-3.5 w-3.5" />
          Add video
        </Button>
      </div>

      {libraryError && (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2.5">
          <p className="text-sm font-medium text-red-400">Backend unreachable</p>
          <p className="mt-0.5 text-xs text-muted break-all">{libraryError}</p>
        </div>
      )}

      {/* Search input area */}
      <div className="space-y-3">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-subtle pointer-events-none" />
            <input
              ref={inputRef}
              className="w-full h-11 pl-10 pr-20 rounded-lg border border-border bg-surface text-sm text-fg placeholder:text-dim hover:border-border-strong focus:outline-none focus:border-accent/50 focus:ring-2 focus:ring-accent-ring transition-colors"
              placeholder={PLACEHOLDERS[mode]}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              onFocus={() => setInputFocused(true)}
              onBlur={() => setTimeout(() => setInputFocused(false), 150)}
            />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 inline-flex items-center gap-1 pointer-events-none">
              <span className="kbd">⌘</span>
              <span className="kbd">K</span>
            </span>
          </div>
          <Button
            variant="primary"
            size="lg"
            onClick={handleSearch}
            disabled={loading || !query.trim()}
          >
            {loading ? <Spinner size="sm" className="text-white" /> : <Search className="h-4 w-4" />}
            Search
          </Button>
        </div>
        {inputFocused && history.length > 0 && !query && (
          <div className="flex flex-wrap gap-1.5">
            {history.map((entry) => (
              <button
                key={`${entry.query}-${entry.mode}`}
                onMouseDown={(e) => {
                  e.preventDefault()
                  setQuery(entry.query)
                  setMode(entry.mode)
                  setTimeout(() => handleSearch(), 0)
                }}
                className="group inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1 text-xs text-muted hover:border-border-strong hover:text-fg transition-colors"
              >
                <span>{entry.query}</span>
                <span className="text-[10px] text-dim">{entry.mode}</span>
                <span
                  onMouseDown={(e) => { e.stopPropagation(); removeHistory(entry.query, entry.mode) }}
                  className="ml-0.5 text-dim hover:text-red-400 cursor-pointer"
                >
                  ×
                </span>
              </button>
            ))}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <ModeSelector value={mode} onChange={handleModeChange} />
          <div className="flex-1" />
        </div>

        <FilterPanel
          videos={videos}
          selectedVideo={selectedVideo}
          onVideoChange={setSelectedVideo}
          scope={scope}
          onScopeChange={setScope}
          filterObjects={filterObjects}
          onFilterObjectsChange={setFilterObjects}
          objectSuggestions={objectSuggestions}
        />
      </div>

      <div className="divider" />

      {/* Results */}
      <div>
        {loading && (
          <div className="flex flex-col items-center justify-center py-24 gap-3">
            <Spinner size="lg" className="text-accent" />
            <p className="text-xs text-muted">Searching…</p>
          </div>
        )}

        {error && !loading && (
          <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2.5">
            <p className="text-sm font-medium text-red-400">Request failed</p>
            <p className="mt-0.5 text-xs text-muted break-all">{error}</p>
          </div>
        )}

        {!loading && !error && searched && hits.length === 0 && (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <h3 className="text-sm font-semibold text-fg">No results</h3>
            <p className="mt-1 text-xs text-muted">
              Try a different query or switch search mode.
            </p>
          </div>
        )}

        {!loading && !error && !searched && (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-md bg-surface2 border border-border">
              <Search className="h-4 w-4 text-muted" />
            </div>
            <h3 className="text-sm font-semibold text-fg">
              Search your video library
            </h3>
            <p className="mt-1 max-w-xs text-xs text-muted">
              Find moments by transcript, visual scene, or action.
              Press <span className="kbd ml-0.5 mr-0.5">⌘</span>
              <span className="kbd">K</span> to focus the search box.
            </p>
          </div>
        )}

        {!loading && hits.length > 0 && (
          <div>
            <p className="mb-3 text-xs text-muted">
              {hits.length} result{hits.length !== 1 ? 's' : ''}
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {hits.map((hit, i) => (
                <ResultCard
                  key={`${hit.video_id}-${hit.start}-${i}`}
                  hit={hit}
                  index={i}
                  title={videoTitles[hit.video_id]}
                  scoreRange={scoreRange}
                />
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
