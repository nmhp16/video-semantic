import { useState, useCallback, useEffect, useRef } from 'react'
import { Search, Plus, X } from 'lucide-react'
import { ModeSelector } from '@/components/ModeSelector'
import { FilterPanel } from '@/components/FilterPanel'
import { ResultCard } from '@/components/ResultCard'
import { IngestModal } from '@/components/IngestModal'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Spinner } from '@/components/ui/spinner'
import { api } from '@/lib/api'
import type { SearchMode, SearchScope, UnifiedSearchHit, VideoMeta } from '@/lib/api'

function ChainStepsInput({
  steps,
  onChange,
}: {
  steps: string[]
  onChange: (steps: string[]) => void
}) {
  const addStep = () => onChange([...steps, ''])
  const updateStep = (i: number, val: string) => {
    const next = [...steps]
    next[i] = val
    onChange(next)
  }
  const removeStep = (i: number) => {
    const next = steps.filter((_, idx) => idx !== i)
    onChange(next.length ? next : [''])
  }

  return (
    <div className="space-y-2">
      <p className="text-xs text-vs-muted font-medium">Action sequence (in order)</p>
      {steps.map((step, i) => (
        <div key={i} className="flex gap-2 items-center">
          <span className="flex-shrink-0 text-xs text-vs-subtle w-4 text-right">{i + 1}.</span>
          <Input
            placeholder={`Step ${i + 1} — e.g. "person picks up cup"`}
            value={step}
            onChange={(e) => updateStep(i, e.target.value)}
            className="h-8 text-xs"
          />
          {steps.length > 1 && (
            <button
              onClick={() => removeStep(i)}
              className="flex-shrink-0 text-vs-muted hover:text-red-400 transition-colors"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      ))}
      <button
        onClick={addStep}
        className="text-xs text-vs-accent-light hover:text-vs-accent transition-colors flex items-center gap-1"
      >
        <Plus className="h-3 w-3" /> Add step
      </button>
    </div>
  )
}

export function SearchPage() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('text')
  const [scope, setScope] = useState<SearchScope>('global')
  const [selectedVideo, setSelectedVideo] = useState('')
  const [filterObjects, setFilterObjects] = useState('')
  const [chainSteps, setChainSteps] = useState([''])
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
      // silently fail — backend may not be running
    }
  }, [selectedVideo])

  useEffect(() => {
    fetchVideos()
  }, [fetchVideos])

  const handleSearch = useCallback(async () => {
    const isChain = mode === 'action_chain'
    const effectiveQuery = isChain ? undefined : query.trim()
    const effectiveSteps = isChain ? chainSteps.filter(Boolean) : undefined

    if (!isChain && !effectiveQuery) return
    if (isChain && (!effectiveSteps || effectiveSteps.length === 0)) return
    if (scope === 'video' && !selectedVideo) return

    setLoading(true)
    setError(null)
    setSearched(true)

    try {
      const res = await api.query({
        query: effectiveQuery,
        mode,
        scope,
        video_id: scope === 'video' ? selectedVideo : undefined,
        filter_objects: filterObjects.trim() || undefined,
        steps: effectiveSteps,
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
  }, [query, mode, scope, selectedVideo, filterObjects, chainSteps])

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
          <h1 className="text-lg font-semibold text-vs-text">Search</h1>
          <p className="text-xs text-vs-muted">{videos.length} video{videos.length !== 1 ? 's' : ''} indexed</p>
        </div>
        <Button variant="default" size="sm" onClick={() => setIngestOpen(true)}>
          <Plus className="h-4 w-4" />
          Add Video
        </Button>
      </div>

      {/* Search area */}
      <div className="flex-shrink-0 px-6 pt-5 pb-4 space-y-3 border-b border-white/7">
        {mode !== 'action_chain' ? (
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-vs-muted pointer-events-none" />
              <input
                ref={inputRef}
                className="w-full h-11 pl-10 pr-4 rounded-lg border border-white/8 bg-vs-surface-2 text-sm text-vs-text placeholder:text-vs-muted focus:outline-none focus:border-vs-accent/50 focus:ring-1 focus:ring-vs-accent/20 transition-colors"
                placeholder={
                  mode === 'text' ? 'Search by transcript…'
                  : mode === 'visual' ? 'Describe a scene or object…'
                  : 'Describe an action or activity…'
                }
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
        ) : (
          <div className="space-y-3">
            <ChainStepsInput steps={chainSteps} onChange={setChainSteps} />
            <Button
              variant="default"
              size="md"
              onClick={handleSearch}
              disabled={loading || chainSteps.filter(Boolean).length === 0}
            >
              {loading ? <Spinner size="sm" /> : <Search className="h-4 w-4" />}
              Find Sequence
            </Button>
          </div>
        )}

        <ModeSelector value={mode} onChange={handleModeChange} />

        <FilterPanel
          videos={videos}
          selectedVideo={selectedVideo}
          onVideoChange={setSelectedVideo}
          scope={scope}
          onScopeChange={setScope}
          filterObjects={filterObjects}
          onFilterObjectsChange={setFilterObjects}
        />
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto px-6 py-5">
        {loading && (
          <div className="flex flex-col items-center justify-center py-24 gap-4">
            <Spinner size="lg" />
            <p className="text-sm text-vs-muted">Searching{mode === 'action_chain' ? ' for sequence' : ''}…</p>
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
            <h3 className="text-base font-semibold text-vs-text mb-1">Search your videos</h3>
            <p className="text-sm text-vs-muted max-w-xs">
              Find moments by text transcript, visual scene, or action across your entire video library.
            </p>
          </div>
        )}

        {!loading && hits.length > 0 && (
          <div>
            <p className="text-xs text-vs-muted mb-4">
              {hits.length} result{hits.length !== 1 ? 's' : ''}
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
