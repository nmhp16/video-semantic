import { useCallback, useEffect, useRef, useState } from 'react'
import { Plus, Search, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
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
} from '@/lib/api'

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
    <div className="rounded-lg border border-border bg-surface/40 p-3 space-y-2">
      <p className="text-xxs font-medium uppercase tracking-wide text-subtle">
        Action sequence
      </p>
      {steps.map((step, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-sm bg-surface2 font-mono text-xxs text-muted">
            {i + 1}
          </span>
          <Input
            placeholder={`Step ${i + 1}`}
            value={step}
            onChange={(e) => updateStep(i, e.target.value)}
            className="h-8 text-sm"
          />
          {steps.length > 1 && (
            <button
              onClick={() => removeStep(i)}
              className="flex-shrink-0 rounded-md p-1 text-subtle hover:text-red-400 hover:bg-surface2 transition-colors"
              aria-label="Remove step"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      ))}
      <button
        onClick={addStep}
        className="inline-flex items-center gap-1 text-xs text-muted hover:text-fg transition-colors"
      >
        <Plus className="h-3 w-3" />
        Add step
      </button>
    </div>
  )
}

const PLACEHOLDERS: Record<SearchMode, string> = {
  text: 'Search transcripts…',
  visual: 'Describe a scene or object…',
  action: 'Describe an action or activity…',
  action_chain: 'Describe each step in order…',
}

export function SearchPage() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('visual')
  const [scope, setScope] = useState<SearchScope>('video')
  const [selectedVideo, setSelectedVideo] = useState('')
  const [filterObjects, setFilterObjects] = useState('')
  const [chainSteps, setChainSteps] = useState([''])
  const [videos, setVideos] = useState<VideoMeta[]>([])
  const [hits, setHits] = useState<UnifiedSearchHit[]>([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [ingestOpen, setIngestOpen] = useState(false)
  const [libraryError, setLibraryError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

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
        k: 24,
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
        {mode !== 'action_chain' ? (
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
        ) : (
          <div className="space-y-3">
            <ChainStepsInput steps={chainSteps} onChange={setChainSteps} />
            <Button
              variant="primary"
              size="md"
              onClick={handleSearch}
              disabled={loading || chainSteps.filter(Boolean).length === 0}
            >
              {loading ? <Spinner size="sm" className="text-white" /> : <Search className="h-4 w-4" />}
              Find sequence
            </Button>
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
        />
      </div>

      <div className="divider" />

      {/* Results */}
      <div>
        {loading && (
          <div className="flex flex-col items-center justify-center py-24 gap-3">
            <Spinner size="lg" className="text-accent" />
            <p className="text-xs text-muted">
              {mode === 'action_chain' ? 'Solving sequence…' : 'Searching…'}
            </p>
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
