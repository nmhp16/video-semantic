import { useState } from 'react'
import { ChevronDown, Globe, Video } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { VideoMeta, SearchScope } from '@/lib/api'

interface FilterPanelProps {
  videos: VideoMeta[]
  selectedVideo: string
  onVideoChange: (id: string) => void
  scope: SearchScope
  onScopeChange: (scope: SearchScope) => void
  filterObjects: string
  onFilterObjectsChange: (val: string) => void
  objectSuggestions: string[]
}

function SegmentedScope({
  scope,
  onScopeChange,
}: Pick<FilterPanelProps, 'scope' | 'onScopeChange'>) {
  return (
    <div className="inline-flex items-center rounded-lg border border-border bg-surface p-0.5">
      <button
        onClick={() => onScopeChange('video')}
        className={cn(
          'inline-flex items-center gap-1.5 h-7 px-2.5 text-xs font-medium rounded-md transition-colors',
          scope === 'video'
            ? 'bg-surface2 text-fg shadow-[inset_0_0_0_1px_#3F3F46]'
            : 'text-muted hover:text-fg',
        )}
      >
        <Video className="h-3 w-3" />
        Single
      </button>
      <button
        onClick={() => onScopeChange('global')}
        className={cn(
          'inline-flex items-center gap-1.5 h-7 px-2.5 text-xs font-medium rounded-md transition-colors',
          scope === 'global'
            ? 'bg-surface2 text-fg shadow-[inset_0_0_0_1px_#3F3F46]'
            : 'text-muted hover:text-fg',
        )}
      >
        <Globe className="h-3 w-3" />
        All
      </button>
    </div>
  )
}

export function FilterPanel({
  videos,
  selectedVideo,
  onVideoChange,
  scope,
  onScopeChange,
  filterObjects,
  onFilterObjectsChange,
  objectSuggestions,
}: FilterPanelProps) {
  const [open, setOpen] = useState(false)

  const dot = (active: boolean, color: string) =>
    cn('w-1.5 h-1.5 rounded-full', active ? color : 'bg-dim')

  return (
    <div className="flex flex-col gap-2">
      {/* Always-visible controls */}
      <div className="flex flex-wrap items-center gap-2">
        <SegmentedScope scope={scope} onScopeChange={onScopeChange} />

        {scope === 'video' && (
          <div className="inline-flex items-center gap-2">
            <label className="text-xs text-muted">Video</label>
            {videos.length === 0 ? (
              <span className="text-xs text-dim italic">none ingested</span>
            ) : (
              <select
                value={selectedVideo}
                onChange={(e) => onVideoChange(e.target.value)}
                className={cn(
                  'h-7 rounded-md border border-border bg-surface pl-2.5 pr-7 text-xs font-mono text-fg',
                  'hover:border-border-strong focus:outline-none focus:border-accent/50 focus:ring-2 focus:ring-accent-ring',
                  'appearance-none bg-[url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%228%22%20height%3D%228%22%20viewBox%3D%220%200%2012%2012%22%3E%3Cpath%20fill%3D%22%23A1A1AA%22%20d%3D%22M6%209L1%204h10z%22%2F%3E%3C%2Fsvg%3E")] bg-no-repeat bg-[right_0.5rem_center]',
                )}
              >
                {videos.map((v) => (
                  <option key={v.video_id} value={v.video_id}>
                    {v.title ?? v.video_id}
                  </option>
                ))}
              </select>
            )}
          </div>
        )}

        <button
          onClick={() => setOpen(!open)}
          className={cn(
            'inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md text-xs font-medium transition-colors',
            open
              ? 'bg-surface2 text-fg border border-border-strong'
              : 'text-muted hover:text-fg hover:bg-surface border border-border',
          )}
        >
          Filters
          <ChevronDown className={cn('h-3 w-3 transition-transform', open && 'rotate-180')} />
        </button>
      </div>

      {/* Collapsible filter body */}
      {open && (
        <div className="rounded-lg border border-border bg-surface/50 p-3 animate-fade-in space-y-3">
          <div>
            <label className="block text-xxs font-medium uppercase tracking-wide text-subtle mb-1.5">
              Filter by keyword
            </label>
            <input
              list="filter-objects-list"
              placeholder="person, knife, cutting board"
              value={filterObjects}
              onChange={(e) => onFilterObjectsChange(e.target.value)}
              className="h-8 w-full rounded-md border border-border bg-surface px-3 text-xs text-fg placeholder:text-dim hover:border-border-strong focus:outline-none focus:border-accent/50 focus:ring-2 focus:ring-accent-ring transition-colors"
            />
            <datalist id="filter-objects-list">
              {objectSuggestions.map((obj) => (
                <option key={obj} value={obj} />
              ))}
            </datalist>
            <p className="mt-1 text-xxs text-subtle">
              Matches words in the lazy-generated caption for each frame.
            </p>
          </div>

          {scope === 'video' && videos.length > 0 && (
            <div>
              <label className="block text-xxs font-medium uppercase tracking-wide text-subtle mb-1.5">
                Index status
              </label>
              <div className="flex flex-wrap gap-1.5">
                {videos
                  .filter((v) => v.video_id === selectedVideo)
                  .map((v) => (
                    <div
                      key={v.video_id}
                      className="inline-flex items-center gap-3 text-xs text-muted"
                    >
                      <span className="inline-flex items-center gap-1.5">
                        <span className={dot(v.has_text_search, 'bg-accent')} /> Text
                      </span>
                      <span className="inline-flex items-center gap-1.5">
                        <span className={dot(v.has_visual_search, 'bg-emerald-400')} /> Visual
                      </span>
                      <span className="inline-flex items-center gap-1.5">
                        <span className={dot(v.has_action_search, 'bg-amber-400')} /> Action
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
