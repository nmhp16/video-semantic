import { useState } from 'react'
import { ChevronDown, ChevronUp, Globe, Video } from 'lucide-react'
import { Input } from '@/components/ui/input'
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
}

export function FilterPanel({
  videos,
  selectedVideo,
  onVideoChange,
  scope,
  onScopeChange,
  filterObjects,
  onFilterObjectsChange,
}: FilterPanelProps) {
  const [open, setOpen] = useState(false)

  return (
    <div className="rounded-lg border border-white/7 bg-vs-surface/50">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-sm text-vs-muted hover:text-vs-text transition-colors"
      >
        <span className="font-medium">Filters</span>
        {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
      </button>

      {open && (
        <div className="border-t border-white/7 px-4 py-4 space-y-4 animate-fade-in">
          {/* Scope */}
          <div>
            <label className="block text-xs font-medium text-vs-muted mb-2">Scope</label>
            <div className="flex gap-2">
              <button
                onClick={() => onScopeChange('video')}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all',
                  scope === 'video'
                    ? 'bg-vs-accent/15 text-vs-accent-light border-vs-accent/30'
                    : 'border-white/8 text-vs-muted hover:bg-white/5'
                )}
              >
                <Video className="h-3 w-3" />
                Single Video
              </button>
              <button
                onClick={() => onScopeChange('global')}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all',
                  scope === 'global'
                    ? 'bg-vs-accent/15 text-vs-accent-light border-vs-accent/30'
                    : 'border-white/8 text-vs-muted hover:bg-white/5'
                )}
              >
                <Globe className="h-3 w-3" />
                All Videos
              </button>
            </div>
          </div>

          {/* Video select */}
          {scope === 'video' && (
            <div>
              <label className="block text-xs font-medium text-vs-muted mb-2">Video</label>
              {videos.length === 0 ? (
                <p className="text-xs text-vs-subtle italic">No videos ingested yet</p>
              ) : (
                <div className="space-y-1 max-h-40 overflow-y-auto pr-1">
                  {videos.map((v) => (
                    <button
                      key={v.video_id}
                      onClick={() => onVideoChange(v.video_id)}
                      className={cn(
                        'w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs transition-all',
                        selectedVideo === v.video_id
                          ? 'bg-vs-accent/15 text-vs-accent-light border border-vs-accent/25'
                          : 'text-vs-muted hover:bg-white/5 border border-transparent'
                      )}
                    >
                      <span className="truncate font-mono">{v.video_id}</span>
                      <div className="flex gap-1 flex-shrink-0 ml-2">
                        {v.has_text_search && <span className="w-1.5 h-1.5 rounded-full bg-vs-accent-light" title="Text" />}
                        {v.has_visual_search && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" title="Visual" />}
                        {v.has_action_search && <span className="w-1.5 h-1.5 rounded-full bg-amber-400" title="Action" />}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Object filter */}
          <div>
            <label className="block text-xs font-medium text-vs-muted mb-2">Filter by object</label>
            <Input
              placeholder="e.g. person, car, laptop"
              value={filterObjects}
              onChange={(e) => onFilterObjectsChange(e.target.value)}
              className="h-8 text-xs"
            />
          </div>
        </div>
      )}
    </div>
  )
}
