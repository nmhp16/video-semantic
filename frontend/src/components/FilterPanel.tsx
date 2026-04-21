import { useState, useEffect } from 'react'
import { ChevronDown, ChevronUp, Globe, Video } from 'lucide-react'
import { cn } from '@/lib/utils'
import { fetchVideoTitle, getCachedTitle } from '@/lib/youtube'
import type { VideoMeta, SearchScope } from '@/lib/api'

interface FilterPanelProps {
  videos: VideoMeta[]
  selectedVideo: string
  onVideoChange: (id: string) => void
  scope: SearchScope
  onScopeChange: (scope: SearchScope) => void
}

export function FilterPanel({ videos, selectedVideo, onVideoChange, scope, onScopeChange }: FilterPanelProps) {
  const [open, setOpen] = useState(false)
  const [, forceUpdate] = useState(0)

  // Fetch titles for all video IDs
  useEffect(() => {
    Promise.all(videos.map(v => fetchVideoTitle(v.video_id))).then(() => forceUpdate(n => n + 1))
  }, [videos.map(v => v.video_id).join(',')])

  const selectedTitle = selectedVideo ? getCachedTitle(selectedVideo) : null

  return (
    <div className="rounded-lg border border-white/7 bg-vs-surface/50">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-sm text-vs-muted hover:text-vs-text transition-colors"
      >
        <span className="font-medium truncate">
          {scope === 'global'
            ? 'All videos'
            : selectedTitle ?? (selectedVideo || 'Select video')}
        </span>
        {open ? <ChevronUp className="h-4 w-4 flex-shrink-0" /> : <ChevronDown className="h-4 w-4 flex-shrink-0" />}
      </button>

      {open && (
        <div className="border-t border-white/7 px-4 py-4 space-y-4 animate-fade-in">
          {/* Scope */}
          <div>
            <label className="block text-xs font-medium text-vs-muted mb-2">Search across</label>
            <div className="flex gap-2">
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
            </div>
          </div>

          {/* Video select */}
          {scope === 'video' && (
            <div>
              <label className="block text-xs font-medium text-vs-muted mb-2">Choose video</label>
              {videos.length === 0 ? (
                <p className="text-xs text-vs-subtle italic">No videos ingested yet</p>
              ) : (
                <div className="space-y-1 max-h-48 overflow-y-auto pr-1">
                  {videos.map((v) => {
                    const title = getCachedTitle(v.video_id)
                    return (
                      <button
                        key={v.video_id}
                        onClick={() => onVideoChange(v.video_id)}
                        className={cn(
                          'w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs transition-all text-left',
                          selectedVideo === v.video_id
                            ? 'bg-vs-accent/15 text-vs-accent-light border border-vs-accent/25'
                            : 'text-vs-muted hover:bg-white/5 border border-transparent'
                        )}
                      >
                        <span className="truncate">{title}</span>
                        <div className="flex gap-1 flex-shrink-0 ml-2">
                          {v.has_text_search   && <span className="w-1.5 h-1.5 rounded-full bg-vs-accent-light" title="Text" />}
                          {v.has_visual_search && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" title="Visual" />}
                          {v.has_action_search && <span className="w-1.5 h-1.5 rounded-full bg-amber-400" title="Action" />}
                        </div>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
