import { useState, useEffect } from 'react'
import { Film, Type, Eye, Activity } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn, isYouTubeId } from '@/lib/utils'
import { fetchVideoTitle, getCachedTitle, ytThumbnail } from '@/lib/youtube'
import type { VideoMeta } from '@/lib/api'

interface VideoLibraryProps {
  videos: VideoMeta[]
  loading: boolean
  onSelect?: (videoId: string) => void
}

function SkeletonCard() {
  return (
    <div className="glass-card p-4 space-y-3">
      <div className="skeleton h-4 w-3/4 rounded" />
      <div className="flex gap-2">
        <div className="skeleton h-5 w-12 rounded-full" />
        <div className="skeleton h-5 w-14 rounded-full" />
      </div>
    </div>
  )
}

function VideoCard({ video, onSelect }: { video: VideoMeta; onSelect?: (id: string) => void }) {
  const [title, setTitle] = useState(getCachedTitle(video.video_id))
  const isYT = isYouTubeId(video.video_id)

  useEffect(() => {
    fetchVideoTitle(video.video_id).then(setTitle)
  }, [video.video_id])

  return (
    <button
      onClick={() => onSelect?.(video.video_id)}
      className={cn(
        'glass-card overflow-hidden text-left w-full transition-all duration-150',
        'hover:border-white/12 hover:shadow-card-hover hover:bg-vs-surface-2/80',
        onSelect && 'cursor-pointer'
      )}
    >
      {/* Thumbnail */}
      {isYT ? (
        <img
          src={ytThumbnail(video.video_id)}
          alt={title}
          className="w-full aspect-video object-cover"
        />
      ) : (
        <div className="w-full aspect-video bg-vs-surface-3 flex items-center justify-center">
          <Film className="h-7 w-7 text-vs-subtle" />
        </div>
      )}

      <div className="p-3 space-y-2">
        <div className="min-w-0">
          <p className="text-sm font-medium text-vs-text truncate">{title}</p>
          <p className="text-xs text-vs-muted mt-0.5">
            {[video.has_text_search, video.has_visual_search, video.has_action_search].filter(Boolean).length} / 3 indexes ready
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <Badge variant={video.has_text_search ? 'success' : 'secondary'} className="text-[10px]">
            <Type className="h-2.5 w-2.5" />
            Text
          </Badge>
          <Badge variant={video.has_visual_search ? 'success' : 'secondary'} className="text-[10px]">
            <Eye className="h-2.5 w-2.5" />
            Visual
          </Badge>
          <Badge variant={video.has_action_search ? 'success' : 'secondary'} className="text-[10px]">
            <Activity className="h-2.5 w-2.5" />
            Action
          </Badge>
        </div>
      </div>
    </button>
  )
}

export function VideoLibrary({ videos, loading, onSelect }: VideoLibraryProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
      </div>
    )
  }

  if (videos.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <div className="h-16 w-16 rounded-2xl bg-vs-surface-2 border border-white/7 flex items-center justify-center mb-4">
          <Film className="h-7 w-7 text-vs-subtle" />
        </div>
        <h3 className="text-base font-semibold text-vs-text mb-1">No videos yet</h3>
        <p className="text-sm text-vs-muted max-w-xs">Add a YouTube video to start indexing and searching.</p>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {videos.map((video) => (
        <VideoCard key={video.video_id} video={video} onSelect={onSelect} />
      ))}
    </div>
  )
}
