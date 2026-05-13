import { useState } from 'react'
import { ExternalLink, Film, RefreshCw, Trash2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Spinner } from '@/components/ui/spinner'
import { cn, isYouTubeId, youtubeUrl } from '@/lib/utils'
import { api } from '@/lib/api'
import type { VideoMeta } from '@/lib/api'

interface VideoLibraryProps {
  videos: VideoMeta[]
  loading: boolean
  onSelect?: (videoId: string) => void
  onDeleted?: (videoId: string) => void
  onReindexed?: () => void
}

function StatusDot({ active, label }: { active: boolean; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted">
      <span
        className={cn(
          'w-1.5 h-1.5 rounded-full',
          active ? 'bg-emerald-400' : 'bg-dim',
        )}
      />
      {label}
    </span>
  )
}

function SkeletonRow() {
  return (
    <tr className="border-b border-border">
      <td className="px-4 py-3"><div className="skeleton h-10 w-16 rounded" /></td>
      <td className="px-4 py-3"><div className="skeleton h-3 w-32 rounded" /></td>
      <td className="px-4 py-3"><div className="skeleton h-3 w-48 rounded" /></td>
      <td className="px-4 py-3"><div className="skeleton h-3 w-16 rounded" /></td>
      <td className="px-4 py-3"><div className="skeleton h-3 w-6 rounded ml-auto" /></td>
    </tr>
  )
}

function Thumbnail({ video }: { video: VideoMeta }) {
  if (video.thumbnail_url) {
    return (
      <img
        src={api.assetUrl(video.thumbnail_url)}
        alt={`Thumbnail for ${video.video_id}`}
        className="h-10 w-16 rounded-sm object-cover bg-surface2"
        loading="lazy"
      />
    )
  }
  return (
    <div className="h-10 w-16 rounded-sm bg-surface2 border border-border flex items-center justify-center">
      <Film className="h-4 w-4 text-dim" />
    </div>
  )
}

export function VideoLibrary({ videos, loading, onSelect, onDeleted, onReindexed }: VideoLibraryProps) {
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [reindexing, setReindexing] = useState<string | null>(null)

  const handleDelete = async (videoId: string) => {
    setDeleting(videoId)
    setDeleteError(null)
    try {
      await api.deleteVideo(videoId)
      setPendingDelete(null)
      onDeleted?.(videoId)
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : 'Delete failed')
    } finally {
      setDeleting(null)
    }
  }

  const handleReindex = async (videoId: string) => {
    setReindexing(videoId)
    try {
      await api.buildContexts([videoId])
      onReindexed?.()
    } finally {
      setReindexing(null)
    }
  }

  if (loading) {
    return (
      <div className="rounded-lg border border-border bg-panel overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-surface/50">
              <th className="px-4 py-2.5 text-left text-xxs font-medium uppercase tracking-wide text-subtle"> </th>
              <th className="px-4 py-2.5 text-left text-xxs font-medium uppercase tracking-wide text-subtle">Title / ID</th>
              <th className="px-4 py-2.5 text-left text-xxs font-medium uppercase tracking-wide text-subtle">Indexes</th>
              <th className="px-4 py-2.5 text-left text-xxs font-medium uppercase tracking-wide text-subtle">Ready</th>
              <th className="px-4 py-2.5 text-right text-xxs font-medium uppercase tracking-wide text-subtle"> </th>
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: 4 }).map((_, i) => <SkeletonRow key={i} />)}
          </tbody>
        </table>
      </div>
    )
  }

  if (videos.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-20 px-6 text-center">
        <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-md bg-surface2 border border-border">
          <div className="h-1.5 w-4 rounded-sm bg-dim" />
        </div>
        <h3 className="text-sm font-semibold text-fg">No videos yet</h3>
        <p className="mt-1 text-xs text-muted max-w-xs">
          Add a YouTube URL to download, transcribe, and build search indexes.
        </p>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-border bg-panel overflow-hidden">
      {deleteError && (
        <div className="border-b border-red-500/30 bg-red-500/10 px-4 py-2 text-xs text-red-300">
          {deleteError}
        </div>
      )}
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-surface/50">
            <th className="px-4 py-2.5 text-left text-xxs font-medium uppercase tracking-wide text-subtle w-20"> </th>
            <th className="px-4 py-2.5 text-left text-xxs font-medium uppercase tracking-wide text-subtle">
              Title / ID
            </th>
            <th className="px-4 py-2.5 text-left text-xxs font-medium uppercase tracking-wide text-subtle">
              Indexes
            </th>
            <th className="px-4 py-2.5 text-left text-xxs font-medium uppercase tracking-wide text-subtle">
              Status
            </th>
            <th className="px-4 py-2.5 text-right text-xxs font-medium uppercase tracking-wide text-subtle"> </th>
          </tr>
        </thead>
        <tbody>
          {videos.map((v) => {
            const ready = [v.has_text_search, v.has_visual_search, v.has_action_search].filter(Boolean).length
            const all = ready === 3
            const confirming = pendingDelete === v.video_id
            const isDeleting = deleting === v.video_id
            return (
              <tr
                key={v.video_id}
                onClick={() => !confirming && onSelect?.(v.video_id)}
                className={cn(
                  'border-b border-border last:border-0 transition-colors',
                  onSelect && !confirming && 'cursor-pointer hover:bg-surface/50',
                )}
              >
                <td className="px-4 py-3">
                  <Thumbnail video={v} />
                </td>
                <td className="px-4 py-3">
                  <div className="flex flex-col gap-0.5">
                    <span className="text-sm font-medium text-fg truncate max-w-xs">
                      {v.title ?? v.video_id}
                    </span>
                    {v.title && (
                      <span className="font-mono text-[10px] text-dim">{v.video_id}</span>
                    )}
                    {v.source_url && (
                      <a href={v.source_url} target="_blank" rel="noreferrer noopener"
                         onClick={(e) => e.stopPropagation()}
                         className="inline-flex items-center gap-1 text-[10px] text-subtle hover:text-fg transition-colors">
                        Source <ExternalLink className="h-2.5 w-2.5" />
                      </a>
                    )}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-4">
                    <StatusDot active={v.has_text_search} label="Text" />
                    <StatusDot active={v.has_visual_search} label="Visual" />
                    <span className="inline-flex items-center gap-1.5">
                      <StatusDot active={v.has_action_search} label="Action" />
                      {v.has_xclip_action && (
                        <Badge variant="neutral" className="text-[9px] px-1 py-0 leading-tight" title="Temporal action search — understands motion between frames">X-CLIP</Badge>
                      )}
                      {v.has_action_search && !v.has_xclip_action && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 leading-tight text-dim" title="Older action index — re-ingest to upgrade">legacy</Badge>
                      )}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <Badge variant={all ? 'success' : ready > 0 ? 'warning' : 'outline'}>
                    {ready}/3 ready
                  </Badge>
                </td>
                <td className="px-4 py-3 text-right">
                  <div className="flex items-center justify-end gap-3">
                    <button
                      onClick={(e) => { e.stopPropagation(); handleReindex(v.video_id) }}
                      disabled={reindexing === v.video_id}
                      title="Rebuild search index"
                      className="rounded-md p-1 text-subtle hover:text-fg hover:bg-surface2 transition-colors disabled:opacity-40"
                    >
                      {reindexing === v.video_id
                        ? <Spinner size="sm" />
                        : <RefreshCw className="h-3.5 w-3.5" />}
                    </button>
                    {isYouTubeId(v.video_id) && (
                      <a
                        href={youtubeUrl(v.video_id)}
                        target="_blank"
                        rel="noreferrer noopener"
                        onClick={(e) => e.stopPropagation()}
                        className="inline-flex items-center gap-1 text-xs text-subtle hover:text-fg transition-colors"
                      >
                        YouTube
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    )}
                    {confirming ? (
                      <div className="inline-flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                        <span className="text-xs text-muted">Delete?</span>
                        <button
                          onClick={() => handleDelete(v.video_id)}
                          disabled={isDeleting}
                          className="inline-flex items-center gap-1 rounded-md border border-red-500/40 bg-red-500/10 px-2 py-0.5 text-xs text-red-300 hover:bg-red-500/20 disabled:opacity-50"
                        >
                          {isDeleting ? <Spinner size="sm" /> : null}
                          Yes
                        </button>
                        <button
                          onClick={() => setPendingDelete(null)}
                          disabled={isDeleting}
                          className="rounded-md border border-border px-2 py-0.5 text-xs text-muted hover:text-fg disabled:opacity-50"
                        >
                          No
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={(e) => { e.stopPropagation(); setPendingDelete(v.video_id); setDeleteError(null) }}
                        aria-label={`Delete ${v.video_id}`}
                        className="rounded-md p-1 text-subtle hover:text-red-400 hover:bg-surface2 transition-colors"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
