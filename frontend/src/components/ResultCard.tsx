import { useState } from 'react'
import { ExternalLink, Clock, Play } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Dialog } from '@/components/ui/dialog'
import { api } from '@/lib/api'
import { formatTimeRange, formatTime, scorePercent, isYouTubeId, youtubeUrl, truncate } from '@/lib/utils'
import { ytThumbnail } from '@/lib/youtube'
import type { UnifiedSearchHit } from '@/lib/api'

interface ResultCardProps {
  hit: UnifiedSearchHit
  index: number
}

function FrameImage({ framePath }: { framePath: string }) {
  const [failed, setFailed] = useState(false)
  const src = api.frameUrl(framePath)

  if (failed) {
    return (
      <div className="w-full aspect-video bg-vs-surface-3 flex items-center justify-center rounded-t-card">
        <Play className="h-8 w-8 text-vs-subtle" />
      </div>
    )
  }

  return (
    <img
      src={src}
      alt="frame"
      onError={() => setFailed(true)}
      className="w-full aspect-video object-cover rounded-t-card"
    />
  )
}

function YouTubeEmbed({ videoId, startSeconds }: { videoId: string; startSeconds: number }) {
  const src = `https://www.youtube.com/embed/${videoId}?start=${Math.floor(startSeconds)}&autoplay=1&rel=0`
  return (
    <div className="w-full aspect-video rounded-lg overflow-hidden border border-white/7">
      <iframe
        src={src}
        className="w-full h-full"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowFullScreen
        title={`${videoId} at ${formatTime(startSeconds)}`}
      />
    </div>
  )
}

export function ResultCard({ hit, index }: ResultCardProps) {
  const [playerOpen, setPlayerOpen] = useState(false)
  const pct = scorePercent(hit.score)
  const isYT = isYouTubeId(hit.video_id)
  const hasFrame = Boolean(hit.frame)

  return (
    <>
      <div
        className="group glass-card overflow-hidden hover:border-white/12 hover:shadow-card-hover transition-all duration-200 animate-slide-up cursor-pointer"
        style={{ animationDelay: `${index * 30}ms` }}
        onClick={() => setPlayerOpen(true)}
      >
        {/* Thumbnail — frame image, YouTube still, or fallback */}
        <div className="relative">
          {hasFrame ? (
            <FrameImage framePath={hit.frame!} />
          ) : isYT ? (
            <img
              src={ytThumbnail(hit.video_id)}
              alt="thumbnail"
              className="w-full aspect-video object-cover rounded-t-card"
            />
          ) : (
            <div className="w-full aspect-video bg-vs-surface-3 flex items-center justify-center">
              <Clock className="h-6 w-6 text-vs-subtle" />
            </div>
          )}
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center rounded-t-card">
            <div className="opacity-0 group-hover:opacity-100 transition-opacity h-10 w-10 rounded-full bg-white/90 flex items-center justify-center shadow-lg">
              <Play className="h-4 w-4 text-black ml-0.5" />
            </div>
          </div>
        </div>

        {/* Card body */}
        <div className="p-3 space-y-2.5">
          {/* Timestamp + score row */}
          <div className="flex items-center justify-between gap-2">
            <span className="flex items-center gap-1 text-xs font-medium text-vs-muted">
              <Clock className="h-3 w-3" />
              {formatTimeRange(hit.start, hit.end)}
            </span>
            <span className="text-xs font-semibold text-vs-accent-light">{pct}%</span>
          </div>

          {/* Score bar */}
          <div className="w-full bg-vs-surface-3 rounded-full h-0.5">
            <div className="score-bar h-0.5" style={{ width: `${pct}%` }} />
          </div>

          {/* Transcript text */}
          {hit.text && (
            <p className="text-xs text-vs-text leading-relaxed line-clamp-3">
              "{truncate(hit.text, 160)}"
            </p>
          )}

          {/* Caption (visual/action mode) */}
          {hit.caption && !hit.text && (
            <p className="text-xs text-vs-muted leading-relaxed line-clamp-3 italic">
              {truncate(hit.caption, 160)}
            </p>
          )}

          {/* Object badges */}
          {hit.objects && hit.objects.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {hit.objects.slice(0, 4).map((obj) => (
                <Badge key={obj} variant="secondary" className="text-[10px] px-1.5 py-0">
                  {obj}
                </Badge>
              ))}
              {hit.objects.length > 4 && (
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                  +{hit.objects.length - 4}
                </Badge>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Player dialog */}
      <Dialog
        open={playerOpen}
        onOpenChange={setPlayerOpen}
        title={`${hit.video_id}`}
        description={`${formatTimeRange(hit.start, hit.end)} · ${pct}% match`}
        className="max-w-3xl"
      >
        <div className="space-y-4">
          {/* Embedded YouTube player or frame image */}
          {isYT ? (
            <YouTubeEmbed videoId={hit.video_id} startSeconds={hit.start} />
          ) : hasFrame ? (
            <img
              src={api.frameUrl(hit.frame!)}
              alt="frame"
              className="w-full rounded-lg border border-white/7"
              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
            />
          ) : null}

          {/* Transcript */}
          {hit.text && (
            <div className="rounded-lg bg-vs-surface-2 p-3 border border-white/7">
              <p className="text-xs text-vs-muted mb-1">Transcript</p>
              <p className="text-sm text-vs-text leading-relaxed">"{hit.text}"</p>
            </div>
          )}

          {/* Caption */}
          {hit.caption && (
            <div className="rounded-lg bg-vs-surface-2 p-3 border border-white/7">
              <p className="text-xs text-vs-muted mb-1">Scene description</p>
              <p className="text-sm text-vs-text leading-relaxed italic">{hit.caption}</p>
            </div>
          )}

          {/* Objects */}
          {hit.objects && hit.objects.length > 0 && (
            <div>
              <p className="text-xs text-vs-muted mb-2">Detected objects</p>
              <div className="flex flex-wrap gap-1.5">
                {hit.objects.map((obj) => (
                  <Badge key={obj} variant="secondary">{obj}</Badge>
                ))}
              </div>
            </div>
          )}

          <div className="flex gap-2 pt-1">
            {isYT && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => window.open(youtubeUrl(hit.video_id, hit.start), '_blank')}
              >
                <ExternalLink className="h-3.5 w-3.5" />
                Open in YouTube
              </Button>
            )}
            <Button variant="outline" size="md" className="ml-auto" onClick={() => setPlayerOpen(false)}>
              Close
            </Button>
          </div>
        </div>
      </Dialog>
    </>
  )
}
