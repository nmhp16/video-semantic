import { useState } from 'react'
import { ExternalLink, Clock, Play } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Dialog } from '@/components/ui/dialog'
import { api } from '@/lib/api'
import { formatTimeRange, scorePercent, isYouTubeId, youtubeUrl, truncate } from '@/lib/utils'
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

export function ResultCard({ hit, index }: ResultCardProps) {
  const [jumpOpen, setJumpOpen] = useState(false)
  const pct = scorePercent(hit.score)
  const isYT = isYouTubeId(hit.video_id)
  const hasFrame = Boolean(hit.frame)

  return (
    <>
      <div
        className="group glass-card overflow-hidden hover:border-white/12 hover:shadow-card-hover transition-all duration-200 animate-slide-up"
        style={{ animationDelay: `${index * 30}ms` }}
      >
        {/* Frame thumbnail */}
        {hasFrame ? (
          <FrameImage framePath={hit.frame!} />
        ) : (
          <div className="w-full aspect-video bg-vs-surface-3 flex items-center justify-center">
            <div className="text-center">
              <Clock className="h-6 w-6 text-vs-subtle mx-auto mb-1" />
              <span className="text-xs text-vs-subtle">{formatTimeRange(hit.start, hit.end)}</span>
            </div>
          </div>
        )}

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

          {/* Video ID */}
          <p className="text-xs text-vs-subtle font-mono truncate">{hit.video_id}</p>

          {/* Transcript text */}
          {hit.text && (
            <p className="text-xs text-vs-text leading-relaxed line-clamp-3">
              "{truncate(hit.text, 160)}"
            </p>
          )}

          {/* Object badges */}
          {hit.objects && hit.objects.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {hit.objects.slice(0, 5).map((obj) => (
                <Badge key={obj} variant="secondary" className="text-[10px] px-1.5 py-0">
                  {obj}
                </Badge>
              ))}
              {hit.objects.length > 5 && (
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                  +{hit.objects.length - 5}
                </Badge>
              )}
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2 pt-0.5">
            <Button
              variant="outline"
              size="sm"
              className="flex-1 text-xs h-7"
              onClick={() => setJumpOpen(true)}
            >
              <Play className="h-3 w-3" />
              Jump to
            </Button>
            {isYT && (
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={() => window.open(youtubeUrl(hit.video_id, hit.start), '_blank')}
                title="Open in YouTube"
              >
                <ExternalLink className="h-3 w-3" />
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Jump to dialog */}
      <Dialog
        open={jumpOpen}
        onOpenChange={setJumpOpen}
        title={`${hit.video_id}`}
        description={`${formatTimeRange(hit.start, hit.end)} · Score ${pct}%`}
        className="max-w-2xl"
      >
        <div className="space-y-4">
          {hasFrame && (
            <img
              src={api.frameUrl(hit.frame!)}
              alt="frame"
              className="w-full rounded-lg border border-white/7"
              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
            />
          )}

          {hit.text && (
            <div className="rounded-lg bg-vs-surface-2 p-3 border border-white/7">
              <p className="text-xs text-vs-muted mb-1">Transcript</p>
              <p className="text-sm text-vs-text leading-relaxed">"{hit.text}"</p>
            </div>
          )}

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

          <div className="flex gap-2 pt-2">
            {isYT && (
              <Button
                variant="default"
                size="md"
                className="flex-1"
                onClick={() => window.open(youtubeUrl(hit.video_id, hit.start), '_blank')}
              >
                <ExternalLink className="h-4 w-4" />
                Open in YouTube at {formatTimeRange(hit.start, hit.end)}
              </Button>
            )}
            <Button variant="outline" size="md" onClick={() => setJumpOpen(false)}>
              Close
            </Button>
          </div>
        </div>
      </Dialog>
    </>
  )
}
