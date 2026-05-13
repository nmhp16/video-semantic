// frontend/src/components/ResultCard.tsx
import { useState } from 'react'
import { ExternalLink, Play } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Dialog } from '@/components/ui/dialog'
import { VideoPlayer } from '@/components/VideoPlayer'
import { api } from '@/lib/api'
import { formatTimeRange, isYouTubeId, youtubeUrl, truncate, cn } from '@/lib/utils'
import type { UnifiedSearchHit, ScoreRange } from '@/lib/api'

interface ResultCardProps {
  hit: UnifiedSearchHit
  index: number
  title?: string
  scoreRange: ScoreRange | null
}

function FrameThumb({ framePath }: { framePath: string }) {
  const [failed, setFailed] = useState(false)
  if (failed) {
    return (
      <div className="w-full aspect-video bg-surface2 flex items-center justify-center">
        <Play className="h-6 w-6 text-dim" />
      </div>
    )
  }
  return (
    <img src={api.frameUrl(framePath)} alt="" onError={() => setFailed(true)}
         loading="lazy" className="w-full aspect-video object-cover" />
  )
}

function normalizeScore(score: number, range: ScoreRange | null): number {
  if (!range) return Math.round(score * 100)
  const { min, max } = range
  if (max === min) return Math.round(score * 100)
  return Math.round(((score - min) / (max - min)) * 100)
}

function ScorePill({ pct }: { pct: number }) {
  const tone =
    pct >= 60 ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
    : pct >= 40 ? 'bg-accent-soft text-accent border-accent/20'
    : 'bg-surface2 text-muted border-border'
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 font-mono text-[10px] leading-none', tone)}>
      {pct}%
    </span>
  )
}

export function ResultCard({ hit, index, title, scoreRange }: ResultCardProps) {
  const [open, setOpen] = useState(false)
  const pct = normalizeScore(hit.score, scoreRange)
  const isYT = isYouTubeId(hit.video_id)
  const hasFrame = Boolean(hit.frame)
  const caption = hit.caption || ''
  const objects = hit.objects ?? []
  const hasCaption = Boolean(caption.trim())
  const hasText = Boolean(hit.text?.trim())
  const primaryText = hasText ? hit.text : hasCaption ? caption : null
  const displayName = title ?? hit.video_id

  return (
    <>
      <article
        className="group rounded-lg border border-border bg-panel overflow-hidden transition-colors duration-150 hover:border-border-strong animate-slide-up cursor-pointer"
        style={{ animationDelay: `${Math.min(index, 20) * 20}ms` }}
        onClick={() => setOpen(true)}
      >
        <div className="relative">
          {hasFrame ? <FrameThumb framePath={hit.frame!} /> : (
            <div className="w-full aspect-video bg-surface2 flex items-center justify-center">
              <span className="font-mono text-xs text-dim">{formatTimeRange(hit.start, hit.end)}</span>
            </div>
          )}
          <div className="absolute inset-0 flex items-end justify-between p-2 bg-gradient-to-t from-black/60 via-transparent to-transparent">
            <span className="inline-flex items-center gap-1 rounded-md bg-black/50 px-1.5 py-0.5 font-mono text-[10px] text-white backdrop-blur-sm">
              {formatTimeRange(hit.start, hit.end)}
            </span>
            <ScorePill pct={pct} />
          </div>
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center opacity-0 transition-opacity duration-150 group-hover:opacity-100">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/95 shadow-lg">
              <Play className="h-4 w-4 text-black translate-x-[1px]" fill="currentColor" />
            </div>
          </div>
        </div>

        <div className="p-3 space-y-1.5">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-fg truncate">{displayName}</span>
            {isYT && (
              <button onClick={(e) => { e.stopPropagation(); window.open(youtubeUrl(hit.video_id, hit.start), '_blank') }}
                      title="Open in YouTube"
                      className="flex-shrink-0 p-1 -m-1 text-subtle hover:text-fg transition-colors">
                <ExternalLink className="h-3 w-3" />
              </button>
            )}
          </div>
          {title && (
            <span className="font-mono text-[10px] text-dim truncate block">{hit.video_id}</span>
          )}
          {primaryText && (
            <p className={cn('text-sm leading-snug line-clamp-3', hasText ? 'text-fg' : 'text-fg/90')}>
              {hasText ? `"${truncate(primaryText, 180)}"` : truncate(primaryText, 180)}
            </p>
          )}
          {objects.length > 0 && (
            <div className="flex flex-wrap gap-1 pt-0.5">
              {objects.slice(0, 4).map((obj) => (
                <Badge key={obj} variant="neutral" className="text-[10px] px-1.5 py-0">{obj}</Badge>
              ))}
              {objects.length > 4 && (
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">+{objects.length - 4}</Badge>
              )}
            </div>
          )}
        </div>
      </article>

      <Dialog open={open} onOpenChange={setOpen} title={displayName}
              description={`${formatTimeRange(hit.start, hit.end)} · ${pct}% match`}
              className="w-[min(94vw,820px)]">
        <div className="space-y-4">
          {open && <VideoPlayer videoId={hit.video_id} startSeconds={hit.start} endSeconds={hit.end} />}
          {hasText && (
            <section>
              <h3 className="text-xxs font-medium uppercase tracking-wide text-subtle mb-1.5">Transcript</h3>
              <p className="rounded-md border border-border bg-surface/50 px-3 py-2 text-sm text-fg leading-relaxed">"{hit.text}"</p>
            </section>
          )}
          {hasCaption && (
            <section>
              <h3 className="text-xxs font-medium uppercase tracking-wide text-subtle mb-1.5">Caption</h3>
              <p className="rounded-md border border-border bg-surface/50 px-3 py-2 text-sm text-fg leading-relaxed">{caption}</p>
            </section>
          )}
          {objects.length > 0 && (
            <section>
              <h3 className="text-xxs font-medium uppercase tracking-wide text-subtle mb-1.5">Keywords</h3>
              <div className="flex flex-wrap gap-1.5">
                {objects.map((obj) => <Badge key={obj} variant="neutral">{obj}</Badge>)}
              </div>
            </section>
          )}
          <div className="flex gap-2 pt-1">
            {isYT && (
              <Button variant="secondary" size="md" className="flex-1"
                      onClick={() => window.open(youtubeUrl(hit.video_id, hit.start), '_blank')}>
                <ExternalLink className="h-4 w-4" />
                Open on YouTube at {formatTimeRange(hit.start, hit.end)}
              </Button>
            )}
            <Button variant="primary" size="md" onClick={() => setOpen(false)}>Close</Button>
          </div>
        </div>
      </Dialog>
    </>
  )
}
