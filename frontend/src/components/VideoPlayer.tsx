import { useEffect, useRef, useState } from 'react'
import { api } from '@/lib/api'
import { cn } from '@/lib/utils'

interface VideoPlayerProps {
  videoId: string
  startSeconds: number
  endSeconds?: number
  className?: string
  autoPlay?: boolean
}

export function VideoPlayer({
  videoId,
  startSeconds,
  endSeconds,
  className,
  autoPlay = true,
}: VideoPlayerProps) {
  const ref = useRef<HTMLVideoElement>(null)
  const [failed, setFailed] = useState(false)
  const src = api.mediaUrl(videoId)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const seek = () => {
      try {
        el.currentTime = startSeconds
        if (autoPlay) el.play().catch(() => {})
      } catch {
        /* ignore */
      }
    }
    if (el.readyState >= 1) {
      seek()
    } else {
      el.addEventListener('loadedmetadata', seek, { once: true })
      return () => el.removeEventListener('loadedmetadata', seek)
    }
  }, [videoId, startSeconds, autoPlay])

  if (failed) {
    return (
      <div
        className={cn(
          'w-full aspect-video rounded-lg bg-surface border border-border flex flex-col items-center justify-center gap-1 text-center p-6',
          className,
        )}
      >
        <p className="text-sm font-medium text-fg">Media unavailable</p>
        <p className="text-xs text-muted">
          The backend does not serve a local file for{' '}
          <span className="font-mono text-fg">{videoId}</span>.
        </p>
      </div>
    )
  }

  return (
    <video
      ref={ref}
      src={`${src}#t=${startSeconds},${endSeconds ?? ''}`}
      controls
      playsInline
      onError={() => setFailed(true)}
      className={cn(
        'w-full aspect-video rounded-lg bg-black border border-border',
        className,
      )}
    />
  )
}
