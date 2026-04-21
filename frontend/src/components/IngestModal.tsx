import { useState, useEffect, useRef } from 'react'
import { CheckCircle, AlertCircle, Plus, Loader2 } from 'lucide-react'
import { Dialog } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import { api } from '@/lib/api'
import type { IngestResponse } from '@/lib/api'

interface IngestModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSuccess: (res: IngestResponse) => void
}

const POLL_MS = 2500

export function IngestModal({ open, onOpenChange, onSuccess }: IngestModalProps) {
  const [url, setUrl] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [status, setStatus] = useState<IngestResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  useEffect(() => () => stopPolling(), [])

  const startPolling = (video_id: string) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const res = await api.getIngestStatus(video_id)
        if (res.status === 'completed') {
          stopPolling()
          setStatus(res)
          onSuccess(res)
        } else if (res.status === 'failed') {
          stopPolling()
          setError(res.error ?? 'Ingestion failed')
          setStatus(null)
        }
      } catch {
        // ignore transient poll errors
      }
    }, POLL_MS)
  }

  const handleClose = (val: boolean) => {
    if (submitting) return
    stopPolling()
    onOpenChange(val)
    if (!val) {
      setUrl('')
      setStatus(null)
      setError(null)
    }
  }

  const handleIngest = async () => {
    if (!url.trim()) return
    setSubmitting(true)
    setError(null)
    setStatus(null)
    try {
      const res = await api.ingest(url.trim())
      if (res.status === 'processing') {
        setStatus(res)
        startPolling(res.video_id)
      } else if (res.status === 'completed') {
        setStatus(res)
        onSuccess(res)
      } else {
        setError(res.error ?? 'Unexpected status')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ingestion failed')
    } finally {
      setSubmitting(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleIngest()
  }

  const isProcessing = status?.status === 'processing'
  const isDone = status?.status === 'completed'

  return (
    <Dialog
      open={open}
      onOpenChange={handleClose}
      title="Add Video"
      description="Paste a YouTube URL to index a new video."
    >
      <div className="space-y-4">
        <div className="space-y-2">
          <label className="text-xs font-medium text-vs-muted">YouTube URL</label>
          <Input
            placeholder="https://www.youtube.com/watch?v=..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={submitting || isProcessing || isDone}
            autoFocus
          />
        </div>

        {/* Processing — shows while background job runs */}
        {isProcessing && (
          <div className="flex items-center gap-3 rounded-lg bg-vs-accent-muted border border-vs-accent/20 px-4 py-3">
            <Loader2 className="h-4 w-4 text-vs-accent-light animate-spin flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-vs-accent-light">Processing video…</p>
              <p className="text-xs text-vs-muted mt-0.5">Downloading, captioning frames, transcribing audio. Checking every {POLL_MS / 1000}s.</p>
            </div>
          </div>
        )}

        {/* Done */}
        {isDone && (
          <div className="flex items-start gap-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 px-4 py-3">
            <CheckCircle className="h-4 w-4 text-emerald-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-emerald-400">
                Ready to search
              </p>
              <p className="text-xs text-vs-muted mt-0.5">
                Video ID: <span className="font-mono text-vs-text">{status?.video_id}</span>
              </p>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="flex items-start gap-3 rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3">
            <AlertCircle className="h-4 w-4 text-red-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-red-400">Failed to ingest</p>
              <p className="text-xs text-vs-muted mt-0.5">{error}</p>
            </div>
          </div>
        )}

        <div className="flex gap-2 pt-1">
          {isDone ? (
            <Button variant="default" size="md" className="flex-1" onClick={() => handleClose(false)}>
              Done
            </Button>
          ) : (
            <Button
              variant="default"
              size="md"
              className="flex-1"
              onClick={handleIngest}
              disabled={submitting || isProcessing || !url.trim()}
            >
              {submitting ? <Spinner size="sm" /> : <Plus className="h-4 w-4" />}
              {submitting ? 'Starting…' : isProcessing ? 'Processing…' : 'Ingest Video'}
            </Button>
          )}
          <Button variant="outline" size="md" onClick={() => handleClose(false)} disabled={submitting}>
            {isDone ? 'Close' : 'Cancel'}
          </Button>
        </div>
      </div>
    </Dialog>
  )
}
