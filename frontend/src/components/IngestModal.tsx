import { useState } from 'react'
import { CheckCircle, AlertCircle, ArrowRight } from 'lucide-react'
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

export function IngestModal({ open, onOpenChange, onSuccess }: IngestModalProps) {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<IngestResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleClose = (val: boolean) => {
    if (!loading) {
      onOpenChange(val)
      if (!val) {
        setUrl('')
        setResult(null)
        setError(null)
      }
    }
  }

  const handleIngest = async () => {
    if (!url.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.ingest(url.trim())
      setResult(res)
      onSuccess(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ingestion failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={handleClose}
      title="Add video"
      description="Paste a YouTube URL to ingest and index."
    >
      <div className="space-y-4">
        <div className="space-y-1.5">
          <label className="text-xxs font-medium uppercase tracking-wide text-subtle">
            Video URL
          </label>
          <Input
            placeholder="https://www.youtube.com/watch?v=…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleIngest()}
            disabled={loading}
            autoFocus
          />
        </div>

        <ul className="space-y-1 text-xs text-muted">
          <li className="flex items-start gap-2">
            <ArrowRight className="h-3 w-3 mt-0.5 flex-shrink-0 text-subtle" />
            Whisper transcription, scene-change frame sampling, and SigLIP indexing.
          </li>
          <li className="flex items-start gap-2">
            <ArrowRight className="h-3 w-3 mt-0.5 flex-shrink-0 text-subtle" />
            Captions are generated lazily on the first visual or action query.
          </li>
        </ul>

        {loading && (
          <div className="flex items-center gap-3 rounded-md border border-accent/30 bg-accent-soft px-3 py-2.5">
            <Spinner size="sm" className="text-accent" />
            <div className="text-xs">
              <p className="font-medium text-accent">Processing…</p>
              <p className="text-muted">This may take a minute or two.</p>
            </div>
          </div>
        )}

        {result && (
          <div className="flex items-start gap-2.5 rounded-md border border-emerald-500/20 bg-emerald-500/10 px-3 py-2.5">
            <CheckCircle className="h-4 w-4 mt-0.5 flex-shrink-0 text-emerald-400" />
            <div className="text-xs">
              <p className="font-medium text-emerald-400">
                {result.status === 'already_exists' ? 'Already indexed' : 'Done'}
              </p>
              <p className="text-muted">
                Video ID <span className="font-mono text-fg">{result.video_id}</span>
              </p>
            </div>
          </div>
        )}

        {error && (
          <div className="flex items-start gap-2.5 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2.5">
            <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0 text-red-400" />
            <div className="text-xs">
              <p className="font-medium text-red-400">Failed</p>
              <p className="text-muted break-all">{error}</p>
            </div>
          </div>
        )}

        <div className="flex gap-2 pt-1">
          {result ? (
            <Button variant="primary" size="md" className="flex-1" onClick={() => handleClose(false)}>
              Done
            </Button>
          ) : (
            <Button
              variant="primary"
              size="md"
              className="flex-1"
              onClick={handleIngest}
              disabled={loading || !url.trim()}
            >
              {loading ? <Spinner size="sm" /> : null}
              {loading ? 'Ingesting' : 'Ingest'}
            </Button>
          )}
          <Button variant="secondary" size="md" onClick={() => handleClose(false)} disabled={loading}>
            Cancel
          </Button>
        </div>
      </div>
    </Dialog>
  )
}
