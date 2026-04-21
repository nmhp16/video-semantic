import { useState } from 'react'
import { CheckCircle, AlertCircle, Plus } from 'lucide-react'
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

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleIngest()
  }

  return (
    <Dialog
      open={open}
      onOpenChange={handleClose}
      title="Add Video"
      description="Paste a YouTube URL to ingest and index a new video."
    >
      <div className="space-y-4">
        <div className="space-y-2">
          <label className="text-xs font-medium text-vs-muted">YouTube URL</label>
          <Input
            placeholder="https://www.youtube.com/watch?v=..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            autoFocus
          />
        </div>

        {/* Loading state */}
        {loading && (
          <div className="flex items-center gap-3 rounded-lg bg-vs-accent-muted border border-vs-accent/20 px-4 py-3">
            <Spinner size="sm" />
            <div>
              <p className="text-sm font-medium text-vs-accent-light">Processing video…</p>
              <p className="text-xs text-vs-muted mt-0.5">Downloading, transcribing, and indexing. This may take a few minutes.</p>
            </div>
          </div>
        )}

        {/* Success */}
        {result && (
          <div className="flex items-start gap-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 px-4 py-3">
            <CheckCircle className="h-4 w-4 text-emerald-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-emerald-400">
                {result.status === 'already_exists' ? 'Already indexed' : 'Ingestion complete'}
              </p>
              <p className="text-xs text-vs-muted mt-0.5">Video ID: <span className="font-mono text-vs-text">{result.video_id}</span></p>
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
          {result ? (
            <Button variant="default" size="md" className="flex-1" onClick={() => handleClose(false)}>
              Done
            </Button>
          ) : (
            <Button
              variant="default"
              size="md"
              className="flex-1"
              onClick={handleIngest}
              disabled={loading || !url.trim()}
            >
              {loading ? <Spinner size="sm" /> : <Plus className="h-4 w-4" />}
              {loading ? 'Ingesting…' : 'Ingest Video'}
            </Button>
          )}
          <Button variant="outline" size="md" onClick={() => handleClose(false)} disabled={loading}>
            Cancel
          </Button>
        </div>
      </div>
    </Dialog>
  )
}
