// frontend/src/components/IngestModal.tsx
import { useState, useEffect, useRef } from 'react'
import { CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { Dialog } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import type { JobStatusResponse } from '@/lib/api'

interface IngestModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSuccess: () => void
}

const STAGES = [
  'Downloading & extracting frames…',
  'Transcribing audio…',
  'Building search index…',
]

function stageIndex(stage: string): number {
  return STAGES.findIndex((s) => s === stage)
}

export function IngestModal({ open, onOpenChange, onSuccess }: IngestModalProps) {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null)
  const [alreadyExists, setAlreadyExists] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  useEffect(() => {
    return () => stopPolling()
  }, [])

  const handleClose = (val: boolean) => {
    if (loading) return
    stopPolling()
    onOpenChange(val)
    if (!val) {
      setUrl('')
      setJobStatus(null)
      setAlreadyExists(false)
      setError(null)
    }
  }

  const handleIngest = async () => {
    if (!url.trim()) return
    setLoading(true)
    setError(null)
    setJobStatus(null)
    setAlreadyExists(false)
    try {
      const res = await api.ingest(url.trim())
      if (res.status === 'already_exists') {
        setAlreadyExists(true)
        setLoading(false)
        onSuccess()
        return
      }
      if (!res.job_id) {
        setError('No job ID returned')
        setLoading(false)
        return
      }
      const jobId = res.job_id
      pollRef.current = setInterval(async () => {
        try {
          const status = await api.ingestStatus(jobId)
          setJobStatus(status)
          if (status.status === 'done') {
            stopPolling()
            setLoading(false)
            onSuccess()
          } else if (status.status === 'error') {
            stopPolling()
            setError(status.error ?? 'Ingestion failed')
            setLoading(false)
          }
        } catch (e) {
          stopPolling()
          setError(e instanceof Error ? e.message : 'Polling failed')
          setLoading(false)
        }
      }, 2000)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ingestion failed')
      setLoading(false)
    }
  }

  const done = jobStatus?.status === 'done'
  const currentStageIdx = jobStatus ? stageIndex(jobStatus.stage) : -1

  return (
    <Dialog open={open} onOpenChange={handleClose} title="Add video"
            description="Paste a YouTube URL to download, transcribe, and index.">
      <div className="space-y-4">
        <div className="space-y-1.5">
          <label className="text-xxs font-medium uppercase tracking-wide text-subtle">
            Video URL
          </label>
          <Input
            placeholder="https://www.youtube.com/watch?v=…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !loading && handleIngest()}
            disabled={loading}
            autoFocus
          />
        </div>

        {loading && jobStatus && (
          <div className="space-y-2 rounded-md border border-accent/30 bg-accent-soft px-3 py-2.5">
            {STAGES.map((stage, i) => {
              const completed = currentStageIdx > i || done
              const active = currentStageIdx === i && !done
              return (
                <div key={stage} className="flex items-center gap-2 text-xs">
                  {completed ? (
                    <CheckCircle className="h-3.5 w-3.5 text-emerald-400 flex-shrink-0" />
                  ) : active ? (
                    <Loader2 className="h-3.5 w-3.5 text-accent animate-spin flex-shrink-0" />
                  ) : (
                    <span className="h-3.5 w-3.5 rounded-full border border-border flex-shrink-0" />
                  )}
                  <span className={active ? 'text-accent font-medium' : completed ? 'text-muted line-through' : 'text-dim'}>
                    {stage}
                  </span>
                </div>
              )
            })}
          </div>
        )}

        {loading && !jobStatus && (
          <div className="flex items-center gap-3 rounded-md border border-accent/30 bg-accent-soft px-3 py-2.5">
            <Loader2 className="h-4 w-4 animate-spin text-accent" />
            <p className="text-xs font-medium text-accent">Starting…</p>
          </div>
        )}

        {alreadyExists && (
          <div className="flex items-start gap-2.5 rounded-md border border-emerald-500/20 bg-emerald-500/10 px-3 py-2.5">
            <CheckCircle className="h-4 w-4 mt-0.5 flex-shrink-0 text-emerald-400" />
            <p className="text-xs font-medium text-emerald-400">Already indexed</p>
          </div>
        )}

        {done && (
          <div className="flex items-start gap-2.5 rounded-md border border-emerald-500/20 bg-emerald-500/10 px-3 py-2.5">
            <CheckCircle className="h-4 w-4 mt-0.5 flex-shrink-0 text-emerald-400" />
            <div className="text-xs">
              <p className="font-medium text-emerald-400">Done</p>
              <p className="text-muted">
                Indexed as <span className="font-mono text-fg">{jobStatus?.video_id}</span>
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
          {done || alreadyExists ? (
            <Button variant="primary" size="md" className="flex-1" onClick={() => handleClose(false)}>
              Done
            </Button>
          ) : (
            <Button variant="primary" size="md" className="flex-1"
                    onClick={handleIngest} disabled={loading || !url.trim()}>
              {loading ? 'Indexing…' : 'Index video'}
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
