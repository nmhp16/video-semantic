// frontend/src/components/IngestModal.tsx
import { useState, useEffect, useRef, useCallback } from 'react'
import { CheckCircle, AlertCircle, Loader2, Upload, Link, X } from 'lucide-react'
import { Dialog } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import type { JobStatusResponse, IngestJobResponse } from '@/lib/api'

interface IngestModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSuccess: () => void
}

type Mode = 'url' | 'file'

const URL_STAGES = [
  'Downloading & extracting frames…',
  'Transcribing audio…',
  'Building search index…',
]

const FILE_STAGES = [
  'Extracting frames…',
  'Transcribing audio…',
  'Building search index…',
]

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function IngestModal({ open, onOpenChange, onSuccess }: IngestModalProps) {
  const [mode, setMode] = useState<Mode>('url')
  const [url, setUrl] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null)
  const [alreadyExists, setAlreadyExists] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const stages = mode === 'file' ? FILE_STAGES : URL_STAGES

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  useEffect(() => {
    return () => stopPolling()
  }, [])

  const reset = () => {
    setUrl('')
    setFile(null)
    setUploadProgress(null)
    setJobStatus(null)
    setAlreadyExists(false)
    setError(null)
    setIsDragging(false)
  }

  const handleClose = (val: boolean) => {
    if (loading) return
    stopPolling()
    onOpenChange(val)
    if (!val) reset()
  }

  const switchMode = (m: Mode) => {
    if (loading) return
    setMode(m)
    reset()
  }

  const startPolling = (jobId: string) => {
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
  }

  const handleJobResponse = (res: IngestJobResponse) => {
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
    startPolling(res.job_id)
  }

  const handleIngest = async () => {
    setLoading(true)
    setError(null)
    setJobStatus(null)
    setAlreadyExists(false)
    setUploadProgress(null)
    try {
      if (mode === 'url') {
        if (!url.trim()) { setLoading(false); return }
        const res = await api.ingest(url.trim())
        handleJobResponse(res)
      } else {
        if (!file) { setLoading(false); return }
        const res = await api.ingestFile(file, (pct) => setUploadProgress(pct))
        setUploadProgress(null)
        handleJobResponse(res)
      }
    } catch (e) {
      setUploadProgress(null)
      setError(e instanceof Error ? e.message : 'Ingestion failed')
      setLoading(false)
    }
  }

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) {
      setFile(f)
      setError(null)
    }
  }, [])

  const done = jobStatus?.status === 'done'
  const currentStageIdx = jobStatus ? stages.findIndex((s) => s === jobStatus.stage) : -1
  const canSubmit = mode === 'url' ? !!url.trim() : !!file

  return (
    <Dialog
      open={open}
      onOpenChange={handleClose}
      title="Add video"
      description="Index a video from a URL or upload a local file."
    >
      <div className="space-y-4">
        {/* Mode tabs */}
        <div className="flex rounded-md border border-border overflow-hidden">
          {(['url', 'file'] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => switchMode(m)}
              disabled={loading}
              className={[
                'flex-1 flex items-center justify-center gap-1.5 py-1.5 text-xs font-medium transition-colors',
                mode === m
                  ? 'bg-accent/20 text-accent'
                  : 'text-muted hover:text-fg hover:bg-white/5 disabled:opacity-50',
              ].join(' ')}
            >
              {m === 'url'
                ? <Link className="h-3.5 w-3.5" />
                : <Upload className="h-3.5 w-3.5" />}
              {m === 'url' ? 'URL' : 'File'}
            </button>
          ))}
        </div>

        {/* URL input */}
        {mode === 'url' && (
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
        )}

        {/* File drop zone */}
        {mode === 'file' && !loading && !done && !alreadyExists && (
          <>
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.mov,.mkv,.avi,.webm,.m4v,.flv"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) { setFile(f); setError(null) }
              }}
            />
            {file ? (
              <div className="flex items-center gap-2.5 rounded-md border border-border bg-white/5 px-3 py-2.5">
                <Upload className="h-4 w-4 text-accent flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-fg truncate">{file.name}</p>
                  <p className="text-xxs text-muted">{formatBytes(file.size)}</p>
                </div>
                <button
                  onClick={() => setFile(null)}
                  className="text-muted hover:text-fg transition-colors p-0.5"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ) : (
              <div
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
                onDragLeave={(e) => { e.preventDefault(); setIsDragging(false) }}
                onClick={() => fileInputRef.current?.click()}
                className={[
                  'flex flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed px-4 py-8 cursor-pointer transition-colors select-none',
                  isDragging
                    ? 'border-accent bg-accent/10 text-accent'
                    : 'border-border hover:border-accent/50 hover:bg-white/5 text-muted',
                ].join(' ')}
              >
                <Upload className="h-6 w-6" />
                <div className="text-center">
                  <p className="text-xs font-medium">Drop video here or click to browse</p>
                  <p className="text-xxs text-dim mt-0.5">MP4 · MOV · MKV · AVI · WebM · M4V · FLV</p>
                </div>
              </div>
            )}
          </>
        )}

        {/* Upload progress bar */}
        {uploadProgress !== null && (
          <div className="space-y-1">
            <div className="flex justify-between text-xxs text-muted">
              <span>Uploading…</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="h-1 w-full rounded-full bg-white/10 overflow-hidden">
              <div
                className="h-full bg-accent transition-all duration-150 ease-out"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Stage progress */}
        {loading && jobStatus && (
          <div className="space-y-2 rounded-md border border-accent/30 bg-accent-soft px-3 py-2.5">
            {stages.map((stage, i) => {
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
                  <span className={
                    active ? 'text-accent font-medium'
                    : completed ? 'text-muted line-through'
                    : 'text-dim'
                  }>
                    {stage}
                  </span>
                </div>
              )
            })}
          </div>
        )}

        {loading && !jobStatus && uploadProgress === null && (
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
            <Button
              variant="primary"
              size="md"
              className="flex-1"
              onClick={handleIngest}
              disabled={loading || !canSubmit}
            >
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
