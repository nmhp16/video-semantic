import { useState, useEffect, useCallback } from 'react'
import { Plus, RefreshCw } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { VideoLibrary } from '@/components/VideoLibrary'
import { IngestModal } from '@/components/IngestModal'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import type { VideoMeta } from '@/lib/api'

export function LibraryPage() {
  const [videos, setVideos] = useState<VideoMeta[]>([])
  const [loading, setLoading] = useState(true)
  const [ingestOpen, setIngestOpen] = useState(false)
  const navigate = useNavigate()

  const fetchVideos = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.getVideos()
      setVideos(res.videos)
    } catch {
      /* backend may not be running */
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchVideos()
  }, [fetchVideos])

  return (
    <div className="flex flex-col py-6 gap-5">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-fg tracking-tight">Library</h1>
          <p className="mt-0.5 text-xs text-muted">
            {loading ? 'Loading…' : `${videos.length} video${videos.length !== 1 ? 's' : ''} indexed`}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" onClick={fetchVideos} disabled={loading}>
            <RefreshCw className={loading ? 'h-3.5 w-3.5 animate-spin' : 'h-3.5 w-3.5'} />
            Refresh
          </Button>
          <Button variant="primary" size="sm" onClick={() => setIngestOpen(true)}>
            <Plus className="h-3.5 w-3.5" />
            Add video
          </Button>
        </div>
      </div>

      <VideoLibrary
        videos={videos}
        loading={loading}
        onSelect={(id) => navigate(`/?video=${encodeURIComponent(id)}`)}
      />

      <IngestModal
        open={ingestOpen}
        onOpenChange={setIngestOpen}
        onSuccess={() => fetchVideos()}
      />
    </div>
  )
}
