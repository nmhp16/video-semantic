import { useState, useEffect, useCallback } from 'react'
import { Plus, RefreshCw } from 'lucide-react'
import { VideoLibrary } from '@/components/VideoLibrary'
import { IngestModal } from '@/components/IngestModal'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import type { VideoMeta } from '@/lib/api'
import { useNavigate } from 'react-router-dom'

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
      // backend might not be running
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchVideos()
  }, [fetchVideos])

  const handleSelectVideo = (videoId: string) => {
    navigate(`/?video=${encodeURIComponent(videoId)}`)
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/7 flex-shrink-0">
        <div>
          <h1 className="text-lg font-semibold text-vs-text">Library</h1>
          <p className="text-xs text-vs-muted">
            {loading ? 'Loading…' : `${videos.length} video${videos.length !== 1 ? 's' : ''} indexed`}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={fetchVideos} disabled={loading}>
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button variant="default" size="sm" onClick={() => setIngestOpen(true)}>
            <Plus className="h-4 w-4" />
            Add Video
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-5">
        <VideoLibrary
          videos={videos}
          loading={loading}
          onSelect={handleSelectVideo}
        />
      </div>

      <IngestModal
        open={ingestOpen}
        onOpenChange={setIngestOpen}
        onSuccess={() => fetchVideos()}
      />
    </div>
  )
}
