import { cn } from '@/lib/utils'
import type { SearchMode } from '@/lib/api'

const MODES: { id: SearchMode; label: string; description: string }[] = [
  { id: 'text', label: 'Text', description: 'Search transcripts semantically' },
  { id: 'visual', label: 'Visual', description: 'Search by visual content & scenes' },
  { id: 'action', label: 'Action', description: 'Find activities and events' },
  { id: 'action_chain', label: 'Chain', description: 'Find ordered action sequences' },
]

interface ModeSelectorProps {
  value: SearchMode
  onChange: (mode: SearchMode) => void
}

export function ModeSelector({ value, onChange }: ModeSelectorProps) {
  return (
    <div className="flex gap-1 p-1 rounded-lg bg-vs-surface-2 border border-white/7">
      {MODES.map((mode) => (
        <button
          key={mode.id}
          onClick={() => onChange(mode.id)}
          title={mode.description}
          className={cn(
            'relative flex-1 px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-150',
            value === mode.id
              ? 'bg-vs-accent text-white shadow-glow-sm'
              : 'text-vs-muted hover:text-vs-text hover:bg-white/5'
          )}
        >
          {mode.label}
        </button>
      ))}
    </div>
  )
}
