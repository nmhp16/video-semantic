import { cn } from '@/lib/utils'
import type { SearchMode } from '@/lib/api'

const MODES: { id: SearchMode; label: string; description: string }[] = [
  { id: 'text',   label: 'Text',   description: 'What was said' },
  { id: 'visual', label: 'Visual', description: 'What was seen' },
  { id: 'action', label: 'Action', description: 'What happened' },
]

interface ModeSelectorProps {
  value: SearchMode
  onChange: (mode: SearchMode) => void
}

export function ModeSelector({ value, onChange }: ModeSelectorProps) {
  return (
    <div className="flex gap-1.5">
      {MODES.map((mode) => (
        <button
          key={mode.id}
          onClick={() => onChange(mode.id)}
          className={cn(
            'flex-1 px-3 py-2 rounded-lg border text-left transition-all duration-150',
            value === mode.id
              ? 'bg-vs-accent/15 border-vs-accent/40 text-vs-accent-light shadow-glow-sm'
              : 'border-white/8 bg-vs-surface-2 text-vs-muted hover:bg-white/5 hover:text-vs-text'
          )}
        >
          <p className={cn('text-xs font-semibold', value === mode.id ? 'text-vs-accent-light' : 'text-vs-text')}>{mode.label}</p>
          <p className="text-[10px] mt-0.5 text-vs-muted leading-tight">{mode.description}</p>
        </button>
      ))}
    </div>
  )
}
