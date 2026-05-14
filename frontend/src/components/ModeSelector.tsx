import { cn } from '@/lib/utils'
import type { SearchMode } from '@/lib/api'

const MODES: { id: SearchMode; label: string; hint: string }[] = [
  { id: 'auto', label: 'Auto', hint: 'Fuses visual + action — best for most queries' },
  { id: 'text', label: 'Text', hint: 'Search spoken words and transcripts' },
  { id: 'visual', label: 'Visual', hint: 'Search by scene, object, or appearance' },
  { id: 'action', label: 'Action', hint: 'Search by activity or motion' },
]

interface ModeSelectorProps {
  value: SearchMode
  onChange: (mode: SearchMode) => void
}

export function ModeSelector({ value, onChange }: ModeSelectorProps) {
  return (
    <div className="inline-flex items-center rounded-lg border border-border bg-surface p-0.5">
      {MODES.map((mode) => {
        const active = value === mode.id
        return (
          <button
            key={mode.id}
            onClick={() => onChange(mode.id)}
            title={mode.hint}
            className={cn(
              'relative h-7 px-3 text-xs font-medium rounded-md transition-colors duration-100',
              active
                ? 'bg-surface2 text-fg shadow-[inset_0_0_0_1px_#3F3F46]'
                : 'text-muted hover:text-fg',
            )}
          >
            {mode.label}
          </button>
        )
      })}
    </div>
  )
}
