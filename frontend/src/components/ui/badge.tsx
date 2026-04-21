import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

type Variant = 'neutral' | 'accent' | 'success' | 'warning' | 'danger' | 'outline'

interface BadgeProps {
  children: ReactNode
  variant?: Variant
  className?: string
}

const variantClass: Record<Variant, string> = {
  neutral: 'bg-surface2 text-muted border border-border',
  accent: 'bg-accent-soft text-accent border border-accent/20',
  success: 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20',
  warning: 'bg-amber-500/10 text-amber-400 border border-amber-500/20',
  danger: 'bg-red-500/10 text-red-400 border border-red-500/20',
  outline: 'border border-border text-muted',
}

export function Badge({ children, variant = 'neutral', className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xxs font-medium whitespace-nowrap',
        variantClass[variant],
        className,
      )}
    >
      {children}
    </span>
  )
}
