import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

interface BadgeProps {
  children: ReactNode
  variant?: 'default' | 'secondary' | 'outline' | 'success' | 'warning'
  className?: string
}

export function Badge({ children, variant = 'default', className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium',
        variant === 'default' && 'bg-vs-accent-muted text-vs-accent-light border border-vs-accent/20',
        variant === 'secondary' && 'bg-white/5 text-vs-muted border border-white/8',
        variant === 'outline' && 'border border-white/10 text-vs-muted',
        variant === 'success' && 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20',
        variant === 'warning' && 'bg-amber-500/10 text-amber-400 border border-amber-500/20',
        className
      )}
    >
      {children}
    </span>
  )
}
