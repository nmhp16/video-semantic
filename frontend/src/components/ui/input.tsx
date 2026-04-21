import { cn } from '@/lib/utils'
import type { InputHTMLAttributes } from 'react'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  className?: string
}

export function Input({ className, ...props }: InputProps) {
  return (
    <input
      className={cn(
        'w-full rounded-lg border border-white/8 bg-vs-surface-2 px-3 py-2 text-sm text-vs-text placeholder:text-vs-muted transition-colors',
        'focus:outline-none focus:border-vs-accent/50 focus:ring-1 focus:ring-vs-accent/20',
        'disabled:opacity-40 disabled:cursor-not-allowed',
        className
      )}
      {...props}
    />
  )
}
