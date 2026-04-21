import { cn } from '@/lib/utils'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'outline' | 'ghost' | 'destructive'
  size?: 'sm' | 'md' | 'lg' | 'icon'
  children: ReactNode
}

export function Button({ variant = 'default', size = 'md', className, children, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 font-medium rounded-lg transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-vs-accent/50 disabled:opacity-40 disabled:cursor-not-allowed',
        variant === 'default' && 'bg-vs-accent hover:bg-vs-accent/90 text-white shadow-glow-sm hover:shadow-glow',
        variant === 'outline' && 'border border-white/10 text-vs-text hover:bg-white/5 hover:border-white/20',
        variant === 'ghost' && 'text-vs-muted hover:text-vs-text hover:bg-white/5',
        variant === 'destructive' && 'bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-600/20',
        size === 'sm' && 'h-8 px-3 text-xs',
        size === 'md' && 'h-9 px-4 text-sm',
        size === 'lg' && 'h-11 px-6 text-base',
        size === 'icon' && 'h-9 w-9 p-0',
        className
      )}
      {...props}
    >
      {children}
    </button>
  )
}
