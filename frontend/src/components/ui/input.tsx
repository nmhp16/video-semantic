import { cn } from '@/lib/utils'
import { forwardRef, type InputHTMLAttributes } from 'react'

type InputProps = InputHTMLAttributes<HTMLInputElement>

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { className, ...props },
  ref,
) {
  return (
    <input
      ref={ref}
      className={cn(
        'w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-fg placeholder:text-dim',
        'transition-colors duration-100',
        'hover:border-border-strong',
        'focus:outline-none focus:border-accent/50 focus:ring-2 focus:ring-accent-ring',
        'disabled:opacity-40 disabled:cursor-not-allowed',
        className,
      )}
      {...props}
    />
  )
})
