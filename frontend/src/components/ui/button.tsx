import { cn } from '@/lib/utils'
import { forwardRef, type ButtonHTMLAttributes } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost' | 'destructive'
type Size = 'xs' | 'sm' | 'md' | 'lg' | 'icon'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
}

const variantClass: Record<Variant, string> = {
  primary:
    'bg-accent text-white hover:bg-accent-hover active:bg-accent active:scale-[0.99]',
  secondary:
    'bg-surface2 text-fg hover:bg-[#32323A] border border-border hover:border-border-strong',
  ghost: 'text-muted hover:text-fg hover:bg-surface2',
  destructive:
    'bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20',
}

const sizeClass: Record<Size, string> = {
  xs: 'h-7 px-2 text-xs gap-1.5',
  sm: 'h-8 px-3 text-xs gap-1.5',
  md: 'h-9 px-4 text-sm gap-2',
  lg: 'h-10 px-5 text-sm gap-2',
  icon: 'h-9 w-9 p-0',
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    { variant = 'secondary', size = 'md', className, ...props },
    ref,
  ) {
    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center rounded-md font-medium transition-all duration-100',
          'disabled:opacity-40 disabled:cursor-not-allowed disabled:pointer-events-none',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-ring focus-visible:ring-offset-2 focus-visible:ring-offset-bg',
          variantClass[variant],
          sizeClass[size],
          className,
        )}
        {...props}
      />
    )
  },
)
