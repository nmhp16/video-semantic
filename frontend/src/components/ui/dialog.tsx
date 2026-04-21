import * as RadixDialog from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

interface DialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title?: string
  description?: string
  children: ReactNode
  className?: string
}

export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  className,
}: DialogProps) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className="fixed inset-0 z-50 bg-black/70 backdrop-blur-[2px] animate-fade-in" />
        <RadixDialog.Content
          className={cn(
            'fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2',
            'w-[min(92vw,520px)] rounded-xl border border-border bg-panel shadow-2xl animate-slide-up',
            className,
          )}
        >
          {(title || description) && (
            <div className="flex items-start justify-between gap-4 px-5 pt-5 pb-4 border-b border-border">
              <div className="min-w-0">
                {title && (
                  <RadixDialog.Title className="text-base font-semibold text-fg truncate">
                    {title}
                  </RadixDialog.Title>
                )}
                {description && (
                  <RadixDialog.Description className="mt-1 text-xs text-muted">
                    {description}
                  </RadixDialog.Description>
                )}
              </div>
              <RadixDialog.Close className="flex-shrink-0 rounded-md p-1 text-muted hover:text-fg hover:bg-surface2 transition-colors">
                <X className="h-4 w-4" />
              </RadixDialog.Close>
            </div>
          )}
          <div className="p-5">{children}</div>
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  )
}
