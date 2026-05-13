import { NavLink } from 'react-router-dom'
import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

function NavTab({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      end={to === '/'}
      className={({ isActive }) =>
        cn(
          'relative inline-flex items-center h-9 px-3 text-sm font-medium rounded-md transition-colors duration-100',
          isActive ? 'text-fg' : 'text-muted hover:text-fg',
        )
      }
    >
      {({ isActive }) => (
        <>
          {label}
          {isActive && (
            <span className="absolute inset-x-2 -bottom-[13px] h-px bg-fg" aria-hidden="true" />
          )}
        </>
      )}
    </NavLink>
  )
}

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="flex flex-col w-full min-h-svh bg-bg">
      {/* Top bar */}
      <header className="sticky top-0 z-30 border-b border-border bg-bg/85 backdrop-blur">
        <div className="mx-auto flex h-12 max-w-[1400px] items-center gap-6 px-5">
          {/* Brand */}
          <NavLink to="/" end className="flex items-center gap-2 mr-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-[5px] bg-accent/90">
              <div className="h-2 w-2 rounded-[1px] bg-white" />
            </div>
            <span className="text-sm font-semibold text-fg tracking-tight">VideoSearch</span>
          </NavLink>

          {/* Tabs */}
          <nav className="flex items-center gap-1">
            <NavTab to="/" label="Search" />
            <NavTab to="/library" label="Library" />
          </nav>

          <div className="flex-1" />

        </div>
      </header>

      {/* Content */}
      <main className="flex-1 w-full">
        <div className="mx-auto w-full max-w-[1400px] px-5">{children}</div>
      </main>
    </div>
  )
}
