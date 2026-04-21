import { NavLink } from 'react-router-dom'
import { Search, Film, Zap } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

interface NavItemProps {
  to: string
  icon: ReactNode
  label: string
}

function NavItem({ to, icon, label }: NavItemProps) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        cn(
          'group flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150',
          isActive
            ? 'bg-vs-accent/15 text-vs-accent-light border border-vs-accent/20'
            : 'text-vs-muted hover:text-vs-text hover:bg-white/5'
        )
      }
    >
      <span className="flex-shrink-0">{icon}</span>
      <span>{label}</span>
    </NavLink>
  )
}

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="flex h-screen w-full overflow-hidden bg-vs-bg">
      {/* Sidebar */}
      <aside className="flex w-60 flex-shrink-0 flex-col border-r border-white/7 bg-vs-surface/40 backdrop-blur-sm">
        {/* Logo */}
        <div className="flex items-center gap-2.5 px-4 py-5 border-b border-white/7">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-accent shadow-glow-sm">
            <Zap className="h-4 w-4 text-white" fill="white" />
          </div>
          <div>
            <span className="text-sm font-semibold text-vs-text">VideoSearch</span>
            <p className="text-xs text-vs-muted">Semantic AI</p>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          <NavItem to="/" icon={<Search className="h-4 w-4" />} label="Search" />
          <NavItem to="/library" icon={<Film className="h-4 w-4" />} label="Library" />
        </nav>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-white/7">
          <p className="text-xs text-vs-subtle">4 search modes</p>
          <p className="text-xs text-vs-subtle">Text · Visual · Action · Chain</p>
        </div>
      </aside>

      {/* Main */}
      <main className="flex flex-1 flex-col overflow-hidden">
        {children}
      </main>
    </div>
  )
}
