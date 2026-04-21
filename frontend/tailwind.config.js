/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'vs-bg': '#08080C',
        'vs-surface': '#0F0F17',
        'vs-surface-2': '#161622',
        'vs-surface-3': '#1C1C2E',
        'vs-border': 'rgba(255,255,255,0.07)',
        'vs-border-hover': 'rgba(255,255,255,0.12)',
        'vs-accent': '#7C3AED',
        'vs-accent-light': '#A78BFA',
        'vs-accent-muted': 'rgba(124,58,237,0.15)',
        'vs-text': '#F1F5F9',
        'vs-muted': '#64748B',
        'vs-subtle': '#334155',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        card: '12px',
        pill: '9999px',
      },
      animation: {
        'fade-in': 'fadeIn 0.2s ease-out',
        'slide-up': 'slideUp 0.25s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 1.5s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      backgroundImage: {
        'gradient-accent': 'linear-gradient(135deg, #7C3AED, #A78BFA)',
        'gradient-card': 'linear-gradient(145deg, #0F0F17, #161622)',
        'shimmer-gradient': 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.04) 50%, transparent 100%)',
      },
      boxShadow: {
        'glow': '0 0 20px rgba(124, 58, 237, 0.25)',
        'glow-sm': '0 0 10px rgba(124, 58, 237, 0.15)',
        'card': '0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.6)',
        'card-hover': '0 4px 12px rgba(0,0,0,0.5)',
      },
    },
  },
  plugins: [],
}
