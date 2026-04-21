/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Neutral surface scale (Tailwind zinc, referenced directly)
        bg: '#09090B',
        panel: '#111113',
        surface: '#18181B',
        surface2: '#27272A',
        border: '#27272A',
        'border-strong': '#3F3F46',
        fg: '#FAFAFA',
        muted: '#A1A1AA',
        subtle: '#71717A',
        dim: '#52525B',

        // Single accent — blue-500 on hover shifts to blue-400
        accent: '#3B82F6',
        'accent-hover': '#60A5FA',
        'accent-soft': 'rgba(59, 130, 246, 0.12)',
        'accent-ring': 'rgba(59, 130, 246, 0.28)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      fontSize: {
        xxs: ['11px', { lineHeight: '14px' }],
        xs: ['12px', { lineHeight: '16px' }],
        sm: ['13px', { lineHeight: '18px' }],
        base: ['14px', { lineHeight: '20px' }],
        lg: ['16px', { lineHeight: '22px' }],
        xl: ['18px', { lineHeight: '26px' }],
        '2xl': ['22px', { lineHeight: '28px' }],
      },
      borderRadius: {
        sm: '4px',
        DEFAULT: '6px',
        md: '8px',
        lg: '10px',
        xl: '12px',
      },
      animation: {
        'fade-in': 'fadeIn 160ms ease-out',
        'slide-up': 'slideUp 180ms cubic-bezier(0.16, 1, 0.3, 1)',
        shimmer: 'shimmer 1500ms linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
    },
  },
  plugins: [],
}
