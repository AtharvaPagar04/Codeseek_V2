/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Base backgrounds — pure dark monochrome
        base: '#0a0a0a',
        surface: '#111111',
        'surface-2': '#161616',
        'surface-3': '#1c1c1c',
        border: '#262626',
        'border-subtle': '#1e1e1e',

        // Text — neutral whites & grays
        'text-primary': '#f0f0f0',
        'text-secondary': '#a0a0a0',
        'text-muted': '#555555',

        // Accent — soft white/light for active states
        accent: '#ffffff',
        'accent-dim': '#d4d4d4',
        'accent-glow': 'rgba(255, 255, 255, 0.06)',

        // Status
        online: '#4ade80',
        offline: '#f87171',
        warning: '#fbbf24',

        // User message
        'user-bg': '#1a1a1a',
        'user-border': '#2a2a2a',
      },
      fontFamily: {
        sans: ['"Inter"', '"IBM Plex Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"IBM Plex Mono"', '"Fira Code"', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.65rem', { lineHeight: '1rem' }],
      },
      borderRadius: {
        'card': '16px',
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        fadeIn: {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        slideIn: {
          from: { transform: 'translateX(-100%)' },
          to: { transform: 'translateX(0)' },
        },
        pulse: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.3' },
        },
        dotBounce: {
          '0%, 80%, 100%': { transform: 'translateY(0)' },
          '40%': { transform: 'translateY(-6px)' },
        },
        slideDown: {
          from: { opacity: '0', maxHeight: '0', transform: 'translateY(-8px)' },
          to: { opacity: '1', maxHeight: '500px', transform: 'translateY(0)' },
        },
      },
      animation: {
        blink: 'blink 1s step-end infinite',
        fadeIn: 'fadeIn 0.2s ease-out',
        slideIn: 'slideIn 0.2s ease-out',
        'dot-1': 'dotBounce 1.2s ease-in-out infinite',
        'dot-2': 'dotBounce 1.2s ease-in-out 0.2s infinite',
        'dot-3': 'dotBounce 1.2s ease-in-out 0.4s infinite',
        slideDown: 'slideDown 0.2s ease-out forwards',
      },
    },
  },
  plugins: [],
};
