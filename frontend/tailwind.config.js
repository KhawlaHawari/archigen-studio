/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        stencil: ['"Saira Stencil One"', '"Instrument Serif"', 'sans-serif'],
        serif: ['"Instrument Serif"', 'Georgia', 'serif'],
        mono: ['"Geist Mono"', 'ui-monospace', 'monospace'],
        sans: ['"Geist"', '-apple-system', 'system-ui', 'sans-serif'],
      },
      colors: {
        // Cinematic palette: night surfaces, warm light text, amber accent.
        // (Names kept from the original theme so existing classes keep working.)
        parchment: {
          DEFAULT: '#0b0e12',
          light: '#11151b',
          card: '#161b22',
          dark: '#1f252e',
        },
        ink: {
          DEFAULT: '#ece6da',
          muted: 'rgba(236, 230, 218, 0.65)',
          subtle: 'rgba(236, 230, 218, 0.40)',
          faint: 'rgba(236, 230, 218, 0.12)',
        },
        terracotta: {
          DEFAULT: '#d4943a',
          vivid: '#e8a33d',
          hover: '#a8711f',
          glow: 'rgba(232, 163, 61, 0.25)',
        },
        teal: {
          cine: '#2a7a6e',
        },
        brand: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          900: '#0c4a6e',
        }
      }
    },
  },
  plugins: [],
}
