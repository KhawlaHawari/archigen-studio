import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173, // 3000 is often taken by other local apps / Docker containers
    strictPort: true,
    proxy: {
      '/predict': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/metrics': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/model': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/design': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/facade': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/furniture': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/models/custom': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
