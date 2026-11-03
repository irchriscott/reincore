import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Tauri dev server: Vite runs on 1420 (matches tauri.conf.json devUrl),
// and the React app talks to the in-process daemon at 127.0.0.1:8989.
// In dev, the daemon is the one started by `npm run dev` (via
// beforeDevCommand) - actually the Tauri Rust process starts it. The proxy
// below is a fallback so a plain `vite` (outside Tauri) still works.
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    proxy: {
      '/v1': {
        target: 'http://127.0.0.1:8989',
        changeOrigin: true,
      },
      '/v1beta': {
        target: 'http://127.0.0.1:8989',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8989',
        changeOrigin: true,
      },
    },
  },
})