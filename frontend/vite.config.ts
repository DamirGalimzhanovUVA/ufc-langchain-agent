import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat': process.env.VITE_API_TARGET ?? 'http://localhost:8000',
    },
  },
})
