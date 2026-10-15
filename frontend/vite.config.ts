import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    // Pinned to IPv4. Vite's default host is `localhost`, which resolves to `::1` here, so
    // the dev server bound an IPv6-only socket: a browser was fine and every scripted probe
    // pointed at 127.0.0.1 was refused. `localhost` still resolves to this address, so
    // nothing that worked before stops working.
    host: "127.0.0.1",
    // Proxying /api keeps the browser same-origin in development, so the app
    // behaves identically whether the backend is reached through this proxy or
    // directly. It also means cookies/CORS never become load-bearing.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        // ag-Grid and lightweight-charts are large and change far less often than app
        // code; splitting them keeps the app chunk small across rebuilds.
        //
        // Plotly returns here now that D1 Regime Detection imports it, which is what the
        // note below was waiting for. Named as its own chunk *and* reached through a lazy
        // route: a manual chunk on its own would still be fetched on first paint, because
        // a static import is a static import however the file is laid out. The two
        // together mean a user who never opens the regime page never downloads it.
        manualChunks: {
          grid: ["ag-grid-community", "ag-grid-react"],
          charts: ["lightweight-charts"],
          plotly: ["plotly.js-dist-min"],
        },
      },
    },
  },
});
