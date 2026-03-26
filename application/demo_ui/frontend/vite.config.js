import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendOrigin = env.VITE_BACKEND_ORIGIN || "http://127.0.0.1:8000";
  return {
    base: command === "build" ? "/static/dist/" : "/",
    plugins: [react()],
    server: {
      port: 3000,
      proxy: {
        "/api": backendOrigin,
        "/ws": {
          target: backendOrigin.replace("http://", "ws://").replace("https://", "wss://"),
          ws: true,
        },
      },
    },
    build: {
      outDir: "../static/dist",
      emptyOutDir: true,
    },
  };
});
