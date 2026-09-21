/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    // Straight into the Go embed directory. emptyOutDir stays false so the
    // tracked dist/.gitkeep survives a build: //go:embed all:dist needs a
    // non-empty directory in a fresh clone. The Makefile does the stale-asset
    // cleanup instead.
    outDir: "../backend/web/dist",
    emptyOutDir: false,
  },
  server: {
    host: "127.0.0.1",
    proxy: { "/api": "http://127.0.0.1:8080" },
  },
  test: {
    environment: "jsdom",
    // Testing Library registers its automatic cleanup through the global
    // afterEach hook; without globals each render stacks on the previous
    // test's DOM and every findByText hits duplicates.
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text-summary", "json-summary", "lcov"],
      reportsDirectory: "../coverage/ui",
      // Explicit, so an untested module is not silently invisible to the
      // ratio the gate reads.
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/main.tsx", "src/**/*.d.ts"],
    },
  },
});
