// Root config for headless vitest. Keeps vitest from picking up vite.config.ts
// (root: "app"), which would re-root the test run inside the app shell.
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
