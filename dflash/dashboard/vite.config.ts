import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        ideas: resolve(__dirname, "ideas.html"),
        diversity: resolve(__dirname, "diversity.html"),
        benchmark: resolve(__dirname, "benchmark.html"),
      },
    },
  },
});
