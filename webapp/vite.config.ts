import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  assetsInclude: ["**/*.wgsl", "**/*.glsl", "**/*.vert", "**/*.frag", "**/*.bmind"],
  build: {
    target: "es2022",
    sourcemap: true,
  },
  worker: {
    format: "es",
  },
});
