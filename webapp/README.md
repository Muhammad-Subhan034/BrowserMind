# webapp

The browser application. Static site, no backend, no framework — TypeScript + Vite + hand-written WGSL/GLSL. See the [top-level README](../README.md) for the full project story, architecture diagram, and measured numbers.

## Run

```bash
npm install
npm run dev
```

The bundled model at `public/models/minilm-sif.bmind` is checked in, so this is enough to try the full app — see [`../modelprep/README.md`](../modelprep/README.md) to rebuild it from scratch.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Vite dev server |
| `npm run build` | Typecheck + production build to `dist/` |
| `npm run typecheck` | `tsc -b --noEmit` |
| `npm run correctness-check` | Drives a real browser (via `playwright-core`, pointed at your system Chrome/Edge — no browser download) to compare live WGSL kernel output against the pure-JS reference implementation, printing a real cosine-similarity report |

## Source layout

```
src/gpu/
  device.ts          WebGPU capability detection (adapter info, features, limits)
  modelFormat.ts      .bmind parser — must stay byte-for-byte in sync with
                       modelprep/include/bmind_format.hpp
  pipeline.ts          buffer/bind-group orchestration, dispatch sequencing,
                       timestamp-query profiling for all 5 WGSL kernels
  profiler.ts          GPUQuerySet wrapper for real per-pass GPU timing
  shaders/
    embedding_lookup.wgsl   token id -> dequantized, SIF-weighted vector
    pooling.wgsl             weighted mean + top-PC removal (workgroup reduction)
    similarity.wgsl          batched cosine similarity, query vs every chunk
    topk.wgsl                 tournament-reduction GPU-side top-k
    project2d.wgsl             batched projection for the embedding map

src/ingest/
  tokenizer.ts        hand-written WordPiece tokenizer (matches modelprep's
                       Python-side tokenization exactly -- both sides must
                       agree on token ids)
  chunk.ts              overlapping token-window document chunking
  documents.ts           TXT/MD direct read, PDF via pdfjs-dist

src/cpu/fallback.ts   pure-JS reference implementation: same math as the WGSL
                       kernels, used both for correctness checking and as the
                       CPU side of the GPU-vs-CPU benchmark

src/viz/
  layout.ts            CPU-side power iteration to find the top-2 PCA axes
  cluster.ts             dependency-free k-means over the full embeddings
  scatter.ts              hand-written GLSL/WebGL2 point renderer

src/db/cache.ts       IndexedDB persistence, keyed by a SHA-256 content hash
                       so unchanged documents never re-run the GPU pipeline

src/bench/benchmark.ts  corpus-size sweep harness backing the in-app
                         "Benchmark" tab

src/ui/                vanilla TypeScript + CSS UI, View Transitions API for
                        phase changes, hand-rolled DOM helpers (no framework)
```

## A note on the UI architecture

There's no virtual DOM and no diffing. Phase transitions (landing → processing → search) fully rebuild the view through `document.startViewTransition()` when available. Within the search view, the shell is built **once** and kept alive — typing in the search box, toggling the benchmark switch, or opening the GPU Internals drawer all patch specific DOM nodes directly (see the `refs` object and `patch*()` methods in `src/ui/app.ts`) rather than re-rendering the tree, which is what keeps the search input's focus and cursor position stable while results update live underneath it.
