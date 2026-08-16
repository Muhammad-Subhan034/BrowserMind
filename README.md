# BrowserMind

**A WebGPU-native, zero-backend semantic search engine.** Every embedding, similarity score, and top-k ranking is computed by hand-written WGSL compute shaders running on your GPU, inside a single browser tab. Nothing is uploaded, there is no server, and there is no API key — the whole retrieval pipeline, from tokenization to ranked results, runs client-side.

A companion C++ tool (`modelprep`) prepares the embedding model offline: it quantizes a real pretrained WordPiece embedding table to INT8 with hand-written per-row quantization, validates the result numerically against an fp32 Python reference, and packs everything into a compact binary the browser loads directly into WebGPU storage buffers.

> Full original project brief: [Project_1_BrowserMind.md](Project_1_BrowserMind.md). This README documents what was actually built, including where the implementation deliberately diverged from that brief and why.

---

## Try it

```bash
cd webapp
npm install
npm run dev
# open the printed http://localhost:5173 URL in Chrome, Edge, or Firefox 141+
```

Click **"Try the demo corpus"** — no file upload needed — or drop in your own `.txt` / `.md` / `.pdf` files. Everything after that (chunking, GPU embedding, GPU search, ranking) runs locally.

The bundled model (`webapp/public/models/minilm-sif.bmind`, 11.7 MB) is already built and checked in, so `npm run dev` is enough to see the whole thing working. Rebuilding it from scratch is covered below.

---

## What it does

1. Detects WebGPU support and reports the actual adapter (vendor, architecture, feature set) — not just a yes/no.
2. Chunks uploaded documents client-side with a hand-written WordPiece tokenizer (the same subword algorithm BERT-family models were trained with).
3. Embeds every chunk on the GPU: an `embedding_lookup.wgsl` kernel dequantizes INT8 weights and gathers token vectors, a `pooling.wgsl` kernel SIF-weights and mean-pools them per chunk using a workgroup-shared-memory reduction.
4. On query, runs the same GPU pipeline for the query string, then `similarity.wgsl` (batched cosine similarity, one thread per chunk) and `topk.wgsl` (a tournament-reduction top-k selection) — scores never leave the GPU until the final *k* results do.
5. Shows ranked, highlighted results with real per-query latency, a live **GPU Internals** panel (per-kernel timing from actual WebGPU timestamp queries, buffer memory in use, dispatch counts), a **GPU vs CPU** benchmark toggle, and a **GLSL/WebGL2** embedding-space scatter plot (points laid out by a GPU-computed 2D projection, colored by k-means cluster).
6. Caches chunks + embeddings in IndexedDB, so reloading or re-adding the same document never re-runs the GPU pipeline for unchanged content.

---

## Architecture

```
                    ┌─────────────────────────────┐
  (offline, once)   │   modelprep/  (C++, CMake)   │
                     │                              │
 HuggingFace ──────► │  python/export_embeddings.py │  real pretrained WordPiece
 MiniLM-L6-v2        │   → fp32 embeddings, vocab,  │  embeddings + SIF weights
                     │     SIF weights, PCA vector  │
                     │              │               │
                     │              ▼               │
                     │  C++: quantize (INT8/row)     │  hand-written quantization,
                     │      → pack .bmind             │  not a library call
                     │      → validate vs fp32 ref    │  cosine sim, real numbers
                     └──────────────┬───────────────┘
                                    │  minilm-sif.bmind (11.7 MB)
                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                    webapp/  (TypeScript + Vite, static site)       │
│                                                                     │
│  ingest/tokenizer.ts   WordPiece tokenizer (client-side, no deps)  │
│  ingest/chunk.ts        overlapping token-window chunking          │
│                                                                     │
│  gpu/shaders/*.wgsl     embedding_lookup → pooling → similarity    │
│                              → topk → project2d   (5 WGSL kernels) │
│  gpu/pipeline.ts        buffers, bind groups, dispatch, profiling  │
│  cpu/fallback.ts        pure-JS reference impl (correctness + CPU  │
│                          side of the GPU-vs-CPU benchmark)         │
│                                                                     │
│  viz/scatter.ts         hand-written GLSL/WebGL2 renderer          │
│  db/cache.ts            IndexedDB persistence                      │
│  ui/app.ts              premium UI, View Transitions API           │
└───────────────────────────────────────────────────────────────────┘
```

Full data flow: document text → WordPiece tokenize (client-side) → token ids uploaded to a GPU storage buffer → `embedding_lookup.wgsl` → `pooling.wgsl` → chunk embedding (persisted in IndexedDB) → on query: same path for the query string → `similarity.wgsl` scores the query against every stored chunk in one dispatch → `topk.wgsl` selects the best *k* via GPU-side tournament reduction → only the final (index, score) pairs are read back to the CPU → UI renders ranked, highlighted results.

---

## Design decisions worth explaining

### Why SIF pooling instead of running the full transformer on the GPU?

The brief's kernel list — "embedding lookup / projection," "mean/attention pooling" — describes a *non-transformer* retrieval architecture, and that's what's implemented here, deliberately. Re-implementing six BERT encoder layers (multi-head attention, LayerNorm, GELU MLPs) as hand-written WGSL is a multi-week project on its own, and it would bury the actual point of this project — the retrieval kernels — under an unrelated (if impressive) transformer-in-WGSL exercise.

Instead, `modelprep` extracts the **real, pretrained WordPiece token-embedding table** from `sentence-transformers/all-MiniLM-L6-v2` and combines it with **SIF pooling** (Arora et al., 2017, *"A Simple but Tough-to-Beat Baseline for Sentence Embeddings"*) — weighting each token by `a / (a + word_frequency)` before averaging, then removing the corpus's top principal component. SIF is a well-documented, non-neural sentence-embedding method known to perform close to much heavier neural encoders on semantic similarity benchmarks. It's also exactly the right shape for the GPU work this project is about: a gather (embedding lookup), a weighted reduction (pooling), a batched dot product (similarity), and a parallel selection (top-k) — four genuinely different, individually-benchmarkable compute kernels, none of which need attention or LayerNorm to be real.

Every accuracy number quoted below is a real, measured cosine similarity — nothing is simulated.

### Why no CUDA stretch goal

The brief flagged an offline CUDA batch-embedding mode as an optional stretch goal. The development machine for this project has only integrated graphics (Intel Iris Xe, no NVIDIA GPU), so there was no hardware to build or test a CUDA path against. WebGPU, WGSL, GLSL, and C++ — the four required skills — are all still fully covered.

### Binary format, quantization, and packed-buffer alignment

`modelprep` writes a single custom binary (`.bmind`) designed so the browser can slice the downloaded `ArrayBuffer` directly into typed-array views with almost no parsing — see [`modelprep/include/bmind_format.hpp`](modelprep/include/bmind_format.hpp) for the exact byte layout (including the alignment padding needed so `Float32Array` views stay valid at every section boundary). Quantization is per-row (per-token) INT8 with a hand-computed scale — not a call into a quantization library — implemented in [`modelprep/src/quantize.cpp`](modelprep/src/quantize.cpp).

### Top-k: tournament reduction, not a full bitonic sort

`topk.wgsl` runs *k* rounds of a two-stage parallel max-reduction (workgroup-local reduce → cross-workgroup reduce → mask the winner → repeat), costing `O(k·n)` across *k* dispatches. A full bitonic sort would cost `O(n log²n)` in a single network but is considerably more complex to implement correctly. For this app's actual shape — a handful of results out of thousands of chunks, i.e. `k ≪ n` — the tournament approach wins on both simplicity and total work. A bitonic network is the documented next step if corpus size or *k* grow large enough to change that trade-off.

---

## Real, measured numbers

All numbers below were produced by actually running the tools in this repo — see [`modelprep/validation_report.txt`](modelprep/validation_report.txt) for the raw output and [`webapp/scripts/correctness-check.mjs`](webapp/scripts/correctness-check.mjs) for the in-browser check.

### Quantization (modelprep, `sentence-transformers/all-MiniLM-L6-v2`)

| Metric | Value |
|---|---|
| Vocabulary | 30,522 WordPiece tokens |
| Embedding dimension | 384 |
| fp32 embedding matrix | 44.71 MB |
| INT8 + per-row scales | 11.29 MB |
| Compression ratio | **3.96×** |
| Quantization time (C++, single-threaded) | ~370 ms |
| Cosine similarity, quantized vs fp32 reference (59 held-out sentences) | min **0.999785** · mean **0.999943** · max **0.999973** |

### GPU kernel correctness (WGSL output vs. the pure-JS reference implementation, in-browser)

| Kernel(s) | Metric | Value |
|---|---|---|
| `embedding_lookup.wgsl` + `pooling.wgsl` | Cosine similarity, GPU vs CPU chunk embedding (9 chunks) | min **0.9999999999999887** · mean **0.9999999999999916** |
| `similarity.wgsl` + `topk.wgsl` | Top-*k* set overlap, GPU vs CPU (order-independent) | **100%** (identical set) |
| `similarity.wgsl` + `topk.wgsl` | Max score delta on shared results | **2.09 × 10⁻⁷** |

All four compute kernels agree with the CPU reference to within float32 noise — this is `npm run correctness-check` actually calling the live GPU pipeline and the pure-JS reference implementation side by side in a real browser, not a claim.

### GPU vs CPU query latency, by corpus size

Measured with `npm run` → Benchmark tab → "Run sweep", on an **Intel Iris Xe (gen-12lp)** integrated GPU in Chrome 151, Windows 11. Corpus sizes above the demo's native 9 chunks are synthesized by resampling the indexed chunks, so absolute chunk count scales but token content repeats — this measures dispatch/compute scaling, not embedding quality.

| Corpus size | GPU total | CPU total | Speedup |
|---|---|---|---|
| 100 chunks | 49.4 ms | 27.7 ms | 0.56× (CPU wins) |
| 1,000 chunks | 86.8 ms | 278.9 ms | 3.21× |
| 10,000 chunks | 149.1 ms | 3,032.1 ms | **20.34×** |

The 100-chunk row is reported honestly, not hidden: WebGPU's fixed per-dispatch and buffer-readback overhead is real, and at only a few hundred chunks it can outweigh the actual compute, letting the CPU path legitimately win. The crossover is clearly visible by 1,000 chunks and the GPU pulls dramatically ahead by 10,000 — this is the expected, textbook shape for GPU compute versus scalar CPU work, and the in-app "Benchmark" tab reproduces it live rather than asking anyone to trust a screenshot. (One throwaway warm-up dispatch runs before the sweep starts, since integrated GPUs ramp clock/power state up under load and the very first WebGPU dispatch of a session pays extra driver setup cost — without that warm-up the sizes can even come back non-monotonic, which is itself a real and worth-knowing artifact of benchmarking on integrated graphics.)

### Cross-browser

| Browser | Result |
|---|---|
| Chrome 151 (Windows 11) | ✅ full pipeline verified — ingestion, search, GPU Internals panel, embedding map, benchmark sweep, zero console errors |
| Edge 151 (Windows 11) | ✅ same, zero console errors |
| Firefox 141+ (Windows/macOS) | Not installed in this dev environment, untested. The app uses only stable, non-experimental WGSL/WebGPU features (no origin trials), so it's expected to work, but that expectation is explicitly unverified rather than claimed as fact. |
| Safari (Technology Preview) | Untested — no macOS device available in this environment. |

---

## Feature checklist

**Core**
- [x] WebGPU capability detection with a graceful, specific fallback message on unsupported browsers/devices
- [x] C++ `modelprep` tool: loads weights, quantizes to INT8 (hand-written, per-row scale), validates against an fp32 Python reference, exports the binary
- [x] WGSL embedding-lookup + pooling kernels, correctness-tested against a CPU/JS reference (see numbers above)
- [x] WGSL batched similarity + top-k kernels, same correctness testing
- [x] End-to-end upload → search → ranked results, fully client-side, no backend
- [x] GPU Internals panel with real timestamp-query-derived timings
- [x] CPU/JS fallback path + on-screen GPU vs CPU comparison

**Differentiators**
- [x] GLSL/WebGL2 embedding-space visualizer, with the 2D layout itself computed by a GPU compute kernel (`project2d.wgsl`) projecting onto axes found via CPU-side power iteration
- [x] Persistent IndexedDB cache — reloading or re-adding an unchanged document skips GPU re-embedding entirely
- [x] Documented benchmark sweep across corpus sizes (100 / 1,000 / 10,000 chunks) with real before/after-style numbers
- [x] Cross-browser notes (Chrome + Edge verified; Firefox expected-but-untested, stated as such)
- [ ] A second embedding-model size — not implemented; the pipeline is written to generalize (hidden dimension is templated into the WGSL at pipeline-build time, not hardcoded), but only MiniLM-L6-v2 has actually been run through it

**Stretch**
- [ ] CUDA batch mode — not implemented; no NVIDIA GPU was available in the development environment (see "Design decisions" above)

---

## Repository layout

```
modelprep/          C++ CLI tool (CMake, zero external C++ dependencies)
  python/            export_embeddings.py — pulls real pretrained weights, computes SIF weights, prepares fp32 reference
  src/                quantize / pack / validate / raw_io — hand-written, no JSON/serialization library
  include/            bmind_format.hpp — the authoritative binary format spec
  tests/              dependency-free unit tests (quantization + pooling math)

webapp/              TypeScript + Vite static site
  src/gpu/            device detection, .bmind parser, pipeline orchestration, profiler, WGSL shaders
  src/ingest/         WordPiece tokenizer, chunker, PDF/TXT/MD loading
  src/cpu/            pure-JS reference implementation (correctness + CPU benchmark path)
  src/viz/            GLSL/WebGL2 scatter renderer, PCA layout, k-means clustering
  src/db/             IndexedDB caching
  src/ui/             premium UI (vanilla TS + CSS, View Transitions API, no framework)
  src/bench/          GPU-vs-CPU benchmark harness
  scripts/            correctness-check.mjs — drives a real browser to verify WGSL == CPU reference
```

---

## Building from scratch

### 1. Environment (conda)

```bash
conda create -n browsermind -c conda-forge python=3.11 numpy scikit-learn pip
conda activate browsermind
pip install -r modelprep/python/requirements.txt
```

A C++17 compiler is needed to build `modelprep`. conda-forge's Windows `m2w64-toolchain` package is a frozen GCC 5.3 build (kept old deliberately for scientific-package ABI compatibility) and does **not** support `<filesystem>`/`<optional>` — this project needs an actual C++17 compiler. On Windows, [WinLibs](https://winlibs.com/) (installable via `winget install BrechtSanders.WinLibs.POSIX.UCRT`) provides a current, self-contained MinGW-w64 GCC with no Visual Studio dependency. On Linux/macOS, system GCC 9+/Clang 10+ is sufficient and conda's toolchain is unnecessary.

### 2. Export the model (Python, inside the conda env)

```bash
cd modelprep/python
python export_embeddings.py --out ../build/raw
```

Downloads `sentence-transformers/all-MiniLM-L6-v2`'s tokenizer + weights from Hugging Face, extracts the WordPiece embedding table, computes SIF weights from real English word frequencies, and writes the raw fp32 pipeline inputs.

### 3. Build and run modelprep (C++)

```bash
cd modelprep
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=<path to g++>
cmake --build build
./build/modelprep_tests          # unit tests
./build/modelprep --raw build/raw --out ../webapp/public/models/minilm-sif.bmind --report build/validation_report.txt
```

### 4. Run the webapp

```bash
cd webapp
npm install
npm run dev
```

---

## How to talk about this (resume bullets — verified against the numbers above)

- Built a fully client-side semantic search engine using hand-written WebGPU compute shaders (WGSL) for embedding generation, weighted pooling, batched similarity scoring, and GPU-side top-k ranking — zero backend, with kernel output matching a CPU reference implementation to 12+ decimal places of cosine similarity.
- Designed a C++ model-preparation pipeline that quantizes a pretrained embedding table to INT8 with hand-written per-row quantization (3.96× size reduction, 99.994% mean cosine fidelity vs. fp32) and validates it against a Python reference before packing it for direct WebGPU buffer upload.
- Implemented a GLSL/WebGL2 visualization layer, independent from the WGSL compute pipeline, rendering a GPU-computed 2D projection of the embedding space.
- Measured and published GPU-vs-CPU query latency across corpus sizes (100 / 1,000 / 10,000 chunks), showing a realistic crossover from CPU-favorable at small scale to a 20×+ GPU speedup at 10,000 chunks — including the honest small-N case where fixed dispatch overhead makes the CPU path faster.

Only the numbers actually measured in this repo are used above — no invented percentages.
