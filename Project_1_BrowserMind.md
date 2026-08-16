# Project 1: BrowserMind — A WebGPU-Native, Zero-Backend Semantic Search & RAG Engine

> **Status: built.** This document is the original planning brief, kept as-is for reference. See **[README.md](README.md)** for what was actually implemented, real measured benchmark numbers, and the places the implementation deliberately diverged from this plan — most notably: SIF pooling over a real pretrained WordPiece embedding table instead of a full transformer forward pass on the GPU (§3.1's kernel list — "embedding lookup / projection," "mean/attention pooling" — describes exactly this architecture, not a transformer), and no CUDA stretch goal (§4, §6) since the development machine has no NVIDIA GPU. Every "Core" checklist item in §4 is done; most "Differentiators" are too. Details and reasoning in the README.

## 0. One-line pitch

A document search tool that tokenizes, embeds, and semantically searches your files **entirely inside the browser tab**, with every heavy computation — embedding matmuls, pooling, similarity scoring, top-k ranking — running as hand-written **WGSL compute shaders** on the user's GPU. No server, no API key, no data ever leaves the device. A companion **C++ command-line tool** prepares and quantizes the embedding model into a GPU-upload-ready binary format, giving you a genuine C++ host-side component alongside the shader work.

This is not "call an embedding API and show results." This is writing the compute kernels yourself, in the exact language (WGSL/WebGPU) and adjacent language (GLSL, for the visualization layer) the job description names, applied to an NLP problem (semantic retrieval) that is directly useful and demoable.

---

## 1. Why this project, why now (market rationale)

Client-side GPU inference in the browser moved from novelty to production pattern in 2026:

- WebGPU is now enabled by default across Chrome, Edge, and Firefox, with Safari catching up — this was not true a couple of years ago, so "I built something real with it" is a genuinely current signal, not a stale skill.
- Engines like WebLLM and Transformers.js v4 have shown embedding generation via WebGPU running 40–75x faster than the WASM/CPU path, and full LLM inference running at 70–80% of native GPU speed inside a browser tab — this is the exact performance envelope your project will be measured against.
- Recruiters and inference companies increasingly separate "used a GPU library" from "wrote the compute shader that does the work" — the latter is what a GPU Programmer role actually tests for.
- It is directly relevant to micro1's stated scope: "design, implement, and optimize GPU-based software using CUDA, WebGPU or GLSL" and "develop host-side logic and integrations in C++." This project hits WebGPU, GLSL, and C++ all in one deliverable — the only GPU language it does not touch is raw CUDA, which is why Project 2 exists.

For an NLP-leaning resume like yours (RAG chatbot, resume parsing, legal LLM), this project reads as "the same engineer, but now working one layer down the stack" — a natural and credible progression, not a pivot.

---

## 2. What the finished product does (from a user's perspective)

1. User opens a web page (no login, no install). The app checks WebGPU availability and reports the adapter/device info (a small but important detail — shows you understand the API's capability-detection model).
2. User uploads or pastes documents (PDF/TXT/MD). The app chunks the text client-side.
3. The app generates embeddings for every chunk using a small embedding model (e.g., a MiniLM-class sentence embedding model, quantized to INT8) — computed via your own WGSL compute pipeline, not a library call.
4. A live "GPU Internals" panel shows, in real time: number of compute passes dispatched, workgroup size chosen, buffer sizes in flight, and per-pass GPU timing captured via WebGPU timestamp queries.
5. User types a natural-language query. The app embeds the query (same GPU pipeline) and runs a batched similarity-scoring + top-k kernel across all chunk embeddings — again on the GPU, not JavaScript.
6. Results appear ranked, with matched passages highlighted, in well under 100ms for corpora in the thousands-of-chunks range.
7. A second view shows an interactive 2D "embedding map" — every chunk plotted as a point, colored by cluster, rendered with a hand-written GLSL/WebGL shader (not a charting library), zoomable and clickable.
8. A benchmark toggle lets the user re-run the same search on a CPU/JavaScript fallback path so they can see, live, the GPU speedup with their own eyes. This single feature does more to prove GPU competence to a non-technical reviewer than any README paragraph could.

---

## 3. System architecture

### 3.1 Two independently valuable components

**A. The offline C++ preparation tool ("modelprep")**
A command-line C++ program that:
- Loads a small pretrained embedding model's exported weights (e.g., exported from a HuggingFace sentence-transformer to a raw tensor / safetensors-style format).
- Performs INT8 quantization of the weight matrices (this is where you demonstrate understanding of quantization, not just calling `torch.quantize`).
- Validates the quantized output numerically against a Python/PyTorch reference (documented tolerance, e.g., cosine similarity of quantized vs. full-precision embeddings > 0.98).
- Packs the quantized weights, vocabulary/tokenizer data, and metadata into a compact binary format designed to be uploaded directly into WebGPU storage buffers with minimal client-side parsing.
- This is your "host-side logic in C++" deliverable, decoupled from the browser so it can be shown and explained independently (e.g., in a CLI demo, not just buried inside the web app).

**B. The browser application**
- **Ingestion & chunking layer** (TypeScript): splits documents into overlapping token windows.
- **WGSL compute pipeline**, consisting of several distinct compute shaders, each a deliberate, separately-benchmarked kernel:
  - *Embedding lookup / projection kernel* — the core matmul-like operation that turns token IDs into vectors.
  - *Mean/attention pooling kernel* — reduces per-token vectors into one chunk-level embedding.
  - *Batched similarity kernel* — computes cosine similarity (or dot product, with pre-normalized vectors) between the query embedding and every stored chunk embedding in a single dispatch.
  - *Top-k selection kernel* — a GPU-side partial sort/selection (e.g., a bitonic-sort-based or tournament-based top-k) rather than pulling all scores back to the CPU and sorting in JavaScript.
- **GLSL/WebGL visualization layer**: a separate rendering pipeline (distinct from the WGSL compute pipeline, deliberately, to show you can work in both shading languages) that draws the embedding-space scatter plot as GPU-shaded points, with a lightweight GPU-computed force-directed or precomputed 2D layout.
- **Instrumentation layer**: WebGPU timestamp queries wrapped around every compute pass, surfaced in the "GPU Internals" panel.

### 3.2 Data flow (conceptual, no code)

Document text → chunking → tokenization (client-side, using a simple, well-documented tokenization scheme matching the model you chose) → token IDs uploaded to a GPU storage buffer → embedding-lookup compute pass → pooling compute pass → chunk embedding stored in a persistent GPU buffer (or IndexedDB-backed cache so re-visits don't recompute) → on query: same pipeline for the query string → similarity compute pass scores query against all stored chunk embeddings → top-k compute pass → results read back to CPU only at the very end, once, as a small array of indices/scores → UI renders ranked list + highlights.

---

## 4. Feature checklist (what "done" looks like)

**Core (must-have for the demo to be credible):**
- [ ] WebGPU capability detection + graceful fallback message on unsupported browsers/devices.
- [ ] C++ modelprep tool: loads weights, quantizes to INT8, validates against reference, exports binary.
- [ ] WGSL embedding-lookup + pooling kernels, correctness-tested against a CPU/JS reference implementation.
- [ ] WGSL batched similarity + top-k kernels, correctness-tested the same way.
- [ ] End-to-end document upload → search → ranked results flow, working in the browser with no backend server.
- [ ] GPU Internals panel with real timestamp-query-derived timings (not fake/simulated numbers).
- [ ] CPU/JS fallback path + on-screen "GPU vs CPU" speed comparison toggle.

**Differentiators (what makes this stand out, not just "complete"):**
- [ ] GLSL/WebGL embedding-space visualizer with real GPU-side layout computation.
- [ ] Persistent client-side embedding cache (IndexedDB) so the tool is genuinely usable across sessions.
- [ ] A documented optimization pass: an initial "naive" WGSL kernel version, then an optimized version (better workgroup sizing, buffer reuse, reduced dispatch count), with a before/after benchmark table in the README — this is the single most important artifact for a GPU-hiring reviewer.
- [ ] Support for at least two different embedding-model sizes, showing the pipeline generalizes rather than being hardcoded to one model's dimensions.
- [ ] Cross-browser test matrix (Chrome, Edge, Firefox at minimum) with notes on any WebGPU implementation differences you hit.

**Stretch (only if time allows):**
- [ ] A small CUDA-accelerated batch mode (offline, native, not in-browser) that pre-embeds very large corpora faster than the C++/CPU path, whose output feeds the same binary format the browser consumes — this explicitly bridges CUDA and WebGPU in one project and is worth strongly considering, since it lets one project touch three of the job's four listed skills (CUDA, WebGPU, C++; GLSL comes from the visualizer).

---

## 5. UI / demoable surface (this is what you screen-share or send a link to)

- **Landing/upload screen** — clean drop zone, WebGPU adapter info shown subtly (signals technical seriousness to anyone who knows what it means, invisible clutter to anyone who doesn't).
- **Processing view** — live progress bar tied to actual compute-pass completion (via GPU timestamp queries / fence-based signaling), not a fake spinner.
- **Search view** — query box, ranked results with highlighted spans, response time displayed in milliseconds, GPU-vs-CPU toggle visibly showing the delta.
- **GPU Internals drawer** — collapsible panel: per-kernel timing table, workgroup size, dispatch count, total buffer memory in use. This is the panel that turns the demo from "a search box" into "clearly the work of someone who understands GPUs."
- **Embedding map view** — the GLSL-rendered scatter plot, pannable/zoomable, click-to-preview chunk text.

Deployment: since there is no backend, this can be hosted as a static site (e.g., GitHub Pages or Vercel static export) — meaning anyone, including a recruiter, can open a link and use it immediately with no setup. That ease of access is itself a big part of why this project is worth the effort.

---

## 6. Suggested build plan (6–9 weeks, part-time)

1. **Week 1 — Foundations**: Learn the WebGPU API surface (devices, adapters, buffers, bind groups, compute pipelines) and WGSL syntax. Stand up the C++ project skeleton (CMake) for modelprep. Pick your embedding model.
2. **Week 2 — C++ modelprep tool**: Load weights, implement INT8 quantization, write the numerical validation harness against a Python reference, export the binary format.
3. **Week 3 — Core WGSL kernels v1 (naive)**: Embedding lookup + pooling, correctness-tested. Get something end-to-end working, even if slow.
4. **Week 4 — Similarity + top-k kernels**: Batched scoring and GPU-side selection, correctness-tested against a JS reference.
5. **Week 5 — Optimization pass**: Profile with WebGPU timestamp queries, identify bottlenecks (dispatch overhead, buffer layout, workgroup size), implement v2 kernels, record before/after numbers.
6. **Week 6 — UI build-out**: Upload/search/results flow, GPU Internals panel wired to real data.
7. **Week 7 — GLSL visualization layer**: Embedding-space scatter plot, GPU-computed layout.
8. **Week 8 — Cross-browser testing, caching, polish**: IndexedDB cache, fallback path, browser test matrix.
9. **Week 9 — Documentation & deployment**: README with architecture diagram, benchmark tables, deployed static demo link, short write-up/case study.

---

## 7. Evaluation metrics to actually record and publish

- Query latency (ms), GPU path vs. CPU/JS fallback path, at multiple corpus sizes (e.g., 100 / 1,000 / 10,000 chunks).
- Embedding throughput (chunks/sec) during ingestion.
- Per-kernel GPU time breakdown (from timestamp queries).
- Memory footprint (GPU buffer bytes in use) at each corpus size.
- Numerical accuracy delta between INT8-quantized and full-precision embeddings (cosine similarity).
- Cross-browser support notes (what worked, what didn't, any WGSL feature gaps hit on Safari/Firefox vs. Chrome).

Publishing these numbers, not just claiming "it's fast," is what separates a portfolio piece a reviewer trusts from one they skim past.

---

## 8. How to talk about this on your resume (draft bullets — edit once built, don't paste unverified)

- "Built a fully client-side semantic search engine using hand-written WebGPU compute shaders (WGSL) for embedding generation, similarity scoring, and GPU-side top-k ranking — zero backend, sub-100ms query latency on 10k+ document chunks."
- "Designed a C++ model-preparation pipeline that quantizes embedding model weights to INT8 and validates numerical accuracy against a PyTorch reference before packing them for direct WebGPU buffer upload."
- "Implemented a GLSL/WebGL visualization layer rendering GPU-computed embedding-space projections, independent from the WGSL compute pipeline used for search."
- "Profiled and optimized WGSL kernels using WebGPU timestamp queries, documenting workgroup-size and buffer-reuse changes that reduced per-query GPU time by [X]% (fill in with your real measured number)."

Only use numbers you actually measured. An unverifiable percentage is worse for you than no percentage at all in an interview.

---

## 9. Honest scope warning

This is a real systems project, not a weekend build. The hardest parts, in order, will likely be: (1) getting WGSL bind group / buffer layout mental model right, (2) debugging GPU compute correctness without good in-shader debugging tools (you will lean heavily on CPU-side reference comparisons), (3) the top-k-on-GPU kernel, which is a genuinely non-trivial parallel algorithm. Budget real time for all three. That difficulty is exactly why finishing it is valuable — it is not padding, it is the point.
