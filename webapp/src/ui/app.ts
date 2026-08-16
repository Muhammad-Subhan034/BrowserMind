import "../styles/main.css";
import { h, hs, mount, transitionMount, escapeHtml, highlight } from "./dom";
import { icons } from "./icons";
import { detectGpu, initGpu, type GpuCapability, type GpuContext } from "../gpu/device";
import { loadBMindModel, type BMindModel } from "../gpu/modelFormat";
import { GpuPipeline, type DispatchInfo } from "../gpu/pipeline";
import type { PassTiming } from "../gpu/profiler";
import { chunkDocument } from "../ingest/chunk";
import { tokenize } from "../ingest/tokenizer";
import { loadFile } from "../ingest/documents";
import { embedBatchCPU, searchCPU } from "../cpu/fallback";
import { hashText, getCachedDoc, getCachedChunks, putDocument, clearAll, type CachedChunk } from "../db/cache";
import { findTop2Axes } from "../viz/layout";
import { kmeans } from "../viz/cluster";
import { ScatterRenderer, type ScatterPoint } from "../viz/scatter";
import { runBenchmarkSweep, type BenchmarkPoint } from "../bench/benchmark";

type Phase = "boot" | "unsupported" | "load-error" | "landing" | "processing" | "search";
type Tab = "results" | "map" | "bench";

interface CorpusChunk {
  id: string;
  docName: string;
  text: string;
  tokenIds: number[];
}

interface ProcessingStep {
  label: string;
  status: "pending" | "active" | "done";
}

interface DocInput {
  name: string;
  text: string;
}

const K_RESULTS = 8;

export class App {
  private root: HTMLElement;
  private phase: Phase = "boot";

  private capability: GpuCapability | null = null;
  private gpu: GpuContext | null = null;
  private model: BMindModel | null = null;
  private pipeline: GpuPipeline | null = null;
  private loadError: string | null = null;

  private chunks: CorpusChunk[] = [];
  private chunkEmbeddings: Float32Array = new Float32Array(0);
  private chunkEmbeddingsBuf: GPUBuffer | null = null;
  private docNames: string[] = [];

  private processingSteps: ProcessingStep[] = [];
  private processingProgress = 0;

  private query = "";
  private results: { chunk: CorpusChunk; score: number }[] = [];
  private lastGpuMs: number | null = null;
  private lastCpuMs: number | null = null;
  private benchmarkOn = false;
  private searchToken = 0;

  private drawerOpen = false;
  private lastTimings: PassTiming[] = [];
  private lastDispatches: DispatchInfo[] = [];
  private lastOpLabel = "";

  private activeTab: Tab = "results";
  private scatterRenderer: ScatterRenderer | null = null;
  private mapPoints: ScatterPoint[] = [];
  private mapDirty = true;

  private toasts: { id: number; text: string; tone: "ok" | "info" }[] = [];

  // Live element refs for the search shell, populated once so per-keystroke
  // and per-toggle updates never rebuild (and never steal focus from) the
  // search input.
  private refs: {
    searchInput?: HTMLInputElement;
    resultsList?: HTMLElement;
    latencyChip?: HTMLElement;
    benchSwitch?: HTMLElement;
    drawer?: HTMLElement;
    drawerScrim?: HTMLElement;
    drawerBody?: HTMLElement;
    resultsSection?: HTMLElement;
    mapSection?: HTMLElement;
    mapCanvas?: HTMLCanvasElement;
    mapPreview?: HTMLElement;
    tabResults?: HTMLElement;
    tabMap?: HTMLElement;
    tabBench?: HTMLElement;
    corpusStrip?: HTMLElement;
    gpuPill?: HTMLElement;
    benchSection?: HTMLElement;
    benchBody?: HTMLElement;
    benchRunBtn?: HTMLButtonElement;
  } = {};

  private benchRunning = false;
  private benchResults: BenchmarkPoint[] = [];

  constructor(root: HTMLElement) {
    this.root = root;
    this.boot();
  }

  private async boot() {
    this.capability = await detectGpu();
    if (!this.capability.supported) {
      this.phase = "unsupported";
      this.render();
      document.getElementById("app-loading")?.remove();
      return;
    }
    try {
      this.gpu = await initGpu();
    } catch (err) {
      this.capability = {
        supported: false,
        reason: err instanceof Error ? err.message : String(err),
        timestampQuerySupported: false,
      };
      this.phase = "unsupported";
      this.render();
      document.getElementById("app-loading")?.remove();
      return;
    }
    try {
      const { model } = await loadBMindModel("models/minilm-sif.bmind");
      this.model = model;
      this.pipeline = new GpuPipeline(this.gpu.device, model, this.capability.timestampQuerySupported);
    } catch (err) {
      // WebGPU itself is fine here -- this is a model-asset problem (bad
      // deploy, wrong base path, offline with no cache), which deserves a
      // different message than "your browser doesn't support WebGPU".
      this.loadError = err instanceof Error ? err.message : String(err);
      this.phase = "load-error";
      this.render();
      document.getElementById("app-loading")?.remove();
      return;
    }
    this.phase = "landing";
    this.render();
    document.getElementById("app-loading")?.remove();
  }

  // -------------------------------------------------------------------
  // Top-level render (called only on phase transitions)
  // -------------------------------------------------------------------
  private render() {
    this.refs = {};
    let node: Node;
    switch (this.phase) {
      case "boot":
        node = h("div");
        break;
      case "unsupported":
        node = this.renderUnsupported();
        break;
      case "load-error":
        node = this.renderLoadError();
        break;
      case "landing":
        node = this.renderShell(this.renderLanding());
        break;
      case "processing":
        node = this.renderShell(this.renderProcessing());
        break;
      case "search":
        node = this.renderShell(this.renderSearch());
        break;
    }
    transitionMount(this.root, () => node);
    if (this.phase === "search" && this.activeTab === "map") {
      requestAnimationFrame(() => this.mountScatter());
    }
  }

  private renderShell(content: Node): Node {
    const gpuPill = h(
      "div",
      { class: `gpu-pill ${this.capability?.supported ? "ok" : "bad"}` },
      h("span", { class: "dot" }),
      this.capability?.adapterInfo?.description || this.capability?.adapterInfo?.vendor || "GPU",
    );
    this.refs.gpuPill = gpuPill;

    return h(
      "div",
      {},
      h("div", { class: "ambient" }),
      h(
        "header",
        { class: "topbar" },
        h(
          "div",
          { class: "brand" },
          h("div", { class: "brand-mark" }, h("div", { html: icons.chip })),
          h("span", { class: "brand-name" }, "BrowserMind"),
        ),
        h(
          "div",
          { style: "display:flex;align-items:center;gap:10px;" },
          gpuPill,
          this.phase === "search" &&
            h(
              "button",
              { class: "btn btn-ghost btn-sm", onclick: () => this.openDrawer() },
              h("div", { html: icons.chip, style: "width:14px;height:14px" }),
              "GPU Internals",
            ),
        ),
      ),
      h("main", { class: "view" }, content),
      this.renderDrawerShell(),
      h("div", { class: "toast-stack" }, ...this.toasts.map((t) => this.renderToast(t))),
    );
  }

  // -------------------------------------------------------------------
  // Unsupported-browser view
  // -------------------------------------------------------------------
  private renderUnsupported(): Node {
    return h(
      "div",
      { class: "unsupported glass" },
      h("div", { style: "font-size:34px;margin-bottom:14px;" }, "⚠️"),
      h("h2", {}, "WebGPU isn't available here"),
      h("p", {}, this.capability?.reason || "This browser or device doesn't expose the WebGPU API BrowserMind is built on."),
      h("p", {}, "BrowserMind runs every embedding and search computation as a real GPU compute shader — there's no CPU-only mode to fall back to for the model itself."),
      h(
        "div",
        { class: "browsers" },
        h("span", {}, "Chrome 113+"),
        h("span", {}, "Edge 113+"),
        h("span", {}, "Firefox 141+ (win/mac)"),
      ),
    );
  }

  private renderLoadError(): Node {
    return h(
      "div",
      { class: "unsupported glass" },
      h("div", { style: "font-size:34px;margin-bottom:14px;" }, "🧩"),
      h("h2", {}, "WebGPU works — the model failed to load"),
      h("p", {}, "Your browser and GPU are fine. BrowserMind couldn't fetch or parse the packed embedding model (public/models/minilm-sif.bmind)."),
      h("p", { class: "mono", style: "font-size:12px;color:var(--text-tertiary);margin-top:10px;" }, this.loadError || ""),
      h(
        "button",
        { class: "btn btn-primary btn-sm", style: "margin-top:20px;", onclick: () => location.reload() },
        "Retry",
      ),
    );
  }

  // -------------------------------------------------------------------
  // Landing view
  // -------------------------------------------------------------------
  private renderLanding(): Node {
    const onFiles = async (files: FileList | File[]) => {
      const arr = Array.from(files).filter((f) => /\.(txt|md|markdown|pdf)$/i.test(f.name));
      if (arr.length === 0) {
        this.toast("Only .txt, .md, and .pdf files are supported.", "info");
        return;
      }
      const docs: DocInput[] = [];
      for (const f of arr) docs.push(await loadFile(f));
      this.ingestDocuments(docs);
    };

    const dropzone = h(
      "div",
      {
        class: "dropzone glass",
        ondragover: (e: DragEvent) => { e.preventDefault(); dropzone.classList.add("dragover"); },
        ondragleave: () => dropzone.classList.remove("dragover"),
        ondrop: (e: DragEvent) => {
          e.preventDefault();
          dropzone.classList.remove("dragover");
          if (e.dataTransfer?.files?.length) onFiles(e.dataTransfer.files);
        },
      },
      h("div", { class: "dropzone-icon" }, h("div", { html: icons.upload, style: "width:22px;height:22px;color:var(--accent-1)" })),
      h("h3", {}, "Drop documents to search"),
      h("p", {}, "TXT, Markdown, or PDF — chunked, embedded, and indexed entirely on your GPU. Nothing is uploaded."),
      h(
        "div",
        { class: "actions" },
        h(
          "label",
          { class: "btn btn-primary" },
          h("div", { html: icons.upload, style: "width:15px;height:15px" }),
          "Choose files",
          h("input", {
            type: "file",
            multiple: true,
            accept: ".txt,.md,.markdown,.pdf",
            onchange: (e: Event) => {
              const input = e.target as HTMLInputElement;
              if (input.files?.length) onFiles(input.files);
            },
          }),
        ),
        h(
          "button",
          { class: "btn btn-ghost", onclick: () => this.ingestDemo() },
          h("div", { html: icons.bolt, style: "width:14px;height:14px" }),
          "Try the demo corpus",
        ),
      ),
    );

    const steps: [string, string, string][] = [
      ["01", "Tokenize", "A hand-written WordPiece tokenizer splits text client-side — the same algorithm BERT-family models were trained with."],
      ["02", "Embed on GPU", "WGSL compute kernels dequantize INT8 weights, gather token embeddings, and SIF-pool them into one vector per chunk."],
      ["03", "Score on GPU", "A batched cosine-similarity kernel scores your query against every chunk in a single dispatch."],
      ["04", "Rank on GPU", "A tournament-reduction top-k kernel selects the best matches — only the final results ever touch the CPU."],
    ];

    return h(
      "div",
      { class: "landing" },
      h(
        "div",
        { class: "hero" },
        h("div", { class: "hero-eyebrow" }, h("div", { html: icons.dot, style: "width:8px;height:8px;color:var(--accent-1)" }), "WebGPU · WGSL · zero backend"),
        h("h1", {}, "Semantic search that never leaves your tab."),
        h("p", { class: "lede" }, "Every embedding, similarity score, and ranking is computed by hand-written WGSL compute shaders running on your own GPU. No server, no API key, no data ever transmitted."),
      ),
      dropzone,
      h(
        "div",
        { class: "steps-strip" },
        ...steps.map(([num, title, desc]) =>
          h(
            "div",
            { class: "step-card glass" },
            h("span", { class: "badge" }, "WGSL"),
            h("div", { class: "num" }, num),
            h("h4", {}, title),
            h("p", {}, desc),
          ),
        ),
      ),
    );
  }

  // -------------------------------------------------------------------
  // Ingestion pipeline
  // -------------------------------------------------------------------
  private async ingestDemo() {
    try {
      const res = await fetch("docs/demo-corpus.json");
      const docs: DocInput[] = await res.json();
      await this.ingestDocuments(docs);
    } catch {
      this.toast("Failed to load the demo corpus.", "info");
    }
  }

  private markStep(i: number, status: ProcessingStep["status"], patch = true) {
    if (this.processingSteps[i]) this.processingSteps[i].status = status;
    if (patch) this.patchProcessing();
  }

  private async ingestDocuments(docs: DocInput[]) {
    if (!this.pipeline || !this.model) return;
    this.phase = "processing";
    this.processingProgress = 4;
    this.processingSteps = [
      { label: "Tokenizing documents (client-side WordPiece)", status: "active" },
      { label: "Checking IndexedDB cache", status: "pending" },
      { label: "Running embedding_lookup.wgsl + pooling.wgsl", status: "pending" },
      { label: "Caching chunks + embeddings locally", status: "pending" },
      { label: "Computing embedding-space layout", status: "pending" },
    ];
    this.render();

    const allChunks: CorpusChunk[] = [...this.chunks];
    const embeddingParts: Float32Array[] = this.chunkEmbeddings.length ? [this.chunkEmbeddings] : [];
    const hiddenDim = this.model.hiddenDim;

    this.markStep(0, "done");
    this.markStep(1, "active");

    for (let i = 0; i < docs.length; i++) {
      const doc = docs[i];
      const hash = await hashText(doc.text);
      const cachedMeta = await getCachedDoc(hash);
      let docChunks: CorpusChunk[];
      let docEmbeddings: Float32Array;

      if (cachedMeta) {
        const cached = await getCachedChunks(hash);
        docChunks = cached.map((c) => ({ id: c.id, docName: c.docName, text: c.text, tokenIds: c.tokenIds }));
        docEmbeddings = new Float32Array(cached.length * hiddenDim);
        cached.forEach((c, ci) => docEmbeddings.set(c.embedding, ci * hiddenDim));
        this.toast(`"${doc.name}" loaded from cache (${cached.length} chunks)`, "ok");
      } else {
        this.markStep(2, "active");
        const chunks = chunkDocument(doc.name, doc.text, this.model);
        docChunks = chunks.map((c) => ({ id: c.id, docName: c.docName, text: c.text, tokenIds: c.tokenIds }));

        if (chunks.length > 0) {
          const embedResult = await this.pipeline.embedBatch(chunks.map((c) => c.tokenIds));
          docEmbeddings = embedResult.embeddings;
          this.lastTimings = embedResult.timings;
          this.lastDispatches = this.pipeline.lastDispatches;
          this.lastOpLabel = `ingest “${doc.name}” (${chunks.length} chunks)`;
          embedResult.buffer.destroy();
        } else {
          docEmbeddings = new Float32Array(0);
        }

        this.markStep(3, "active");
        const cachedChunks: CachedChunk[] = docChunks.map((c, ci) => ({
          id: c.id,
          docHash: hash,
          docName: c.docName,
          text: c.text,
          tokenIds: c.tokenIds,
          embedding: docEmbeddings.slice(ci * hiddenDim, (ci + 1) * hiddenDim),
          charStart: 0,
          charEnd: 0,
        }));
        await putDocument({ hash, name: doc.name, chunkCount: docChunks.length, addedAt: Date.now() }, cachedChunks);
      }

      allChunks.push(...docChunks);
      embeddingParts.push(docEmbeddings);
      if (!this.docNames.includes(doc.name)) this.docNames.push(doc.name);

      this.processingProgress = 8 + Math.round(62 * (i + 1) / docs.length);
      this.patchProcessing();
    }

    this.markStep(1, "done");
    this.markStep(2, "done");
    this.markStep(3, "done");
    this.markStep(4, "active");

    this.chunks = allChunks;
    const total = new Float32Array(allChunks.length * hiddenDim);
    let o = 0;
    for (const part of embeddingParts) { total.set(part, o); o += part.length; }
    this.chunkEmbeddings = total;

    this.chunkEmbeddingsBuf?.destroy();
    this.chunkEmbeddingsBuf = this.pipeline.uploadEmbeddings(total);
    this.mapDirty = true;
    this.exposeDebugHook();

    this.processingProgress = 100;
    this.markStep(4, "done");
    await new Promise((r) => setTimeout(r, 260));

    this.phase = "search";
    this.render();
  }

  private renderProcessing(): Node {
    const circumference = 2 * Math.PI * 54;
    const offset = circumference * (1 - this.processingProgress / 100);
    return h(
      "div",
      { class: "processing" },
      h(
        "div",
        { class: "processing-ring" },
        hs(
          "svg",
          { viewBox: "0 0 120 120" },
          hs("defs", {}, hs("linearGradient", { id: "ringGrad", x1: "0", y1: "0", x2: "1", y2: "1" },
            hs("stop", { offset: "0%", "stop-color": "#6ea8ff" }),
            hs("stop", { offset: "100%", "stop-color": "#b28dff" }))),
          hs("circle", { class: "track", cx: "60", cy: "60", r: "54" }),
          hs("circle", { class: "fill", cx: "60", cy: "60", r: "54", "stroke-dasharray": `${circumference}`, "stroke-dashoffset": `${offset}` }),
        ),
        h("div", { class: "pct", id: "processing-pct" }, `${this.processingProgress}%`),
      ),
      h("h2", {}, "Indexing on your GPU"),
      h(
        "div",
        { class: "stage-list", id: "stage-list" },
        ...this.processingSteps.map((s) => this.renderStageRow(s)),
      ),
    );
  }

  private renderStageRow(s: ProcessingStep): Node {
    const icon = s.status === "done" ? icons.check : s.status === "active" ? icons.spinner : icons.dot;
    return h(
      "div",
      { class: `stage-row ${s.status}` },
      h("div", { class: `icon ${s.status === "active" ? "spin" : ""}`, html: icon }),
      s.label,
    );
  }

  private patchProcessing() {
    if (this.phase !== "processing") return;
    const pctEl = document.getElementById("processing-pct");
    if (pctEl) pctEl.textContent = `${this.processingProgress}%`;
    const ring = document.querySelector(".processing-ring .fill") as SVGCircleElement | null;
    if (ring) {
      const circumference = 2 * Math.PI * 54;
      ring.setAttribute("stroke-dashoffset", `${circumference * (1 - this.processingProgress / 100)}`);
    }
    const list = document.getElementById("stage-list");
    if (list) {
      list.replaceChildren(...this.processingSteps.map((s) => this.renderStageRow(s)));
    }
  }

  // -------------------------------------------------------------------
  // Search view
  // -------------------------------------------------------------------
  private renderSearch(): Node {
    const input = h("input", {
      type: "text",
      placeholder: `Search ${this.chunks.length} chunks across ${this.docNames.length} document${this.docNames.length === 1 ? "" : "s"}…`,
      value: this.query,
      oninput: (e: Event) => this.onQueryInput((e.target as HTMLInputElement).value),
    }) as HTMLInputElement;
    this.refs.searchInput = input;

    const benchSwitch = h("div", { class: `switch ${this.benchmarkOn ? "on" : ""}` });
    this.refs.benchSwitch = benchSwitch;

    const latencyChip = h("div", { class: "latency-chip mono" }, "—");
    this.refs.latencyChip = latencyChip;

    const resultsList = h("div", { class: "results-list" }, this.renderEmptyState());
    this.refs.resultsList = resultsList;

    const resultsSection = h("div", { class: "fade-section" }, resultsList);
    this.refs.resultsSection = resultsSection;

    const mapCanvas = h("canvas", {}) as HTMLCanvasElement;
    this.refs.mapCanvas = mapCanvas;
    const mapPreview = h("div", { class: "map-preview" });
    this.refs.mapPreview = mapPreview;
    const mapSection = h(
      "div",
      { class: "map-view hidden" },
      h(
        "div",
        { class: "map-canvas-wrap glass" },
        mapCanvas,
        h("div", { class: "map-hint mono" }, "scroll to zoom · drag to pan · click a point"),
        mapPreview,
      ),
    );
    this.refs.mapSection = mapSection;

    const tabResults = h("button", { class: "view-tab active", onclick: () => this.showTab("results") }, "Results");
    const tabMap = h("button", { class: "view-tab", onclick: () => this.showTab("map") }, "Embedding map");
    const tabBench = h("button", { class: "view-tab", onclick: () => this.showTab("bench") }, "Benchmark");
    this.refs.tabResults = tabResults;
    this.refs.tabMap = tabMap;
    this.refs.tabBench = tabBench;

    const benchBody = h("div", {}, this.renderBenchIntro());
    this.refs.benchBody = benchBody;
    const benchSection = h("div", { class: "fade-section hidden" }, benchBody);
    this.refs.benchSection = benchSection;

    const corpusStrip = h("div", { class: "corpus-strip" }, ...this.docNames.map((n) => h("span", { class: "doc-chip" }, h("div", { html: icons.doc, style: "width:11px;height:11px" }), n)));
    this.refs.corpusStrip = corpusStrip;

    return h(
      "div",
      { class: "search-view" },
      h(
        "div",
        { class: "search-bar-wrap" },
        h(
          "div",
          { class: "search-bar glass" },
          h("div", { html: icons.search, style: "width:17px;height:17px;color:var(--text-tertiary);flex-shrink:0;" }),
          input,
          h(
            "button",
            { class: "btn btn-primary btn-sm", onclick: () => this.runSearch(input.value) },
            "Search",
          ),
        ),
        h(
          "div",
          { class: "search-meta-row" },
          h(
            "div",
            { class: "search-toggles" },
            h(
              "label",
              { class: "toggle", onclick: (e: Event) => { e.preventDefault(); this.toggleBenchmark(); } },
              benchSwitch,
              "GPU vs CPU",
            ),
            h(
              "button",
              { class: "btn btn-ghost btn-sm", onclick: () => this.addMoreDocuments() },
              "+ Add documents",
            ),
            h(
              "button",
              { class: "btn btn-ghost btn-sm", onclick: () => this.resetAll() },
              "Start over",
            ),
          ),
          latencyChip,
        ),
        corpusStrip,
      ),
      h("div", { class: "view-tabs" }, tabResults, tabMap, tabBench),
      resultsSection,
      mapSection,
      benchSection,
    );
  }

  private renderEmptyState(): Node {
    return h(
      "div",
      { class: "empty-state" },
      h("h3", {}, "Type a query to search on your GPU"),
      h("p", {}, "Every keystroke re-runs the full WGSL pipeline — lookup, pooling, similarity, and top-k."),
    );
  }

  private onQueryInput(value: string) {
    this.query = value;
    window.clearTimeout(this._debounce);
    this._debounce = window.setTimeout(() => this.runSearch(value), 260);
  }
  private _debounce = 0;

  private async runSearch(query: string) {
    this.query = query;
    const token = ++this.searchToken;
    if (!this.pipeline || !this.model || !this.chunkEmbeddingsBuf || this.chunks.length === 0 || !query.trim()) {
      this.results = [];
      this.lastGpuMs = null;
      this.lastCpuMs = null;
      this.patchResults();
      this.patchLatency();
      return;
    }

    const { ids } = tokenize(query, this.model);
    if (ids.length === 0) {
      this.results = [];
      this.patchResults();
      return;
    }

    const t0 = performance.now();
    const queryEmbed = await this.pipeline.embedBatch([ids]);
    const searchResult = await this.pipeline.search(queryEmbed.embeddings, this.chunkEmbeddingsBuf, this.chunks.length, K_RESULTS);
    const gpuMs = performance.now() - t0;
    queryEmbed.buffer.destroy();

    if (token !== this.searchToken) return; // a newer query superseded this one

    this.lastTimings = [...queryEmbed.timings, ...searchResult.timings];
    this.lastDispatches = this.pipeline.lastDispatches;
    this.lastOpLabel = `search "${query}"`;
    this.lastGpuMs = gpuMs;

    this.results = Array.from(searchResult.indices)
      .map((idx, i) => ({ chunk: this.chunks[idx], score: searchResult.scores[i] }))
      .filter((r) => r.chunk);

    if (this.benchmarkOn) {
      const cT0 = performance.now();
      const cpuQueryEmbedding = embedBatchCPU([ids], this.model);
      searchCPU(cpuQueryEmbedding, this.chunkEmbeddings, this.chunks.length, this.model.hiddenDim, K_RESULTS);
      this.lastCpuMs = performance.now() - cT0;
    } else {
      this.lastCpuMs = null;
    }

    this.patchResults();
    this.patchLatency();
    this.patchDrawer();
  }

  private patchResults() {
    const list = this.refs.resultsList;
    if (!list) return;
    if (this.results.length === 0) {
      list.replaceChildren(this.renderEmptyState());
      return;
    }
    list.replaceChildren(
      ...this.results.map((r, i) =>
        h(
          "div",
          { class: "result-card glass", style: `animation-delay:${i * 28}ms` },
          h(
            "div",
            { class: "result-head" },
            h("span", { class: "result-doc" }, r.chunk.docName),
            h("span", { class: "result-score mono" }, `cos ${r.score.toFixed(3)}`),
          ),
          h("p", { class: "result-text", html: highlight(r.chunk.text, this.query) }),
        ),
      ),
    );
  }

  private patchLatency() {
    const chip = this.refs.latencyChip;
    if (!chip) return;
    if (this.lastGpuMs == null) {
      chip.replaceChildren("—");
      return;
    }
    const parts: Node[] = [h("span", {}, "GPU "), h("b", {}, `${this.lastGpuMs.toFixed(1)}ms`)];
    if (this.lastCpuMs != null) {
      const speedup = this.lastCpuMs / Math.max(this.lastGpuMs, 0.001);
      parts.push(
        h("span", {}, " · CPU "),
        h("b", {}, `${this.lastCpuMs.toFixed(1)}ms`),
        h("span", { class: "delta" }, ` · ${speedup.toFixed(1)}× faster`),
      );
    }
    chip.replaceChildren(...parts);
  }

  private toggleBenchmark() {
    this.benchmarkOn = !this.benchmarkOn;
    this.refs.benchSwitch?.classList.toggle("on", this.benchmarkOn);
    if (this.query.trim()) this.runSearch(this.query);
  }

  private showTab(tab: Tab) {
    this.activeTab = tab;
    this.refs.tabResults?.classList.toggle("active", tab === "results");
    this.refs.tabMap?.classList.toggle("active", tab === "map");
    this.refs.tabBench?.classList.toggle("active", tab === "bench");
    this.refs.resultsSection?.classList.toggle("hidden", tab !== "results");
    this.refs.mapSection?.classList.toggle("hidden", tab !== "map");
    this.refs.benchSection?.classList.toggle("hidden", tab !== "bench");
    if (tab === "map") this.mountScatter();
  }

  // -------------------------------------------------------------------
  // Benchmark sweep: the same GPU-vs-CPU comparison as the live toggle,
  // but at synthesized corpus sizes (100 / 1,000 / 10,000 chunks) so the
  // GPU's real advantage is visible past its fixed per-dispatch overhead
  // -- at the demo corpus's native size (a handful of chunks) that
  // overhead can make the CPU path look faster, which is real and worth
  // showing honestly, not hiding.
  private renderBenchIntro(): Node {
    return h(
      "div",
      { style: "padding:8px 4px 30px;" },
      h("p", { style: "color:var(--text-secondary);font-size:13.5px;line-height:1.6;max-width:640px;margin-bottom:18px;" },
        "The live “GPU vs CPU” toggle above times exactly what's on screen — at only a handful of chunks, fixed GPU dispatch/readback overhead can outweigh the actual compute, and the CPU path can legitimately win. This sweep synthesizes larger corpora from your indexed chunks to show where that crosses over."),
      h(
        "button",
        { class: "btn btn-primary btn-sm", onclick: () => this.runSweep() },
        this.benchRunning ? "Running…" : "Run sweep (100 / 1,000 / 10,000 chunks)",
      ),
    );
  }

  private async runSweep() {
    if (!this.pipeline || !this.model || this.chunks.length === 0 || this.benchRunning) return;
    this.benchRunning = true;
    this.benchResults = [];
    this.patchBench();

    const baseChunks = this.chunks.map((c) => c.tokenIds);
    const { ids: queryIds } = tokenize(this.query.trim() || "semantic search on the gpu", this.model);

    await runBenchmarkSweep(this.pipeline, this.model, baseChunks, queryIds, [100, 1000, 10000], K_RESULTS, (point) => {
      this.benchResults.push(point);
      this.patchBench();
    });

    this.benchRunning = false;
    this.patchBench();
  }

  private patchBench() {
    const body = this.refs.benchBody;
    if (!body) return;
    const rows = this.benchResults.map((p) =>
      h(
        "tr",
        {},
        h("td", {}, p.corpusSize.toLocaleString()),
        h("td", {}, `${p.gpuTotalMs.toFixed(2)}ms`),
        h("td", {}, `${p.cpuTotalMs.toFixed(2)}ms`),
        h("td", { style: p.speedup >= 1 ? "color:var(--success)" : "color:var(--warning)" }, `${p.speedup.toFixed(2)}×`),
      ),
    );
    const children: Node[] = [this.renderBenchIntro()];
    if (this.benchResults.length > 0) {
      children.push(
        h(
          "table",
          { class: "kernel-table", style: "max-width:520px;" },
          h("thead", {}, h("tr", {}, h("th", {}, "Corpus size"), h("th", {}, "GPU total"), h("th", {}, "CPU total"), h("th", {}, "Speedup"))),
          h("tbody", {}, ...rows),
        ),
      );
    }
    body.replaceChildren(...children);
  }

  private async mountScatter() {
    if (!this.refs.mapCanvas || !this.pipeline || !this.model || this.chunks.length === 0) return;

    if (this.mapDirty) {
      const axes = findTop2Axes(this.chunkEmbeddings, this.chunks.length, this.model.hiddenDim);
      const { points } = await this.pipeline.project2D(this.chunkEmbeddingsBuf!, this.chunks.length, axes);
      const clusters = kmeans(this.chunkEmbeddings, this.chunks.length, this.model.hiddenDim, 7);
      this.mapPoints = Array.from({ length: this.chunks.length }, (_, i) => ({
        x: points[i * 2], y: points[i * 2 + 1], cluster: clusters[i],
      }));
      this.mapDirty = false;
    }

    if (!this.scatterRenderer) {
      this.scatterRenderer = new ScatterRenderer(this.refs.mapCanvas);
      this.scatterRenderer.setOnPick((index) => this.showMapPreview(index));
    }
    this.scatterRenderer.setPoints(this.mapPoints);
  }

  private showMapPreview(index: number) {
    const chunk = this.chunks[index];
    const preview = this.refs.mapPreview;
    if (!chunk || !preview) return;
    this.scatterRenderer?.setSelected(index);
    preview.replaceChildren(
      h("div", { class: "doc" }, chunk.docName),
      h("p", {}, chunk.text.length > 220 ? chunk.text.slice(0, 220) + "…" : chunk.text),
    );
    preview.classList.add("visible");
  }

  private addMoreDocuments() {
    const input = h("input", {
      type: "file",
      multiple: true,
      accept: ".txt,.md,.markdown,.pdf",
      onchange: async (e: Event) => {
        const files = (e.target as HTMLInputElement).files;
        if (!files?.length) return;
        const docs: DocInput[] = [];
        for (const f of Array.from(files)) docs.push(await loadFile(f));
        this.ingestDocuments(docs);
      },
    }) as HTMLInputElement;
    input.click();
  }

  private async resetAll() {
    await clearAll();
    this.chunks = [];
    this.chunkEmbeddings = new Float32Array(0);
    this.chunkEmbeddingsBuf?.destroy();
    this.chunkEmbeddingsBuf = null;
    this.docNames = [];
    this.results = [];
    this.query = "";
    this.mapDirty = true;
    this.scatterRenderer?.dispose();
    this.scatterRenderer = null;
    this.phase = "landing";
    this.render();
  }

  // -------------------------------------------------------------------
  // GPU internals drawer (built once per shell, toggled via classes)
  // -------------------------------------------------------------------
  private renderDrawerShell(): Node {
    const body = h("div", { class: "drawer-body" });
    this.refs.drawerBody = body;
    const scrim = h("div", { class: "drawer-scrim", onclick: () => this.closeDrawer() });
    const drawer = h(
      "aside",
      { class: "drawer" },
      h(
        "div",
        { class: "drawer-header" },
        h("h3", {}, "GPU internals"),
        h("button", { class: "close-btn", "aria-label": "Close GPU internals panel", onclick: () => this.closeDrawer() }, h("div", { html: icons.close, style: "width:15px;height:15px" })),
      ),
      body,
    );
    this.refs.drawer = drawer;
    this.refs.drawerScrim = scrim;
    return h("div", {}, scrim, drawer);
  }

  private openDrawer() {
    this.drawerOpen = true;
    this.refs.drawer?.classList.add("open");
    this.refs.drawerScrim?.classList.add("open");
    this.patchDrawer();
  }
  private closeDrawer() {
    this.drawerOpen = false;
    this.refs.drawer?.classList.remove("open");
    this.refs.drawerScrim?.classList.remove("open");
  }

  private patchDrawer() {
    const body = this.refs.drawerBody;
    if (!body) return;

    const info = this.capability?.adapterInfo;
    const limits = this.capability?.limits || {};
    const modelBytes = this.pipeline?.modelBufferBytes ?? 0;
    const chunkBytes = this.chunks.length * (this.model?.hiddenDim ?? 0) * 4;
    const maxTiming = Math.max(1, ...this.lastTimings.map((t) => t.gpuMicroseconds ?? 0));

    body.replaceChildren(
      h(
        "div",
        { class: "drawer-section" },
        h("h4", {}, "Device"),
        h(
          "dl",
          { class: "kv-grid" },
          h("dt", {}, "Vendor"), h("dd", {}, info?.vendor || "—"),
          h("dt", {}, "Architecture"), h("dd", {}, info?.architecture || "—"),
          h("dt", {}, "Description"), h("dd", { style: "text-align:right;max-width:220px;" }, info?.description || "—"),
          h("dt", {}, "Timestamp queries"), h("dd", {}, this.capability?.timestampQuerySupported ? "supported" : "unavailable"),
          h("dt", {}, "Max storage buffer"), h("dd", {}, formatBytes(limits.maxStorageBufferBindingSize || 0)),
        ),
      ),
      h(
        "div",
        { class: "drawer-section" },
        h("h4", {}, "Memory in use"),
        h(
          "dl",
          { class: "kv-grid" },
          h("dt", {}, "Model weights (INT8)"), h("dd", {}, formatBytes(modelBytes)),
          h("dt", {}, "Chunk embeddings"), h("dd", {}, formatBytes(chunkBytes)),
          h("dt", {}, "Chunks indexed"), h("dd", {}, `${this.chunks.length}`),
          h("dt", {}, "Vocabulary size"), h("dd", {}, `${this.model?.vocabSize ?? 0}`),
        ),
      ),
      h(
        "div",
        { class: "drawer-section" },
        h("h4", {}, this.lastOpLabel ? `Last op: ${this.lastOpLabel}` : "Per-kernel GPU timing"),
        this.lastTimings.length === 0
          ? h("p", { style: "color:var(--text-tertiary);font-size:12.5px;" }, "Run a search to see live kernel timings.")
          : h(
              "table",
              { class: "kernel-table" },
              h("thead", {}, h("tr", {}, h("th", {}, "Kernel"), h("th", {}, "GPU time"))),
              h(
                "tbody",
                {},
                ...this.lastTimings.map((t) =>
                  h(
                    "tr",
                    {},
                    h("td", {}, t.label),
                    h(
                      "td",
                      { class: "bar-cell" },
                      t.gpuMicroseconds != null ? `${t.gpuMicroseconds.toFixed(1)}µs` : "n/a",
                      t.gpuMicroseconds != null &&
                        h("div", { class: "kernel-bar", style: `width:${Math.max(4, (t.gpuMicroseconds / maxTiming) * 100)}%` }),
                    ),
                  ),
                ),
              ),
            ),
      ),
      h(
        "div",
        { class: "drawer-section" },
        h("h4", {}, "Dispatches"),
        this.lastDispatches.length === 0
          ? h("p", { style: "color:var(--text-tertiary);font-size:12.5px;" }, "—")
          : h(
              "dl",
              { class: "kv-grid" },
              ...this.lastDispatches.flatMap((d) => [
                h("dt", {}, d.kernel),
                h("dd", {}, `${d.workgroupCount}×wg(${d.workgroupSize})`),
              ]),
            ),
      ),
    );
  }

  // -------------------------------------------------------------------
  // Toasts
  // -------------------------------------------------------------------
  private toast(text: string, tone: "ok" | "info" = "info") {
    const id = Date.now() + Math.random();
    this.toasts.push({ id, text, tone });
    this.renderToastStack();
    setTimeout(() => {
      this.toasts = this.toasts.filter((t) => t.id !== id);
      this.renderToastStack();
    }, 3400);
  }

  private renderToast(t: { id: number; text: string; tone: "ok" | "info" }): Node {
    return h(
      "div",
      { class: "toast glass", style: t.tone === "ok" ? "color:var(--success)" : "" },
      t.text,
    );
  }

  private renderToastStack() {
    const stack = document.querySelector(".toast-stack");
    if (!stack) return;
    stack.replaceChildren(...this.toasts.map((t) => this.renderToast(t)));
  }

  // Exposes internals on `window` for scripts/correctness-check.mjs to
  // drive from outside the app (real GPU-vs-CPU numerical validation,
  // documented in the top-level README). Not used by the UI itself.
  private exposeDebugHook() {
    (window as unknown as Record<string, unknown>).__browsermind_debug__ = {
      pipeline: this.pipeline,
      model: this.model,
      chunks: this.chunks,
      chunkEmbeddings: this.chunkEmbeddings,
      chunkEmbeddingsBuf: this.chunkEmbeddingsBuf,
    };
  }
}

function formatBytes(n: number): string {
  if (n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 2)} ${units[i]}`;
}
