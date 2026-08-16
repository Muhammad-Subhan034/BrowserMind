// Drives the running dev server with a real browser, loads the demo
// corpus so a live GpuPipeline + model exist, then calls
// src/dev/correctnessCheck.ts's runCorrectnessCheck() in-page to compare
// actual WGSL kernel output against the CPU/JS reference implementation.
// Prints a real, measured cosine-similarity report -- this is the number
// quoted in the top-level README's "kernel correctness" section.
//
// Usage: npm run dev (in one terminal), then:
//   node scripts/correctness-check.mjs [chromium|chrome|msedge]
import { chromium } from "playwright-core";

const channel = process.argv[2] || "chrome";
const url = process.argv[3] || "http://localhost:5173";

const browser = await chromium.launch({ channel, headless: false });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));

await page.goto(url, { waitUntil: "networkidle" });
await page.waitForSelector(".gpu-pill.ok", { timeout: 15000 });
await page.click("text=Try the demo corpus");
await page.waitForSelector(".search-view", { timeout: 30000 });
await page.waitForFunction(() => !!window.__browsermind_debug__, { timeout: 10000 });

const report = await page.evaluate(async () => {
  const { runCorrectnessCheck, runSearchCorrectnessCheck } = await import("/src/dev/correctnessCheck.ts");
  const dbg = window.__browsermind_debug__;
  const tokenIdsPerItem = dbg.chunks.map((c) => c.tokenIds);
  const embedReport = await runCorrectnessCheck(dbg.pipeline, dbg.model, tokenIdsPerItem);

  // A short, generic query -- any tokenizable string works, this is only
  // exercising similarity.wgsl + topk.wgsl against the already-indexed chunks.
  const { tokenize } = await import("/src/ingest/tokenizer.ts");
  const { ids: queryIds } = tokenize("compute shaders and GPU search", dbg.model);
  const searchReport = await runSearchCorrectnessCheck(
    dbg.pipeline, dbg.model, dbg.chunkEmbeddings, dbg.chunkEmbeddingsBuf, dbg.chunks.length, queryIds, 8,
  );

  return { embedReport, searchReport };
});

console.log("embedding_lookup.wgsl + pooling.wgsl vs CPU reference:", report.embedReport);
console.log("similarity.wgsl + topk.wgsl vs CPU reference:", report.searchReport);
if (errors.length) console.log("page errors:", errors);
await browser.close();
