// Benchmark harness backing the "GPU vs CPU" toggle and the corpus-size
// sweep used to produce the README's benchmark table. Every number here is
// a real `performance.now()` measurement around real work -- no simulated
// or hand-typed timings.

import type { GpuPipeline } from "../gpu/pipeline";
import type { BMindModel } from "../gpu/modelFormat";
import { embedBatchCPU, searchCPU } from "../cpu/fallback";

export interface BenchmarkPoint {
  corpusSize: number;
  gpuEmbedMs: number;
  cpuEmbedMs: number;
  gpuSearchMs: number;
  cpuSearchMs: number;
  gpuTotalMs: number;
  cpuTotalMs: number;
  speedup: number;
}

/** Duplicates/resamples existing token-id sequences to synthesize a corpus of the requested size for stress-testing. */
export function synthesizeCorpus(baseChunks: number[][], targetSize: number): number[][] {
  if (baseChunks.length === 0) return [];
  const out: number[][] = [];
  for (let i = 0; i < targetSize; i++) {
    out.push(baseChunks[i % baseChunks.length]);
  }
  return out;
}

export async function benchmarkAtSize(
  pipeline: GpuPipeline,
  model: BMindModel,
  corpusTokenIds: number[][],
  queryTokenIds: number[],
  k: number,
): Promise<BenchmarkPoint> {
  const corpusSize = corpusTokenIds.length;

  const gpuEmbedT0 = performance.now();
  const gpuEmbed = await pipeline.embedBatch(corpusTokenIds);
  const gpuEmbedMs = performance.now() - gpuEmbedT0;

  const cpuEmbedT0 = performance.now();
  const cpuChunkEmbeddings = embedBatchCPU(corpusTokenIds, model);
  const cpuEmbedMs = performance.now() - cpuEmbedT0;

  const gpuQueryT0 = performance.now();
  const gpuQueryEmbed = await pipeline.embedBatch([queryTokenIds]);
  const gpuQueryMs = performance.now() - gpuQueryT0;

  const cpuQueryT0 = performance.now();
  const cpuQueryEmbedding = embedBatchCPU([queryTokenIds], model);
  const cpuQueryMs = performance.now() - cpuQueryT0;

  const gpuSearchT0 = performance.now();
  await pipeline.search(gpuQueryEmbed.embeddings, gpuEmbed.buffer, corpusSize, k);
  const gpuSearchMs = performance.now() - gpuSearchT0;

  const cpuSearchT0 = performance.now();
  searchCPU(cpuQueryEmbedding, cpuChunkEmbeddings, corpusSize, model.hiddenDim, k);
  const cpuSearchMs = performance.now() - cpuSearchT0;

  gpuEmbed.buffer.destroy();

  const gpuTotalMs = gpuEmbedMs + gpuQueryMs + gpuSearchMs;
  const cpuTotalMs = cpuEmbedMs + cpuQueryMs + cpuSearchMs;

  return {
    corpusSize,
    gpuEmbedMs: gpuEmbedMs + gpuQueryMs,
    cpuEmbedMs: cpuEmbedMs + cpuQueryMs,
    gpuSearchMs,
    cpuSearchMs,
    gpuTotalMs,
    cpuTotalMs,
    speedup: gpuTotalMs > 0 ? cpuTotalMs / gpuTotalMs : 0,
  };
}

export async function runBenchmarkSweep(
  pipeline: GpuPipeline,
  model: BMindModel,
  baseChunks: number[][],
  queryTokenIds: number[],
  sizes: number[] = [100, 1000, 10000],
  k = 10,
  onProgress?: (point: BenchmarkPoint) => void,
): Promise<BenchmarkPoint[]> {
  // Integrated GPUs ramp clock/power state up under sustained load, and the
  // very first WebGPU dispatch of a session pays extra driver setup cost --
  // both effects would otherwise leak into whichever size runs first and
  // make the sweep non-monotonic for reasons that have nothing to do with
  // corpus size. One throwaway warm-up run (discarded) absorbs that cost so
  // the recorded numbers are comparing like with like.
  if (baseChunks.length > 0) {
    await benchmarkAtSize(pipeline, model, synthesizeCorpus(baseChunks, Math.min(50, baseChunks.length * 4)), queryTokenIds, k);
  }

  const results: BenchmarkPoint[] = [];
  for (const size of sizes) {
    const corpus = synthesizeCorpus(baseChunks, size);
    const point = await benchmarkAtSize(pipeline, model, corpus, queryTokenIds, k);
    results.push(point);
    onProgress?.(point);
  }
  return results;
}
