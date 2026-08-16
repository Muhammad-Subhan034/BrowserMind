// Numerically compares the real GPU pipeline's output (embedding_lookup.wgsl
// + pooling.wgsl, actually dispatched and read back from the GPU) against
// the pure-JS reference implementation (cpu/fallback.ts) for the same
// token-id sequences. This is the "WGSL kernels, correctness-tested against
// a CPU/JS reference implementation" checklist item -- run from the browser
// console (or driven headlessly, see scripts/correctness-check.mjs) so the
// numbers in the README are real measurements, not claims.

import type { GpuPipeline } from "../gpu/pipeline";
import type { BMindModel } from "../gpu/modelFormat";
import { embedBatchCPU, searchCPU } from "../cpu/fallback";

export interface CorrectnessReport {
  itemCount: number;
  minCosine: number;
  meanCosine: number;
  maxCosine: number;
}

export interface SearchCorrectnessReport {
  /** Fraction of the GPU top-k set that the CPU reference also ranks in its own top-k (order-independent). */
  topKSetOverlap: number;
  /** Max absolute difference between GPU and CPU cosine scores for indices both selected. */
  maxScoreDelta: number;
}

/** Validates similarity.wgsl + topk.wgsl against searchCPU() on the same chunk embeddings. */
export async function runSearchCorrectnessCheck(
  pipeline: GpuPipeline,
  model: BMindModel,
  chunkEmbeddings: Float32Array,
  chunkEmbeddingsBuf: GPUBuffer,
  numChunks: number,
  queryTokenIds: number[],
  k: number,
): Promise<SearchCorrectnessReport> {
  const queryEmbed = await pipeline.embedBatch([queryTokenIds]);
  const gpuResult = await pipeline.search(queryEmbed.embeddings, chunkEmbeddingsBuf, numChunks, k);
  const cpuQueryEmbedding = embedBatchCPU([queryTokenIds], model);
  const cpuResult = searchCPU(cpuQueryEmbedding, chunkEmbeddings, numChunks, model.hiddenDim, k);
  queryEmbed.buffer.destroy();

  const cpuSet = new Set(Array.from(cpuResult.indices));
  let overlap = 0;
  let maxDelta = 0;
  for (let i = 0; i < gpuResult.indices.length; i++) {
    const idx = gpuResult.indices[i];
    if (cpuSet.has(idx)) {
      overlap++;
      const cpuScoreIdx = Array.from(cpuResult.indices).indexOf(idx);
      maxDelta = Math.max(maxDelta, Math.abs(gpuResult.scores[i] - cpuResult.scores[cpuScoreIdx]));
    }
  }

  return {
    topKSetOverlap: gpuResult.indices.length ? overlap / gpuResult.indices.length : 1,
    maxScoreDelta: maxDelta,
  };
}

function cosine(a: Float32Array, b: Float32Array, offset: number, dim: number): number {
  let dot = 0, na = 0, nb = 0;
  for (let d = 0; d < dim; d++) {
    const av = a[offset + d];
    const bv = b[offset + d];
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 1e-12 ? dot / denom : 1;
}

export async function runCorrectnessCheck(
  pipeline: GpuPipeline,
  model: BMindModel,
  tokenIdsPerItem: number[][],
): Promise<CorrectnessReport> {
  const gpuResult = await pipeline.embedBatch(tokenIdsPerItem);
  gpuResult.buffer.destroy();
  const cpuResult = embedBatchCPU(tokenIdsPerItem, model);

  const dim = model.hiddenDim;
  let min = Infinity, max = -Infinity, sum = 0;
  for (let i = 0; i < tokenIdsPerItem.length; i++) {
    const c = cosine(gpuResult.embeddings, cpuResult, i * dim, dim);
    min = Math.min(min, c);
    max = Math.max(max, c);
    sum += c;
  }
  const n = tokenIdsPerItem.length;
  return { itemCount: n, minCosine: n ? min : 1, meanCosine: n ? sum / n : 1, maxCosine: n ? max : 1 };
}
