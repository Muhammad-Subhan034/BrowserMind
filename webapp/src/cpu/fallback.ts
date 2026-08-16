// Pure-JavaScript reference implementation of the exact same math the WGSL
// kernels compute: dequantize -> SIF-weighted mean pool -> remove top
// principal component -> cosine similarity -> top-k. Two jobs:
//
//   1. Correctness reference during development (embedBatchCPU's output
//      should match GpuPipeline.embedBatch's output to within quantization
//      noise -- see webapp/tests/pipeline.test.ts).
//   2. The CPU path for the on-screen "GPU vs CPU" benchmark toggle, which
//      is the single most convincing piece of evidence for GPU competence
//      the app has: the same real work, run twice, timed honestly.

import type { BMindModel } from "../gpu/modelFormat";

export function embedBatchCPU(tokenIdsPerItem: number[][], model: BMindModel): Float32Array {
  const { hiddenDim, pcDim, quantMatrix, rowScales, sifWeights, pcComponent } = model;
  const out = new Float32Array(tokenIdsPerItem.length * hiddenDim);

  const row = new Float32Array(hiddenDim);
  for (let item = 0; item < tokenIdsPerItem.length; item++) {
    const accum = new Float32Array(hiddenDim);
    let weightSum = 0;

    for (const tokenId of tokenIdsPerItem[item]) {
      const w = sifWeights[tokenId];
      if (w <= 0) continue;
      const scale = rowScales[tokenId];
      const base = tokenId * hiddenDim;
      for (let d = 0; d < hiddenDim; d++) row[d] = quantMatrix[base + d] * scale;
      for (let d = 0; d < hiddenDim; d++) accum[d] += row[d] * w;
      weightSum += w;
    }

    const outBase = item * hiddenDim;
    if (weightSum > 1e-8) {
      for (let d = 0; d < hiddenDim; d++) out[outBase + d] = accum[d] / weightSum;
    }

    for (let p = 0; p < pcDim; p++) {
      const pcBase = p * hiddenDim;
      let dot = 0;
      for (let d = 0; d < hiddenDim; d++) dot += out[outBase + d] * pcComponent[pcBase + d];
      for (let d = 0; d < hiddenDim; d++) out[outBase + d] -= dot * pcComponent[pcBase + d];
    }
  }

  return out;
}

export interface CpuSearchResult {
  indices: Uint32Array;
  scores: Float32Array;
}

export function searchCPU(
  query: Float32Array,
  chunkEmbeddings: Float32Array,
  numChunks: number,
  hiddenDim: number,
  k: number,
): CpuSearchResult {
  const scores = new Float32Array(numChunks);
  let normQ = 0;
  for (let d = 0; d < hiddenDim; d++) normQ += query[d] * query[d];
  normQ = Math.sqrt(normQ);

  for (let c = 0; c < numChunks; c++) {
    const base = c * hiddenDim;
    let dot = 0;
    let normC = 0;
    for (let d = 0; d < hiddenDim; d++) {
      const cv = chunkEmbeddings[base + d];
      dot += query[d] * cv;
      normC += cv * cv;
    }
    normC = Math.sqrt(normC);
    const denom = normQ * normC;
    scores[c] = denom > 1e-12 ? dot / denom : 0;
  }

  const effectiveK = Math.max(0, Math.min(k, numChunks));
  const order = Array.from({ length: numChunks }, (_, i) => i);
  order.sort((a, b) => scores[b] - scores[a]);

  const indices = new Uint32Array(effectiveK);
  const outScores = new Float32Array(effectiveK);
  for (let i = 0; i < effectiveK; i++) {
    indices[i] = order[i];
    outScores[i] = scores[order[i]];
  }
  return { indices, scores: outScores };
}
