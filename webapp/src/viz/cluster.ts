// Small, dependency-free k-means, run on the full chunk embeddings (not
// the 2D projection) so cluster color reflects real semantic similarity
// rather than just proximity after a lossy 2D projection.

export function kmeans(embeddings: Float32Array, numItems: number, hiddenDim: number, k: number, iters = 15): Int32Array {
  if (numItems === 0) return new Int32Array(0);
  const clampedK = Math.max(1, Math.min(k, numItems));

  const centroids = new Float32Array(clampedK * hiddenDim);
  const step = Math.floor(numItems / clampedK) || 1;
  for (let c = 0; c < clampedK; c++) {
    const src = Math.min(c * step, numItems - 1);
    centroids.set(embeddings.subarray(src * hiddenDim, src * hiddenDim + hiddenDim), c * hiddenDim);
  }

  const assignments = new Int32Array(numItems);

  for (let iter = 0; iter < iters; iter++) {
    for (let i = 0; i < numItems; i++) {
      const base = i * hiddenDim;
      let best = 0;
      let bestDist = Infinity;
      for (let c = 0; c < clampedK; c++) {
        const cBase = c * hiddenDim;
        let dist = 0;
        for (let d = 0; d < hiddenDim; d++) {
          const diff = embeddings[base + d] - centroids[cBase + d];
          dist += diff * diff;
        }
        if (dist < bestDist) { bestDist = dist; best = c; }
      }
      assignments[i] = best;
    }

    const sums = new Float32Array(clampedK * hiddenDim);
    const counts = new Int32Array(clampedK);
    for (let i = 0; i < numItems; i++) {
      const c = assignments[i];
      counts[c]++;
      const base = i * hiddenDim;
      const cBase = c * hiddenDim;
      for (let d = 0; d < hiddenDim; d++) sums[cBase + d] += embeddings[base + d];
    }
    for (let c = 0; c < clampedK; c++) {
      if (counts[c] === 0) continue;
      const cBase = c * hiddenDim;
      for (let d = 0; d < hiddenDim; d++) centroids[cBase + d] = sums[cBase + d] / counts[c];
    }
  }

  return assignments;
}
