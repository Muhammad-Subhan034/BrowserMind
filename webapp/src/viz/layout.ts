// Finds a 2D basis (the top two principal directions) for the embedding
// map, via power iteration over the hidden_dim x hidden_dim Gram matrix
// X^T X. hidden_dim is small (384), so forming and iterating on that
// matrix is cheap even for large corpora -- it's the O(num_chunks x
// hidden_dim) part (actually projecting every point onto the basis) that
// scales with corpus size, and that part runs on the GPU
// (gpu/shaders/project2d.wgsl), not here.

export function findTop2Axes(embeddings: Float32Array, numItems: number, hiddenDim: number): Float32Array {
  if (numItems === 0) return new Float32Array(2 * hiddenDim);

  // Center the data first so the projection reflects variance, not the
  // mean offset all embeddings share.
  const mean = new Float32Array(hiddenDim);
  for (let i = 0; i < numItems; i++) {
    const base = i * hiddenDim;
    for (let d = 0; d < hiddenDim; d++) mean[d] += embeddings[base + d];
  }
  for (let d = 0; d < hiddenDim; d++) mean[d] /= numItems;

  const centered = new Float32Array(numItems * hiddenDim);
  for (let i = 0; i < numItems; i++) {
    const base = i * hiddenDim;
    for (let d = 0; d < hiddenDim; d++) centered[base + d] = embeddings[base + d] - mean[d];
  }

  // Gram matrix G = X^T X (hiddenDim x hiddenDim).
  const gram = new Float64Array(hiddenDim * hiddenDim);
  for (let i = 0; i < numItems; i++) {
    const base = i * hiddenDim;
    for (let a = 0; a < hiddenDim; a++) {
      const va = centered[base + a];
      if (va === 0) continue;
      const rowBase = a * hiddenDim;
      for (let b = a; b < hiddenDim; b++) {
        gram[rowBase + b] += va * centered[base + b];
      }
    }
  }
  for (let a = 0; a < hiddenDim; a++) {
    for (let b = 0; b < a; b++) gram[a * hiddenDim + b] = gram[b * hiddenDim + a];
  }

  const axis0 = powerIteration(gram, hiddenDim, null);
  const axis1 = powerIteration(gram, hiddenDim, axis0);

  const axes = new Float32Array(2 * hiddenDim);
  axes.set(axis0, 0);
  axes.set(axis1, hiddenDim);
  return axes;
}

function matVec(gram: Float64Array, n: number, v: Float64Array, out: Float64Array) {
  for (let a = 0; a < n; a++) {
    let sum = 0;
    const rowBase = a * n;
    for (let b = 0; b < n; b++) sum += gram[rowBase + b] * v[b];
    out[a] = sum;
  }
}

function powerIteration(gram: Float64Array, n: number, deflateAgainst: Float64Array | null, iters = 60): Float64Array {
  let v = new Float64Array(n);
  for (let i = 0; i < n; i++) v[i] = Math.sin(i * 12.9898 + 1) * 0.5 + 0.5001;
  normalize(v);

  const tmp = new Float64Array(n);
  for (let iter = 0; iter < iters; iter++) {
    matVec(gram, n, v, tmp);
    if (deflateAgainst) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += tmp[i] * deflateAgainst[i];
      for (let i = 0; i < n; i++) tmp[i] -= dot * deflateAgainst[i];
    }
    v.set(tmp);
    normalize(v);
  }
  return v;
}

function normalize(v: Float64Array) {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm > 1e-12) for (let i = 0; i < v.length; i++) v[i] /= norm;
}
