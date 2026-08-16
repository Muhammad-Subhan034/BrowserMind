// Kernel 3/4: Batched cosine similarity.
//
// Embarrassingly parallel across chunks: one thread computes one full
// 384-dimension dot product (and both norms) between the query embedding
// and its assigned chunk embedding. With thousands of chunks and hundreds
// of dimensions each, this alone is tens to hundreds of thousands of
// multiply-adds dispatched as a single GPU pass -- the kind of batched
// vector-similarity workload GPUs are built for.

const HIDDEN_DIM: u32 = __HIDDEN_DIM__;

struct Params {
  hidden_dim: u32,
  num_chunks: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query_embedding: array<f32>;    // hidden_dim
@group(0) @binding(2) var<storage, read> chunk_embeddings: array<f32>;   // num_chunks * hidden_dim
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;       // num_chunks

const WORKGROUP_SIZE: u32 = 64u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let chunk_idx = gid.x;
  if (chunk_idx >= params.num_chunks) {
    return;
  }

  let base = chunk_idx * HIDDEN_DIM;
  var dot: f32 = 0.0;
  var norm_q: f32 = 0.0;
  var norm_c: f32 = 0.0;

  for (var d: u32 = 0u; d < HIDDEN_DIM; d = d + 1u) {
    let q = query_embedding[d];
    let c = chunk_embeddings[base + d];
    dot = dot + q * c;
    norm_q = norm_q + q * q;
    norm_c = norm_c + c * c;
  }

  let denom = sqrt(norm_q) * sqrt(norm_c);
  scores[chunk_idx] = select(0.0, dot / denom, denom > 1e-12);
}
