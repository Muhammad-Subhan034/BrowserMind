// Kernel 2/4: SIF-weighted mean pooling + common-component removal.
//
// One WORKGROUP per chunk (not one thread) -- each chunk can have up to a
// few hundred tokens x 384 dimensions to reduce, so the dimension axis is
// spread across WORKGROUP_SIZE threads, each thread owning every Nth
// dimension. This is a textbook GPU reduction pattern: partition the
// reduction axis across threads, reduce locally in workgroup-shared
// memory, synchronize with barriers, repeat.
//
// Two reductions happen here:
//   1. Weighted sum -> mean over the chunk's token range (per-dimension,
//      no cross-thread communication needed since each thread's dims are
//      independent).
//   2. Removing the shared top principal component (SIF's second step,
//      Arora et al. 2017): this needs a single scalar dot product across
//      *all* 384 dimensions, so it requires an actual tree reduction in
//      workgroup-shared memory -- the classic "reduce 128 partial sums to
//      one" pattern with a halving stride and a barrier each step.
//
// HIDDEN_DIM is templated in at pipeline-build time (see pipeline.ts) so
// the same shader source works for any embedding model dimension, not
// just MiniLM's 384.

const HIDDEN_DIM: u32 = __HIDDEN_DIM__;
const WORKGROUP_SIZE: u32 = 128u;

struct Params {
  hidden_dim: u32,
  pc_dim: u32,
  num_chunks: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> chunk_offsets: array<u32>;       // num_chunks + 1
@group(0) @binding(2) var<storage, read> token_ids: array<u32>;
@group(0) @binding(3) var<storage, read> sif_weights: array<f32>;
@group(0) @binding(4) var<storage, read> weighted_token_vecs: array<f32>; // from embedding_lookup.wgsl
@group(0) @binding(5) var<storage, read> pc_component: array<f32>;       // pc_dim * hidden_dim
@group(0) @binding(6) var<storage, read_write> chunk_embeddings: array<f32>; // num_chunks * hidden_dim

var<workgroup> pooled: array<f32, HIDDEN_DIM>;
var<workgroup> partial: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let chunk_idx = wgid.x;
  if (chunk_idx >= params.num_chunks) {
    return;
  }

  let token_start = chunk_offsets[chunk_idx];
  let token_end = chunk_offsets[chunk_idx + 1u];

  // --- Step 1: weighted mean over the chunk's tokens, one thread per
  // dimension-lane (each thread owns dims lid, lid+128, lid+256, ...). ---
  var weight_sum: f32 = 0.0;
  for (var t = token_start; t < token_end; t = t + 1u) {
    weight_sum = weight_sum + sif_weights[token_ids[t]];
  }
  let inv_weight = select(0.0, 1.0 / weight_sum, weight_sum > 1e-8);

  var dim = lid.x;
  loop {
    if (dim >= HIDDEN_DIM) { break; }
    var accum: f32 = 0.0;
    for (var t = token_start; t < token_end; t = t + 1u) {
      accum = accum + weighted_token_vecs[t * HIDDEN_DIM + dim];
    }
    pooled[dim] = accum * inv_weight;
    dim = dim + WORKGROUP_SIZE;
  }
  workgroupBarrier();

  // --- Step 2: remove the projection onto each shared principal
  // component, one component at a time (pc_dim is typically 1). ---
  for (var p: u32 = 0u; p < params.pc_dim; p = p + 1u) {
    let pc_base = p * HIDDEN_DIM;

    var local_dot: f32 = 0.0;
    dim = lid.x;
    loop {
      if (dim >= HIDDEN_DIM) { break; }
      local_dot = local_dot + pooled[dim] * pc_component[pc_base + dim];
      dim = dim + WORKGROUP_SIZE;
    }
    partial[lid.x] = local_dot;
    workgroupBarrier();

    var stride: u32 = WORKGROUP_SIZE / 2u;
    loop {
      if (stride == 0u) { break; }
      if (lid.x < stride) {
        partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let dot = partial[0];
    workgroupBarrier();

    dim = lid.x;
    loop {
      if (dim >= HIDDEN_DIM) { break; }
      pooled[dim] = pooled[dim] - dot * pc_component[pc_base + dim];
      dim = dim + WORKGROUP_SIZE;
    }
    workgroupBarrier();
  }

  // --- Write the finished chunk embedding out. ---
  dim = lid.x;
  loop {
    if (dim >= HIDDEN_DIM) { break; }
    chunk_embeddings[chunk_idx * HIDDEN_DIM + dim] = pooled[dim];
    dim = dim + WORKGROUP_SIZE;
  }
}
