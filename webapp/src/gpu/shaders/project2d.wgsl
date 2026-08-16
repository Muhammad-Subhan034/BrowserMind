// Bonus kernel: batched projection of every chunk embedding onto a 2D
// basis, used to lay out the embedding-space scatter plot the GLSL
// renderer draws. The 2D basis itself (the top two principal directions)
// is found once via a short CPU-side power iteration over the much
// smaller hidden_dim x hidden_dim Gram matrix (see viz/layout.ts) -- but
// the O(num_chunks * hidden_dim) work of actually projecting every chunk
// onto that basis runs here, on the GPU, exactly like similarity.wgsl.

const HIDDEN_DIM: u32 = __HIDDEN_DIM__;

struct Params {
  hidden_dim: u32,
  num_chunks: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> chunk_embeddings: array<f32>; // num_chunks * hidden_dim
@group(0) @binding(2) var<storage, read> axes: array<f32>;             // 2 * hidden_dim (axis0 then axis1)
@group(0) @binding(3) var<storage, read_write> points: array<f32>;     // num_chunks * 2

const WORKGROUP_SIZE: u32 = 64u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let chunk_idx = gid.x;
  if (chunk_idx >= params.num_chunks) {
    return;
  }

  let base = chunk_idx * HIDDEN_DIM;
  var x: f32 = 0.0;
  var y: f32 = 0.0;
  for (var d: u32 = 0u; d < HIDDEN_DIM; d = d + 1u) {
    let v = chunk_embeddings[base + d];
    x = x + v * axes[d];
    y = y + v * axes[HIDDEN_DIM + d];
  }

  points[chunk_idx * 2u] = x;
  points[chunk_idx * 2u + 1u] = y;
}
