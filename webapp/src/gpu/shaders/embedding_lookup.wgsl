// Kernel 1/4: Embedding lookup + SIF weighting.
//
// One GPU thread per (token instance, 4-dim quad). Token instances across
// every chunk are flattened into a single ragged array (`token_ids`),
// addressed by `chunk_offsets` (CSR-style), so chunks of different lengths
// need no padding.
//
// The quantized embedding matrix is uploaded as packed u32s -- four int8
// lanes per u32 -- because core WGSL storage buffers have no native 8-bit
// element type. Each thread unpacks its own quad, dequantizes it with that
// token's per-row scale (int8 * scale ~= original fp32, see
// modelprep/src/quantize.cpp), and multiplies by the token's SIF pooling
// weight so the next kernel (pooling.wgsl) only has to sum and divide.
//
// This is the "core matmul-like operation that turns token IDs into
// vectors": a gather (indexed read) from the embedding table followed by a
// per-element scale -- the GPU-parallel form of an embedding lookup.

struct Params {
  vocab_size: u32,
  hidden_dim: u32,
  pc_dim: u32,
  num_token_instances: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read> quant_matrix: array<u32>;   // packed int8x4
@group(0) @binding(3) var<storage, read> row_scales: array<f32>;
@group(0) @binding(4) var<storage, read> sif_weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> weighted_token_vecs: array<f32>;

fn unpack_i8(packed: u32, lane: u32) -> f32 {
  // Each byte is a signed int8 stored little-endian within the u32.
  let shifted = packed >> (lane * 8u);
  let byte = shifted & 0xFFu;
  // Sign-extend the low 8 bits to i32, then convert.
  let signed = (i32(byte) << 24u) >> 24u;
  return f32(signed);
}

const WORKGROUP_SIZE: u32 = 64u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let quads_per_row = params.hidden_dim / 4u;
  let total_quads = params.num_token_instances * quads_per_row;
  let idx = gid.x;
  if (idx >= total_quads) {
    return;
  }

  let token_instance = idx / quads_per_row;
  let quad = idx % quads_per_row;

  let token_id = token_ids[token_instance];
  let weight = sif_weights[token_id];
  let scale = row_scales[token_id];

  let matrix_quad_index = token_id * quads_per_row + quad;
  let packed = quant_matrix[matrix_quad_index];

  let out_base = token_instance * params.hidden_dim + quad * 4u;
  for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {
    let dequantized = unpack_i8(packed, lane) * scale;
    weighted_token_vecs[out_base + lane] = dequantized * weight;
  }
}
