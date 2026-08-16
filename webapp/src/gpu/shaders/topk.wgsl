// Kernel 4/4: GPU-side top-k selection via repeated tournament reduction.
//
// Scores never come back to the CPU mid-pipeline -- only the final k
// (index, score) pairs do. Each of the k rounds runs a two-stage parallel
// max-reduction ("tournament": every thread pair competes, the winner
// advances) entirely on the GPU:
//
//   stage A (reduce_partial): every workgroup reduces its slice of the
//     scores array down to one (value, index) pair -- a standard
//     shared-memory tree reduction, just tracking an index alongside the
//     max instead of only the value.
//   stage B (reduce_final): a single workgroup reduces the (small) array
//     of per-workgroup partials down to the one global max, records it as
//     round `r`'s result, then masks that index to -inf in the scores
//     buffer so the next round's stage A never picks it again.
//
// This costs O(k * n) work across k rounds, versus O(n log^2 n) for a full
// bitonic top-k network. For the k << n case this app targets (a handful
// of results out of thousands-to-tens-of-thousands of chunks), that
// trade-off wins on both simplicity and total dispatches; a bitonic
// network is the documented next step if k or corpus size grows large
// enough to change that trade-off (see README benchmark notes).

struct Params {
  num_chunks: u32,
  num_workgroups: u32,
  round: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_vals: array<f32>;
@group(0) @binding(3) var<storage, read_write> partial_idxs: array<u32>;
@group(0) @binding(4) var<storage, read_write> topk_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> topk_scores: array<f32>;

const WORKGROUP_SIZE: u32 = 256u;
const NEG_INF: f32 = -3.4e38;

var<workgroup> vals: array<f32, WORKGROUP_SIZE>;
var<workgroup> idxs: array<u32, WORKGROUP_SIZE>;

fn tree_reduce_argmax(lid: u32) {
  var stride: u32 = WORKGROUP_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lid < stride) {
      if (vals[lid + stride] > vals[lid]) {
        vals[lid] = vals[lid + stride];
        idxs[lid] = idxs[lid + stride];
      }
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn reduce_partial(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  if (idx < params.num_chunks) {
    vals[lid.x] = scores[idx];
    idxs[lid.x] = idx;
  } else {
    vals[lid.x] = NEG_INF;
    idxs[lid.x] = 0u;
  }
  workgroupBarrier();

  tree_reduce_argmax(lid.x);

  if (lid.x == 0u) {
    partial_vals[wgid.x] = vals[0];
    partial_idxs[wgid.x] = idxs[0];
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn reduce_final(@builtin(local_invocation_id) lid: vec3<u32>) {
  if (lid.x < params.num_workgroups) {
    vals[lid.x] = partial_vals[lid.x];
    idxs[lid.x] = partial_idxs[lid.x];
  } else {
    vals[lid.x] = NEG_INF;
    idxs[lid.x] = 0u;
  }
  workgroupBarrier();

  tree_reduce_argmax(lid.x);

  if (lid.x == 0u) {
    let winner = idxs[0];
    topk_indices[params.round] = winner;
    topk_scores[params.round] = vals[0];
    scores[winner] = NEG_INF; // remove from contention for the next round
  }
}
