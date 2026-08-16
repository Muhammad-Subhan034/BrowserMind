// BrowserMind packed model format (".bmind")
//
// This is the ONLY file the browser ever downloads to get a working
// embedding model. It is written once, here, by modelprep::pack(), and
// parsed on the TypeScript side by webapp/src/ingest/modelFormat.ts -- the
// two implementations must stay byte-for-byte in sync with this layout.
//
// All integers are unsigned 32-bit little-endian. All floats are IEEE-754
// 32-bit little-endian. There is no padding between sections; every section
// size is fully determined by the header fields that precede it.
//
//   offset  field                      type          count
//   ------  -------------------------  ------------  -----------------
//           magic                      char[4]        "BMND"
//           version                    u32            1
//           vocab_size                 u32
//           hidden_dim                 u32
//           pc_dim                     u32
//           unk_id                     u32
//           cls_id                     u32
//           sep_id                     u32
//           pad_id                     u32
//           sif_a                      f32
//           --- vocab section ---
//           vocab_offsets              u32[vocab_size + 1]
//           vocab_blob                 u8[vocab_offsets[vocab_size]]   (UTF-8, concatenated, no separators)
//           vocab_pad                  u8[pad]                          (zero bytes, see below)
//           --- quantized embedding matrix ---
//           quant_matrix               i8[vocab_size * hidden_dim]     (row-major: token_id * hidden_dim + dim)
//           row_scales                 f32[vocab_size]                 (dequant: value ~= i8 * row_scales[row])
//           --- pooling weights ---
//           sif_weights                f32[vocab_size]
//           pc_component               f32[pc_dim * hidden_dim]
//
// Design rationale: every section is either a flat primitive array or a
// byte blob addressed by an offset table, so the browser can slice the
// downloaded ArrayBuffer directly into typed-array views (Int8Array /
// Float32Array / Uint32Array) with zero parsing beyond reading the header --
// this is the "minimal client-side parsing" the format is designed for,
// since those bytes get uploaded almost verbatim into WebGPU storage
// buffers.
//
// `vocab_pad` exists purely for alignment: vocab_blob's length is
// arbitrary (it's concatenated UTF-8 text), but every section after it
// needs to sit on a 4-byte boundary so it can be viewed as a
// Float32Array/Uint32Array directly over the downloaded ArrayBuffer without
// a copy (JS typed arrays with element size > 1 throw if constructed at a
// misaligned byteOffset). pad = (4 - vocab_blob.length % 4) % 4 zero bytes.

#pragma once
#include <cstdint>

namespace bmind {

constexpr char kMagic[4] = {'B', 'M', 'N', 'D'};
constexpr uint32_t kVersion = 1;

struct Header {
    uint32_t version;
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t pc_dim;
    uint32_t unk_id;
    uint32_t cls_id;
    uint32_t sep_id;
    uint32_t pad_id;
    float sif_a;
};

}  // namespace bmind
