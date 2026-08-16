// Hand-written per-row (per-channel) INT8 quantization for the embedding
// matrix. This is the numerically interesting part of modelprep: rather
// than a single global scale for the whole matrix, each row (one token's
// embedding vector) gets its own scale, computed from that row's own
// max-abs value. Per-row scaling matters a lot here because token
// embedding magnitudes vary widely (frequent, "boring" tokens tend to sit
// closer to the origin than rare, information-dense ones); a single global
// scale would waste most of the int8 range on the common case and clip or
// under-resolve the rest.
#pragma once

#include <cstdint>
#include <vector>

namespace modelprep {

struct QuantizedMatrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<int8_t> data;    // rows * cols, row-major
    std::vector<float> row_scales;  // rows; dequant: value ~= data[i] * row_scales[row]
};

// Quantizes `fp32` (rows*cols, row-major) into signed INT8 with one scale
// per row. Values are scaled so that the largest-magnitude element in each
// row maps to +-127 (int8's symmetric range, avoiding -128 to keep the
// quantization symmetric around zero).
QuantizedMatrix quantize_int8_per_row(const float* fp32, uint32_t rows, uint32_t cols);

// Reconstructs a single row back to fp32 (used by validate.cpp to measure
// quantization error, and mirrors exactly what the WGSL embedding-lookup
// kernel does on the GPU: int8_value * row_scale).
void dequantize_row(const QuantizedMatrix& qm, uint32_t row, float* out);

}  // namespace modelprep
