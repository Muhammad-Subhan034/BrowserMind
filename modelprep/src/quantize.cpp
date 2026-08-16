#include "quantize.hpp"

#include <algorithm>
#include <cmath>

namespace modelprep {

QuantizedMatrix quantize_int8_per_row(const float* fp32, uint32_t rows, uint32_t cols) {
    QuantizedMatrix qm;
    qm.rows = rows;
    qm.cols = cols;
    qm.data.resize(static_cast<size_t>(rows) * cols);
    qm.row_scales.resize(rows);

    constexpr float kInt8Max = 127.0f;

    for (uint32_t r = 0; r < rows; ++r) {
        const float* row = fp32 + static_cast<size_t>(r) * cols;
        float max_abs = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            max_abs = std::max(max_abs, std::fabs(row[c]));
        }

        // A near-zero row (shouldn't happen for a trained embedding table,
        // but guard anyway) would otherwise produce a division blow-up.
        float scale = (max_abs > 1e-8f) ? (max_abs / kInt8Max) : 0.0f;
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        qm.row_scales[r] = scale;

        int8_t* out_row = qm.data.data() + static_cast<size_t>(r) * cols;
        for (uint32_t c = 0; c < cols; ++c) {
            float scaled = row[c] * inv_scale;
            float rounded = std::round(scaled);
            rounded = std::clamp(rounded, -kInt8Max, kInt8Max);
            out_row[c] = static_cast<int8_t>(rounded);
        }
    }

    return qm;
}

void dequantize_row(const QuantizedMatrix& qm, uint32_t row, float* out) {
    const int8_t* in_row = qm.data.data() + static_cast<size_t>(row) * qm.cols;
    float scale = qm.row_scales[row];
    for (uint32_t c = 0; c < qm.cols; ++c) {
        out[c] = static_cast<float>(in_row[c]) * scale;
    }
}

}  // namespace modelprep
