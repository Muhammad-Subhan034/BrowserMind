#include "validate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace modelprep {

namespace {

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0f;
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

}  // namespace

std::vector<float> embed_quantized(const std::vector<uint32_t>& token_ids,
                                    const QuantizedMatrix& quantized,
                                    const std::vector<float>& sif_weights,
                                    const std::vector<float>& pc_component,
                                    uint32_t pc_dim,
                                    uint32_t hidden_dim) {
    std::vector<float> accum(hidden_dim, 0.0f);
    std::vector<float> row(hidden_dim, 0.0f);
    float weight_sum = 0.0f;

    for (uint32_t tid : token_ids) {
        if (tid >= quantized.rows) continue;  // out-of-vocab guard
        float w = sif_weights[tid];
        if (w <= 0.0f) continue;
        dequantize_row(quantized, tid, row.data());
        for (uint32_t d = 0; d < hidden_dim; ++d) accum[d] += row[d] * w;
        weight_sum += w;
    }

    if (weight_sum > 1e-8f) {
        for (uint32_t d = 0; d < hidden_dim; ++d) accum[d] /= weight_sum;
    }

    for (uint32_t p = 0; p < pc_dim; ++p) {
        const float* pc = pc_component.data() + static_cast<size_t>(p) * hidden_dim;
        double dot = 0.0;
        for (uint32_t d = 0; d < hidden_dim; ++d) dot += static_cast<double>(accum[d]) * pc[d];
        for (uint32_t d = 0; d < hidden_dim; ++d) accum[d] -= static_cast<float>(dot) * pc[d];
    }

    return accum;
}

ValidationReport run_validation(const std::vector<ValidationExample>& examples,
                                 const QuantizedMatrix& quantized,
                                 const std::vector<float>& sif_weights,
                                 const std::vector<float>& pc_component,
                                 uint32_t pc_dim,
                                 uint32_t hidden_dim) {
    ValidationReport report;
    report.per_example.reserve(examples.size());

    float min_c = std::numeric_limits<float>::max();
    float max_c = std::numeric_limits<float>::lowest();
    double sum_c = 0.0;

    for (const auto& ex : examples) {
        auto reconstructed = embed_quantized(ex.token_ids, quantized, sif_weights, pc_component, pc_dim, hidden_dim);
        float cos = cosine_similarity(reconstructed, ex.reference_embedding);
        report.per_example.push_back({ex.text, cos});
        min_c = std::min(min_c, cos);
        max_c = std::max(max_c, cos);
        sum_c += cos;
    }

    report.min_cosine = examples.empty() ? 0.0f : min_c;
    report.max_cosine = examples.empty() ? 0.0f : max_c;
    report.mean_cosine = examples.empty() ? 0.0f : static_cast<float>(sum_c / examples.size());
    return report;
}

}  // namespace modelprep
