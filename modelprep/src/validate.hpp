// Re-runs the SIF embedding pipeline using the *quantized* weights and
// compares the result, sentence by sentence, against the fp32 reference
// embeddings computed in Python (export_embeddings.py). This is the "C++
// validates the quantized output numerically against a Python reference"
// deliverable: it is not a unit test of quantize.cpp in isolation, it is an
// end-to-end numerical check of exactly the pipeline the browser will run.
#pragma once

#include <string>
#include <vector>

#include "quantize.hpp"
#include "raw_io.hpp"

namespace modelprep {

struct ValidationResult {
    std::string text;
    float cosine_similarity = 0.0f;
};

struct ValidationReport {
    std::vector<ValidationResult> per_example;
    float min_cosine = 0.0f;
    float mean_cosine = 0.0f;
    float max_cosine = 0.0f;
};

// Reconstructs a chunk embedding from token ids using the quantized matrix,
// mirroring exactly what the WGSL pooling kernel will do on the GPU:
// dequantize each token row, weight it by that token's SIF weight, average,
// then subtract its projection onto the shared top principal component.
std::vector<float> embed_quantized(const std::vector<uint32_t>& token_ids,
                                    const QuantizedMatrix& quantized,
                                    const std::vector<float>& sif_weights,
                                    const std::vector<float>& pc_component,
                                    uint32_t pc_dim,
                                    uint32_t hidden_dim);

ValidationReport run_validation(const std::vector<ValidationExample>& examples,
                                 const QuantizedMatrix& quantized,
                                 const std::vector<float>& sif_weights,
                                 const std::vector<float>& pc_component,
                                 uint32_t pc_dim,
                                 uint32_t hidden_dim);

}  // namespace modelprep
