// Minimal, dependency-free correctness tests for quantize.cpp and
// validate.cpp's embed_quantized(). Deliberately not using a test
// framework: this project has zero external C++ dependencies by design
// (see modelprep/README.md), and these checks are simple enough that a
// framework would add more ceremony than value. Run via `ctest` or
// directly as an executable; any failed check prints and the process
// exits 1.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "quantize.hpp"
#include "validate.hpp"

namespace {

int g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failures;
    } else {
        std::printf("ok: %s\n", msg);
    }
}

void test_quantize_basic() {
    // 2 rows x 4 cols. Row 0 has max-abs 2.0, row 1 has max-abs 0.5.
    std::vector<float> data = {
        1.0f, -2.0f, 0.5f, 0.0f,
        0.25f, -0.5f, 0.1f, -0.1f,
    };
    auto qm = modelprep::quantize_int8_per_row(data.data(), 2, 4);

    check(qm.rows == 2 && qm.cols == 4, "quantized matrix shape matches input");

    // Row 0's max-abs element (-2.0) must quantize to exactly -127.
    check(qm.data[1] == -127, "row 0 max-magnitude element maps to -127");

    std::vector<float> row0(4), row1(4);
    modelprep::dequantize_row(qm, 0, row0.data());
    modelprep::dequantize_row(qm, 1, row1.data());

    for (int c = 0; c < 4; ++c) {
        float err = std::fabs(row0[c] - data[c]);
        float bound = qm.row_scales[0] * 1.0001f;  // one quantization step
        check(err <= bound, "row 0 dequantized value within one quant step of original");
    }
    for (int c = 0; c < 4; ++c) {
        float err = std::fabs(row1[c] - data[4 + c]);
        float bound = qm.row_scales[1] * 1.0001f;
        check(err <= bound, "row 1 dequantized value within one quant step of original");
    }
}

void test_quantize_zero_row() {
    std::vector<float> data = {0.0f, 0.0f, 0.0f};
    auto qm = modelprep::quantize_int8_per_row(data.data(), 1, 3);
    check(qm.row_scales[0] == 0.0f, "all-zero row gets zero scale (no div-by-zero)");
    std::vector<float> out(3);
    modelprep::dequantize_row(qm, 0, out.data());
    check(out[0] == 0.0f && out[1] == 0.0f && out[2] == 0.0f, "all-zero row dequantizes to zero");
}

void test_embed_quantized_weighted_average() {
    // 3-token vocab, 2-dim embeddings, no PC removal (pc_dim = 0).
    std::vector<float> fp32 = {
        1.0f, 0.0f,   // token 0
        0.0f, 1.0f,   // token 1
        2.0f, 2.0f,   // token 2 (unused below, weight 0)
    };
    auto qm = modelprep::quantize_int8_per_row(fp32.data(), 3, 2);
    std::vector<float> sif_weights = {1.0f, 3.0f, 0.0f};  // token 2 excluded
    std::vector<float> pc_component;  // empty, pc_dim = 0

    auto result = modelprep::embed_quantized({0, 1}, qm, sif_weights, pc_component, 0, 2);
    // Weighted average: (1*[1,0] + 3*[0,1]) / 4 = [0.25, 0.75]
    check(std::fabs(result[0] - 0.25f) < 0.02f, "weighted pooling x-component matches expected average");
    check(std::fabs(result[1] - 0.75f) < 0.02f, "weighted pooling y-component matches expected average");
}

void test_embed_quantized_pc_removal() {
    std::vector<float> fp32 = {1.0f, 0.0f};
    auto qm = modelprep::quantize_int8_per_row(fp32.data(), 1, 2);
    std::vector<float> sif_weights = {1.0f};
    std::vector<float> pc_component = {1.0f, 0.0f};  // removes the x-axis entirely

    auto result = modelprep::embed_quantized({0}, qm, sif_weights, pc_component, 1, 2);
    check(std::fabs(result[0]) < 1e-4f, "component parallel to pc is fully removed");
}

}  // namespace

int main() {
    test_quantize_basic();
    test_quantize_zero_row();
    test_embed_quantized_weighted_average();
    test_embed_quantized_pc_removal();

    if (g_failures > 0) {
        std::fprintf(stderr, "\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
