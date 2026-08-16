// modelprep -- BrowserMind's C++ host-side model preparation tool.
//
//   modelprep --raw <dir> --out <file.bmind> [--report <file.txt>] [--min-cosine 0.98]
//
// Pipeline: load the fp32 artifacts exported by python/export_embeddings.py
// -> quantize the embedding matrix to INT8 (per-row scale, hand-written,
// see quantize.cpp) -> pack everything into the browser-ready .bmind binary
// (see bmind_format.hpp) -> validate the quantized pipeline's output
// against the fp32 Python reference on a held-out sentence set, reporting
// cosine similarity. Exits non-zero if mean cosine similarity falls below
// --min-cosine, so this can gate a build/deploy script.
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

#include "pack.hpp"
#include "quantize.hpp"
#include "raw_io.hpp"
#include "validate.hpp"

namespace fs = std::filesystem;

namespace {

struct Args {
    fs::path raw_dir;
    fs::path out_path;
    std::optional<fs::path> report_path;
    float min_cosine = 0.98f;
};

[[noreturn]] void usage_and_exit(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " --raw <dir> --out <file.bmind> [--report <file.txt>] [--min-cosine 0.98]\n";
    std::exit(2);
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) usage_and_exit(argv[0]);
            (void)flag;
            return argv[++i];
        };
        if (a == "--raw") args.raw_dir = next(a.c_str());
        else if (a == "--out") args.out_path = next(a.c_str());
        else if (a == "--report") args.report_path = fs::path(next(a.c_str()));
        else if (a == "--min-cosine") args.min_cosine = std::stof(next(a.c_str()));
        else if (a == "--help" || a == "-h") usage_and_exit(argv[0]);
        else {
            std::cerr << "unrecognized argument: " << a << "\n";
            usage_and_exit(argv[0]);
        }
    }
    if (args.raw_dir.empty() || args.out_path.empty()) usage_and_exit(argv[0]);
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    using clock = std::chrono::steady_clock;
    Args args = parse_args(argc, argv);

    try {
        auto t0 = clock::now();

        std::cout << "[modelprep] loading raw pipeline inputs from " << args.raw_dir << "\n";
        auto manifest = modelprep::read_manifest(args.raw_dir / "manifest.bin");
        auto vocab = modelprep::read_vocab(args.raw_dir / "vocab.bin");
        auto fp32_matrix = modelprep::read_f32_array(
            args.raw_dir / "embeddings_fp32.bin",
            static_cast<size_t>(manifest.vocab_size) * manifest.hidden_dim);
        auto sif_weights = modelprep::read_f32_array(args.raw_dir / "sif_weights.bin", manifest.vocab_size);
        auto pc_component = modelprep::read_f32_array(
            args.raw_dir / "pc_component.bin",
            static_cast<size_t>(manifest.pc_dim) * manifest.hidden_dim);
        auto validation_examples = modelprep::read_validation(args.raw_dir / "validation.bin", manifest.hidden_dim);

        std::cout << "[modelprep] model: " << manifest.model_name << " (" << manifest.model_revision << ")\n";
        std::cout << "[modelprep] vocab_size=" << manifest.vocab_size
                  << " hidden_dim=" << manifest.hidden_dim
                  << " pc_dim=" << manifest.pc_dim << "\n";

        auto t_quant0 = clock::now();
        auto quantized = modelprep::quantize_int8_per_row(fp32_matrix.data(), manifest.vocab_size, manifest.hidden_dim);
        auto t_quant1 = clock::now();
        double quant_ms = std::chrono::duration<double, std::milli>(t_quant1 - t_quant0).count();

        double fp32_bytes = static_cast<double>(fp32_matrix.size()) * sizeof(float);
        double int8_bytes = static_cast<double>(quantized.data.size()) + quantized.row_scales.size() * sizeof(float);
        std::cout << "[modelprep] quantized " << manifest.vocab_size << " rows in " << quant_ms << " ms  ("
                  << (fp32_bytes / 1024.0 / 1024.0) << " MB fp32 -> "
                  << (int8_bytes / 1024.0 / 1024.0) << " MB int8+scales, "
                  << (fp32_bytes / int8_bytes) << "x smaller)\n";

        fs::create_directories(args.out_path.parent_path());
        modelprep::pack_bmind(args.out_path, manifest, vocab, quantized, sif_weights, pc_component);
        auto out_size = fs::file_size(args.out_path);
        std::cout << "[modelprep] wrote " << args.out_path << " (" << (out_size / 1024.0 / 1024.0) << " MB)\n";

        std::cout << "[modelprep] validating quantized pipeline against " << validation_examples.size()
                  << " fp32 reference embeddings...\n";
        auto report = modelprep::run_validation(validation_examples, quantized, sif_weights, pc_component,
                                                  manifest.pc_dim, manifest.hidden_dim);

        std::cout << "[modelprep] cosine similarity (quantized vs fp32 reference): "
                  << "min=" << report.min_cosine << " mean=" << report.mean_cosine
                  << " max=" << report.max_cosine << "\n";

        if (args.report_path) {
            std::ofstream rf(*args.report_path);
            rf << "BrowserMind modelprep validation report\n";
            rf << "model: " << manifest.model_name << " (" << manifest.model_revision << ")\n";
            rf << "vocab_size: " << manifest.vocab_size << "  hidden_dim: " << manifest.hidden_dim << "\n";
            rf << "quantization: int8, per-row scale\n";
            rf << "fp32 size: " << (fp32_bytes / 1024.0 / 1024.0) << " MB\n";
            rf << "int8 size: " << (int8_bytes / 1024.0 / 1024.0) << " MB\n";
            rf << "compression ratio: " << (fp32_bytes / int8_bytes) << "x\n";
            rf << "quantization time: " << quant_ms << " ms\n\n";
            rf << "cosine similarity (quantized vs fp32 reference):\n";
            rf << "  min:  " << report.min_cosine << "\n";
            rf << "  mean: " << report.mean_cosine << "\n";
            rf << "  max:  " << report.max_cosine << "\n\n";
            rf << "per-example:\n";
            for (const auto& r : report.per_example) {
                char buf[16];
                std::snprintf(buf, sizeof(buf), "%.5f", r.cosine_similarity);
                rf << "  " << buf << "  " << r.text << "\n";
            }
            std::cout << "[modelprep] wrote validation report to " << *args.report_path << "\n";
        }

        auto t1 = clock::now();
        std::cout << "[modelprep] done in " << std::chrono::duration<double>(t1 - t0).count() << "s\n";

        if (report.mean_cosine < args.min_cosine) {
            std::cerr << "[modelprep] FAILED: mean cosine similarity " << report.mean_cosine
                      << " is below threshold " << args.min_cosine << "\n";
            return 1;
        }
        std::cout << "[modelprep] OK: mean cosine similarity " << report.mean_cosine
                  << " meets threshold " << args.min_cosine << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[modelprep] ERROR: " << ex.what() << "\n";
        return 1;
    }
}
