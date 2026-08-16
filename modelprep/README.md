# modelprep

C++ host-side tool that turns a pretrained embedding model into a compact, browser-ready binary. See the [top-level README](../README.md) for how this fits into the full BrowserMind project and why the model architecture is SIF pooling rather than a full transformer.

Zero external C++ dependencies — CMake, the standard library, and nothing else. Raw intermediate files exchanged with the Python export stage use a hand-rolled binary format rather than JSON, so there's no serialization library to vendor either.

## Pipeline

```
python/export_embeddings.py         (conda env, run once)
  → downloads sentence-transformers/all-MiniLM-L6-v2 from Hugging Face
  → extracts the real, pretrained WordPiece token-embedding matrix (fp32)
  → computes SIF pooling weights from real English word frequencies (wordfreq)
  → tokenizes a curated validation/demo corpus, computes fp32 reference embeddings
  → writes build/raw/*.bin  (manifest, embeddings, vocab, sif_weights, pc_component, validation)

modelprep  (this C++ tool)
  → quantize.cpp:   per-row INT8 quantization, hand-written (src/quantize.cpp)
  → pack.cpp:       assembles the final .bmind binary (include/bmind_format.hpp)
  → validate.cpp:   re-runs the SIF pipeline on the quantized weights, compares
                     cosine similarity against the fp32 Python reference
  → main.cpp:        CLI orchestrating the above, exits non-zero if mean cosine
                      similarity falls below --min-cosine (default 0.98)
```

## Build

Requires a real C++17 compiler — see the note in the top-level README about why conda-forge's `m2w64-toolchain` (GCC 5.3) won't work on Windows, and use [WinLibs](https://winlibs.com/) or a system GCC 9+/Clang 10+ instead.

```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=<path-to-g++>
cmake --build build
```

## Run

```bash
# 1. Export raw pipeline inputs (inside the `browsermind` conda env)
cd python && python export_embeddings.py --out ../build/raw

# 2. Quantize, pack, and validate
cd ..
./build/modelprep --raw build/raw --out ../webapp/public/models/minilm-sif.bmind \
                   --report build/validation_report.txt --min-cosine 0.98
```

## Test

```bash
./build/modelprep_tests
```

Dependency-free checks (no test framework) covering: per-row quantization correctness (values reconstruct within one quantization step, the max-magnitude element in each row maps to exactly ±127, all-zero rows don't divide by zero), and the pooling + principal-component-removal math (`embed_quantized`) against hand-computed expected results.

## The `.bmind` binary format

See [`include/bmind_format.hpp`](include/bmind_format.hpp) for the authoritative, field-by-field layout — it's the single source of truth both this tool's writer (`src/pack.cpp`) and the browser's reader (`webapp/src/gpu/modelFormat.ts`) are kept in sync with by hand. Every section is a flat primitive array or a byte blob addressed by an offset table, so the browser can view the downloaded `ArrayBuffer` directly as typed arrays with effectively zero parsing.

## Real numbers from the last run

See [`validation_report.txt`](validation_report.txt) for the full output. Summary: 30,522-token vocabulary, 384-dim embeddings, 44.71 MB fp32 → 11.29 MB INT8+scales (3.96× smaller), quantized-vs-fp32 cosine similarity mean **0.999943** across 59 held-out sentences.
