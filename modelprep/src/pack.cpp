#include "pack.hpp"

#include <fstream>
#include <stdexcept>

#include "bmind_format.hpp"

namespace modelprep {

namespace {

void write_u32(std::ofstream& f, uint32_t v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void write_f32(std::ofstream& f, float v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

}  // namespace

void pack_bmind(const std::filesystem::path& out_path,
                 const RawManifest& manifest,
                 const RawVocab& vocab,
                 const QuantizedMatrix& quantized,
                 const std::vector<float>& sif_weights,
                 const std::vector<float>& pc_component) {
    if (vocab.size() != manifest.vocab_size) {
        throw std::runtime_error("vocab size mismatch between manifest and vocab table");
    }
    if (quantized.rows != manifest.vocab_size || quantized.cols != manifest.hidden_dim) {
        throw std::runtime_error("quantized matrix shape mismatch with manifest");
    }
    if (sif_weights.size() != manifest.vocab_size) {
        throw std::runtime_error("sif_weights size mismatch with manifest");
    }
    if (pc_component.size() != static_cast<size_t>(manifest.pc_dim) * manifest.hidden_dim) {
        throw std::runtime_error("pc_component size mismatch with manifest");
    }

    std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("failed to open output " + out_path.string());

    f.write(bmind::kMagic, 4);
    write_u32(f, bmind::kVersion);
    write_u32(f, manifest.vocab_size);
    write_u32(f, manifest.hidden_dim);
    write_u32(f, manifest.pc_dim);
    write_u32(f, manifest.unk_id);
    write_u32(f, manifest.cls_id);
    write_u32(f, manifest.sep_id);
    write_u32(f, manifest.pad_id);
    write_f32(f, manifest.sif_a);

    for (uint32_t off : vocab.offsets) write_u32(f, off);
    f.write(reinterpret_cast<const char*>(vocab.blob.data()), static_cast<std::streamsize>(vocab.blob.size()));

    // Pad to a 4-byte boundary so every section from here on can be viewed
    // as a Float32Array/Uint32Array directly over the browser's downloaded
    // ArrayBuffer with zero copies (see bmind_format.hpp).
    size_t pad = (4 - (vocab.blob.size() % 4)) % 4;
    static const char kZeros[4] = {0, 0, 0, 0};
    if (pad > 0) f.write(kZeros, static_cast<std::streamsize>(pad));

    f.write(reinterpret_cast<const char*>(quantized.data.data()),
            static_cast<std::streamsize>(quantized.data.size()));
    f.write(reinterpret_cast<const char*>(quantized.row_scales.data()),
            static_cast<std::streamsize>(quantized.row_scales.size() * sizeof(float)));

    f.write(reinterpret_cast<const char*>(sif_weights.data()),
            static_cast<std::streamsize>(sif_weights.size() * sizeof(float)));
    f.write(reinterpret_cast<const char*>(pc_component.data()),
            static_cast<std::streamsize>(pc_component.size() * sizeof(float)));

    if (!f) throw std::runtime_error("write failure while packing " + out_path.string());
}

}  // namespace modelprep
