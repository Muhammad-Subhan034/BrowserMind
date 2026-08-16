// Readers for the intermediate "raw" binary files written by
// modelprep/python/export_embeddings.py. These are internal exchange files
// between the Python export stage and this C++ quantize/pack/validate
// stage -- not consumed by the browser. See export_embeddings.py's
// write_manifest / write_vocab_bin / write_validation_bin for the paired
// writer implementations.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace modelprep {

struct RawManifest {
    uint32_t vocab_size = 0;
    uint32_t hidden_dim = 0;
    uint32_t unk_id = 0;
    uint32_t cls_id = 0;
    uint32_t sep_id = 0;
    uint32_t pad_id = 0;
    float sif_a = 0.0f;
    uint32_t pc_dim = 0;
    std::string model_name;
    std::string model_revision;
};

struct ValidationExample {
    std::string text;
    std::vector<uint32_t> token_ids;
    std::vector<float> reference_embedding;
};

RawManifest read_manifest(const std::filesystem::path& path);

// Returns a vocab_size+1 length offset table and the concatenated UTF-8 blob.
struct RawVocab {
    std::vector<uint32_t> offsets;
    std::vector<uint8_t> blob;
    uint32_t size() const { return offsets.empty() ? 0 : static_cast<uint32_t>(offsets.size() - 1); }
    std::string token(uint32_t id) const;
};
RawVocab read_vocab(const std::filesystem::path& path);

std::vector<float> read_f32_array(const std::filesystem::path& path, size_t expected_count);

std::vector<ValidationExample> read_validation(const std::filesystem::path& path, uint32_t hidden_dim);

}  // namespace modelprep
