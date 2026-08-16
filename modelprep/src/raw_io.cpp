#include "raw_io.hpp"

#include <fstream>
#include <stdexcept>

namespace modelprep {

namespace {

std::ifstream open_binary(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("failed to open " + path.string());
    }
    return f;
}

uint32_t read_u32(std::ifstream& f) {
    uint32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!f) throw std::runtime_error("unexpected EOF reading u32");
    return v;
}

float read_f32(std::ifstream& f) {
    float v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!f) throw std::runtime_error("unexpected EOF reading f32");
    return v;
}

std::string read_string(std::ifstream& f) {
    uint32_t len = read_u32(f);
    std::string s(len, '\0');
    if (len > 0) {
        f.read(s.data(), static_cast<std::streamsize>(len));
        if (!f) throw std::runtime_error("unexpected EOF reading string");
    }
    return s;
}

}  // namespace

RawManifest read_manifest(const std::filesystem::path& path) {
    auto f = open_binary(path);
    char magic[4];
    f.read(magic, 4);
    if (!f || magic[0] != 'B' || magic[1] != 'M' || magic[2] != 'P' || magic[3] != 'R') {
        throw std::runtime_error("bad manifest magic in " + path.string());
    }
    uint32_t version = read_u32(f);
    if (version != 1) throw std::runtime_error("unsupported manifest version");

    RawManifest m;
    m.vocab_size = read_u32(f);
    m.hidden_dim = read_u32(f);
    m.unk_id = read_u32(f);
    m.cls_id = read_u32(f);
    m.sep_id = read_u32(f);
    m.pad_id = read_u32(f);
    m.sif_a = read_f32(f);
    m.pc_dim = read_u32(f);
    m.model_name = read_string(f);
    m.model_revision = read_string(f);
    return m;
}

std::string RawVocab::token(uint32_t id) const {
    if (id + 1 >= offsets.size()) return "";
    uint32_t start = offsets[id];
    uint32_t end = offsets[id + 1];
    return std::string(reinterpret_cast<const char*>(blob.data()) + start, end - start);
}

RawVocab read_vocab(const std::filesystem::path& path) {
    auto f = open_binary(path);
    uint32_t vocab_size = read_u32(f);

    RawVocab v;
    v.offsets.resize(vocab_size + 1);
    for (uint32_t i = 0; i <= vocab_size; ++i) {
        v.offsets[i] = read_u32(f);
    }
    uint32_t blob_len = v.offsets[vocab_size];
    v.blob.resize(blob_len);
    if (blob_len > 0) {
        f.read(reinterpret_cast<char*>(v.blob.data()), blob_len);
        if (!f) throw std::runtime_error("unexpected EOF reading vocab blob");
    }
    return v;
}

std::vector<float> read_f32_array(const std::filesystem::path& path, size_t expected_count) {
    auto f = open_binary(path);
    std::vector<float> data(expected_count);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expected_count * sizeof(float)));
    if (!f) throw std::runtime_error("unexpected EOF reading f32 array from " + path.string());
    return data;
}

std::vector<ValidationExample> read_validation(const std::filesystem::path& path, uint32_t hidden_dim) {
    auto f = open_binary(path);
    uint32_t n = read_u32(f);
    std::vector<ValidationExample> out;
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        ValidationExample ex;
        ex.text = read_string(f);
        uint32_t num_tokens = read_u32(f);
        ex.token_ids.resize(num_tokens);
        for (uint32_t t = 0; t < num_tokens; ++t) {
            ex.token_ids[t] = read_u32(f);
        }
        ex.reference_embedding.resize(hidden_dim);
        f.read(reinterpret_cast<char*>(ex.reference_embedding.data()),
               static_cast<std::streamsize>(hidden_dim * sizeof(float)));
        if (!f) throw std::runtime_error("unexpected EOF reading validation embedding");
        out.push_back(std::move(ex));
    }
    return out;
}

}  // namespace modelprep
