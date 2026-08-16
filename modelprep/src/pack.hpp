// Assembles the final ".bmind" file the browser downloads. See
// include/bmind_format.hpp for the exact byte layout this writes.
#pragma once

#include <filesystem>

#include "quantize.hpp"
#include "raw_io.hpp"

namespace modelprep {

void pack_bmind(const std::filesystem::path& out_path,
                 const RawManifest& manifest,
                 const RawVocab& vocab,
                 const QuantizedMatrix& quantized,
                 const std::vector<float>& sif_weights,
                 const std::vector<float>& pc_component);

}  // namespace modelprep
