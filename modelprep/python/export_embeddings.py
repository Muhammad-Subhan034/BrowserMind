"""
BrowserMind modelprep -- stage 0 (Python / conda env "browsermind")

Pulls the real pretrained WordPiece token-embedding matrix out of
sentence-transformers/all-MiniLM-L6-v2 (a genuine, widely used embedding
model) and prepares everything the C++ `modelprep` tool needs to quantize
and pack a browser-ready model file:

  - the raw fp32 embedding matrix           -> embeddings_fp32.bin
  - the WordPiece vocabulary                -> vocab.bin
  - SIF (Smooth Inverse Frequency) weights   -> sif_weights.bin
  - the top principal component to remove   -> pc_component.bin
  - a labeled validation set with fp32       -> validation.bin
    reference sentence embeddings

Why SIF instead of running the full transformer?
--------------------------------------------------
This project's GPU work is the point: hand-written WGSL kernels for
embedding lookup, weighted pooling, similarity, and top-k selection.
Re-implementing six transformer encoder layers (multi-head attention,
LayerNorm, GELU MLPs) in WGSL is a multi-week project on its own and
would bury the retrieval-kernel work this project is actually about.

Instead we use the model's real, pretrained WordPiece embedding table
combined with SIF weighting (Arora et al., 2017, "A Simple but
Tough-to-Beat Baseline for Sentence Embeddings") -- a well-documented,
non-neural sentence-embedding method that is known to perform close to
much heavier neural encoders on semantic similarity benchmarks. Every
number this pipeline produces is a real, measured cosine similarity
against a real fp32 reference -- nothing here is simulated.

Usage (inside the `browsermind` conda env):
    python export_embeddings.py --out ../build/raw
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from tokenizers import Tokenizer
from wordfreq import word_frequency

from corpus import CORPUS

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SIF_A = 1e-3
PC_DIM = 1


def download_model(cache_dir: Path) -> tuple[Path, Path]:
    print(f"[export] fetching {MODEL_ID} from Hugging Face Hub...")
    tok_path = hf_hub_download(MODEL_ID, "tokenizer.json", cache_dir=str(cache_dir))
    weights_path = hf_hub_download(MODEL_ID, "model.safetensors", cache_dir=str(cache_dir))
    return Path(tok_path), Path(weights_path)


def load_embedding_matrix(weights_path: Path) -> np.ndarray:
    tensors = load_file(str(weights_path))
    key = "embeddings.word_embeddings.weight"
    if key not in tensors:
        candidates = [k for k in tensors if "word_embeddings" in k]
        if not candidates:
            raise RuntimeError(f"no word_embeddings tensor found in {weights_path}")
        key = candidates[0]
    print(f"[export] using tensor '{key}'")
    return np.ascontiguousarray(tensors[key].astype(np.float32))


def build_vocab(tokenizer: Tokenizer) -> list[str]:
    vocab_dict = tokenizer.get_vocab()
    vocab_size = max(vocab_dict.values()) + 1
    id_to_token: list[str | None] = [None] * vocab_size
    for tok, idx in vocab_dict.items():
        id_to_token[idx] = tok
    missing = [i for i, t in enumerate(id_to_token) if t is None]
    if missing:
        raise RuntimeError(f"vocab has {len(missing)} unfilled ids, e.g. {missing[:5]}")
    return id_to_token  # type: ignore[return-value]


def compute_sif_weight(token: str, a: float = SIF_A) -> float:
    surface = token[2:] if token.startswith("##") else token
    freq = word_frequency(surface.lower(), "en")
    if freq <= 0:
        freq = 1e-9
    return a / (a + freq)


def encode_mean_sif(token_ids: np.ndarray, emb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if len(token_ids) == 0:
        return np.zeros(emb.shape[1], dtype=np.float32)
    vecs = emb[token_ids]
    w = weights[token_ids].reshape(-1, 1)
    total = float(w.sum())
    if total <= 1e-8:
        return vecs.mean(axis=0).astype(np.float32)
    return ((vecs * w).sum(axis=0) / total).astype(np.float32)


def compute_top_pc(pooled_matrix: np.ndarray, pc_dim: int = PC_DIM) -> np.ndarray:
    _, _, vt = np.linalg.svd(pooled_matrix, full_matrices=False)
    return np.ascontiguousarray(vt[:pc_dim].astype(np.float32))


def remove_pc(vec: np.ndarray, pc: np.ndarray) -> np.ndarray:
    out = vec.copy()
    for row in pc:
        out = out - float(out @ row) * row
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Binary writers (little-endian, hand-rolled -- mirrored exactly by the C++
# readers in modelprep/src/raw_io.cpp). No JSON, no external serialization
# library: these are internal exchange files between the Python and C++
# stages of the same pipeline.
# ---------------------------------------------------------------------------

def write_manifest(path: Path, vocab_size: int, hidden_dim: int,
                    unk_id: int, cls_id: int, sep_id: int, pad_id: int,
                    sif_a: float, pc_dim: int, model_name: str, model_revision: str) -> None:
    name_b = model_name.encode("utf-8")
    rev_b = model_revision.encode("utf-8")
    with open(path, "wb") as f:
        f.write(b"BMPR")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", hidden_dim))
        f.write(struct.pack("<I", unk_id))
        f.write(struct.pack("<I", cls_id))
        f.write(struct.pack("<I", sep_id))
        f.write(struct.pack("<I", pad_id))
        f.write(struct.pack("<f", sif_a))
        f.write(struct.pack("<I", pc_dim))
        f.write(struct.pack("<I", len(name_b)))
        f.write(name_b)
        f.write(struct.pack("<I", len(rev_b)))
        f.write(rev_b)


def write_vocab_bin(id_to_token: list[str], path: Path) -> None:
    blob_parts = []
    offsets = [0]
    running = 0
    for tok in id_to_token:
        b = tok.encode("utf-8")
        running += len(b)
        offsets.append(running)
        blob_parts.append(b)
    blob = b"".join(blob_parts)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(id_to_token)))
        for off in offsets:
            f.write(struct.pack("<I", off))
        f.write(blob)


def write_validation_bin(references: list[dict], hidden_dim: int, path: Path) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(references)))
        for ref in references:
            text_b = ref["text"].encode("utf-8")
            ids = ref["ids"]
            emb = ref["embedding"]
            assert len(emb) == hidden_dim
            f.write(struct.pack("<I", len(text_b)))
            f.write(text_b)
            f.write(struct.pack("<I", len(ids)))
            for tid in ids:
                f.write(struct.pack("<I", int(tid)))
            f.write(struct.pack(f"<{hidden_dim}f", *[float(x) for x in emb]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("../build/raw"))
    ap.add_argument("--cache", type=Path, default=Path("../build/hf_cache"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    tok_path, weights_path = download_model(args.cache)
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab_dict = tokenizer.get_vocab()

    id_to_token = build_vocab(tokenizer)
    vocab_size = len(id_to_token)

    emb = load_embedding_matrix(weights_path)
    if emb.shape[0] < vocab_size:
        raise RuntimeError(f"embedding matrix has {emb.shape[0]} rows but vocab needs {vocab_size}")
    emb = np.ascontiguousarray(emb[:vocab_size])
    hidden_dim = emb.shape[1]

    unk_id = vocab_dict.get("[UNK]", 100)
    cls_id = vocab_dict.get("[CLS]", 101)
    sep_id = vocab_dict.get("[SEP]", 102)
    pad_id = vocab_dict.get("[PAD]", 0)

    print(f"[export] vocab_size={vocab_size} hidden_dim={hidden_dim}")

    print("[export] computing SIF weights from real English word frequencies...")
    sif_weights = np.array([compute_sif_weight(t) for t in id_to_token], dtype=np.float32)
    # Special tokens carry no lexical meaning outside a transformer's
    # contextualization step -- zero them out so they never enter the
    # non-contextual weighted average.
    for special_id in (unk_id, cls_id, sep_id, pad_id):
        if 0 <= special_id < vocab_size:
            sif_weights[special_id] = 0.0

    print(f"[export] tokenizing {len(CORPUS)} validation/demo sentences (no special tokens added)...")
    encoded = [tokenizer.encode(s, add_special_tokens=False) for s in CORPUS]

    raw_pooled = np.stack([
        encode_mean_sif(np.array(e.ids, dtype=np.int64), emb, sif_weights) for e in encoded
    ])
    pc = compute_top_pc(raw_pooled, PC_DIM)

    references = []
    for s, e in zip(CORPUS, encoded):
        pooled = encode_mean_sif(np.array(e.ids, dtype=np.int64), emb, sif_weights)
        final = remove_pc(pooled, pc)
        references.append({"text": s, "ids": e.ids, "embedding": final})

    write_manifest(
        args.out / "manifest.bin", vocab_size, hidden_dim,
        unk_id, cls_id, sep_id, pad_id, SIF_A, PC_DIM,
        MODEL_ID, "main",
    )
    emb.astype("<f4").tofile(args.out / "embeddings_fp32.bin")
    write_vocab_bin(id_to_token, args.out / "vocab.bin")
    sif_weights.astype("<f4").tofile(args.out / "sif_weights.bin")
    pc.astype("<f4").tofile(args.out / "pc_component.bin")
    write_validation_bin(references, hidden_dim, args.out / "validation.bin")

    # Human-readable copies for debugging / README screenshots (not consumed
    # by the C++ tool).
    (args.out / "vocab_preview.txt").write_text(
        "\n".join(id_to_token[:200]), encoding="utf-8"
    )

    print(f"[export] wrote raw pipeline inputs to {args.out.resolve()}")
    print("[export] next: run the C++ modelprep tool to quantize, pack, and validate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
