// Parser for the ".bmind" binary format produced by modelprep (see
// modelprep/include/bmind_format.hpp for the authoritative byte layout --
// this file must stay in sync with it by hand, there is no shared schema
// generator, so the header comment there is the single source of truth).
//
// The whole point of this format is that parsing it costs almost nothing:
// we read nine header fields, then every remaining section is either a
// flat typed-array view directly over the downloaded ArrayBuffer, or a
// vocab blob sliced with TextDecoder. Nothing here allocates a copy of the
// large embedding matrix.

export interface BMindModel {
  vocabSize: number;
  hiddenDim: number;
  pcDim: number;
  unkId: number;
  clsId: number;
  sepId: number;
  padId: number;
  sifA: number;

  vocabOffsets: Uint32Array;
  vocabBlob: Uint8Array;

  /** Int8, row-major: quantMatrix[tokenId * hiddenDim + dim] */
  quantMatrix: Int8Array;
  /** Dequant multiplier per row: value ~= quantMatrix[row] * rowScales[row] */
  rowScales: Float32Array;
  sifWeights: Float32Array;
  /** pcDim * hiddenDim, row-major */
  pcComponent: Float32Array;

  tokenToId: Map<string, number>;
  idToToken: string[];
}

const MAGIC = "BMND";

export function parseBMind(buffer: ArrayBuffer): BMindModel {
  const view = new DataView(buffer);
  let offset = 0;

  const magicBytes = new Uint8Array(buffer, 0, 4);
  const magic = String.fromCharCode(...magicBytes);
  if (magic !== MAGIC) {
    throw new Error(`bad .bmind magic: expected "${MAGIC}", got "${magic}"`);
  }
  offset += 4;

  const version = view.getUint32(offset, true); offset += 4;
  if (version !== 1) throw new Error(`unsupported .bmind version ${version}`);

  const vocabSize = view.getUint32(offset, true); offset += 4;
  const hiddenDim = view.getUint32(offset, true); offset += 4;
  const pcDim = view.getUint32(offset, true); offset += 4;
  const unkId = view.getUint32(offset, true); offset += 4;
  const clsId = view.getUint32(offset, true); offset += 4;
  const sepId = view.getUint32(offset, true); offset += 4;
  const padId = view.getUint32(offset, true); offset += 4;
  const sifA = view.getFloat32(offset, true); offset += 4;

  const vocabOffsets = new Uint32Array(buffer, offset, vocabSize + 1);
  offset += (vocabSize + 1) * 4;

  const blobLen = vocabOffsets[vocabSize];
  const vocabBlob = new Uint8Array(buffer, offset, blobLen);
  offset += blobLen;
  offset += (4 - (blobLen % 4)) % 4; // skip alignment padding, see bmind_format.hpp

  const matrixLen = vocabSize * hiddenDim;
  const quantMatrix = new Int8Array(buffer, offset, matrixLen);
  offset += matrixLen;

  const rowScales = new Float32Array(buffer, offset, vocabSize);
  offset += vocabSize * 4;

  const sifWeights = new Float32Array(buffer, offset, vocabSize);
  offset += vocabSize * 4;

  const pcComponent = new Float32Array(buffer, offset, pcDim * hiddenDim);
  offset += pcDim * hiddenDim * 4;

  const decoder = new TextDecoder("utf-8");
  const idToToken: string[] = new Array(vocabSize);
  const tokenToId = new Map<string, number>();
  for (let i = 0; i < vocabSize; i++) {
    const start = vocabOffsets[i];
    const end = vocabOffsets[i + 1];
    const tok = decoder.decode(vocabBlob.subarray(start, end));
    idToToken[i] = tok;
    tokenToId.set(tok, i);
  }

  return {
    vocabSize, hiddenDim, pcDim, unkId, clsId, sepId, padId, sifA,
    vocabOffsets, vocabBlob, quantMatrix, rowScales, sifWeights, pcComponent,
    tokenToId, idToToken,
  };
}

export async function loadBMindModel(url: string): Promise<{ model: BMindModel; byteSize: number }> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to fetch model at ${url}: ${res.status}`);
  const buffer = await res.arrayBuffer();
  return { model: parseBMind(buffer), byteSize: buffer.byteLength };
}
