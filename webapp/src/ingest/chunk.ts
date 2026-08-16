import type { BMindModel } from "../gpu/modelFormat";
import { tokenize } from "./tokenizer";

export interface Chunk {
  id: string;
  docName: string;
  text: string;
  tokenIds: number[];
  charStart: number;
  charEnd: number;
}

export interface ChunkOptions {
  /** Tokens per chunk window. */
  windowSize?: number;
  /** Tokens shared between consecutive windows. */
  overlap?: number;
}

const DEFAULTS: Required<ChunkOptions> = { windowSize: 128, overlap: 32 };

/**
 * Splits one document's text into overlapping token windows. Tokenization
 * happens once over the whole document; windows are then cut directly from
 * the resulting token-id array, so overlap costs nothing extra to compute.
 */
export function chunkDocument(
  docName: string,
  text: string,
  model: BMindModel,
  opts: ChunkOptions = {},
): Chunk[] {
  const { windowSize, overlap } = { ...DEFAULTS, ...opts };
  const stride = Math.max(1, windowSize - overlap);

  const { ids, spans } = tokenize(text, model);
  if (ids.length === 0) return [];

  const chunks: Chunk[] = [];
  for (let start = 0; start < ids.length; start += stride) {
    const end = Math.min(start + windowSize, ids.length);
    const tokenIds = ids.slice(start, end);
    const charStart = spans[start][0];
    const charEnd = spans[end - 1][1];
    chunks.push({
      id: `${docName}#${start}`,
      docName,
      text: text.slice(charStart, charEnd),
      tokenIds,
      charStart,
      charEnd,
    });
    if (end >= ids.length) break;
  }
  return chunks;
}
