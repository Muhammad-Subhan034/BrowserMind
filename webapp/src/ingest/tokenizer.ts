// Hand-written WordPiece tokenizer, client-side, matching the exact
// algorithm BERT-family models (including MiniLM) were trained with:
// lowercase + whitespace/punctuation pre-tokenization, then greedy
// longest-match-first subword splitting against the model's vocabulary,
// with "##" marking any piece that continues the previous one.
//
// This has to independently reproduce what `tokenizers.Tokenizer.encode`
// does in modelprep/python/export_embeddings.py (with add_special_tokens
// disabled there too) -- both sides must agree on token ids or the
// pipeline the browser runs diverges from the one modelprep validated.
//
// Reference: Wu et al., "Google's Neural Machine Translation System"
// (2016), section on WordPiece; Devlin et al., BERT (2019), Appendix A.2.

import type { BMindModel } from "../gpu/modelFormat";

const MAX_CHARS_PER_WORD = 100;

function isWhitespace(ch: string): boolean {
  return ch === " " || ch === "\t" || ch === "\n" || ch === "\r" || /\s/.test(ch);
}

function isPunctuation(ch: string): boolean {
  const code = ch.codePointAt(0)!;
  // ASCII punctuation ranges, same set BERT's BasicTokenizer treats as
  // "always split on" (not covered by \p{P}, e.g. `$`, `+`, `^`).
  if (
    (code >= 33 && code <= 47) ||
    (code >= 58 && code <= 64) ||
    (code >= 91 && code <= 96) ||
    (code >= 123 && code <= 126)
  ) {
    return true;
  }
  return /\p{P}/u.test(ch);
}

/** Lowercase + split into words and standalone punctuation tokens. */
export function basicTokenize(text: string): string[] {
  const normalized = text.normalize("NFKC").toLowerCase();
  const words: string[] = [];
  let current = "";

  for (const ch of normalized) {
    if (isWhitespace(ch)) {
      if (current) { words.push(current); current = ""; }
    } else if (isPunctuation(ch)) {
      if (current) { words.push(current); current = ""; }
      words.push(ch);
    } else {
      current += ch;
    }
  }
  if (current) words.push(current);
  return words;
}

/** Greedy longest-match-first WordPiece split of a single pre-tokenized word. */
function wordPieceSplit(word: string, tokenToId: Map<string, number>, unkId: number): number[] {
  if (word.length > MAX_CHARS_PER_WORD) return [unkId];

  const ids: number[] = [];
  let start = 0;
  const chars = Array.from(word);

  while (start < chars.length) {
    let end = chars.length;
    let matchedId = -1;
    let matchedLen = 0;

    while (end > start) {
      let candidate = chars.slice(start, end).join("");
      if (start > 0) candidate = "##" + candidate;
      const id = tokenToId.get(candidate);
      if (id !== undefined) {
        matchedId = id;
        matchedLen = end - start;
        break;
      }
      end--;
    }

    if (matchedId === -1) return [unkId];
    ids.push(matchedId);
    start += matchedLen;
  }

  return ids;
}

export interface TokenizeResult {
  ids: number[];
  /** Character span [start, end) in the original text for each id, for highlighting. */
  spans: Array<[number, number]>;
}

/**
 * Full pipeline: basic tokenize + WordPiece split, tracking the original
 * character span of every emitted subword token so the UI can highlight
 * matched passages precisely.
 */
export function tokenize(text: string, model: BMindModel): TokenizeResult {
  const ids: number[] = [];
  const spans: Array<[number, number]> = [];

  const normalized = text.normalize("NFKC");
  let i = 0;
  const n = normalized.length;

  while (i < n) {
    while (i < n && isWhitespace(normalized[i])) i++;
    if (i >= n) break;

    const wordStart = i;
    let word: string;
    if (isPunctuation(normalized[i])) {
      word = normalized[i];
      i++;
    } else {
      let j = i;
      while (j < n && !isWhitespace(normalized[j]) && !isPunctuation(normalized[j])) j++;
      word = normalized.slice(i, j);
      i = j;
    }
    const wordEnd = i;

    const lower = word.toLowerCase();
    const pieceIds = wordPieceSplit(lower, model.tokenToId, model.unkId);
    for (const id of pieceIds) {
      ids.push(id);
      // Character-level span attribution per sub-piece isn't meaningful
      // without re-running the piece boundaries against the original
      // string; we attribute the whole word's span to each of its pieces,
      // which is enough for passage highlighting at chunk granularity.
      spans.push([wordStart, wordEnd]);
    }
  }

  return { ids, spans };
}
