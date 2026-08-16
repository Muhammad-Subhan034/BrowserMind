// Persistent client-side cache: once a document has been chunked and
// embedded, its chunks + embeddings are stored in IndexedDB keyed by a
// content hash, so reloading the page (or re-adding the same file) never
// re-runs the GPU pipeline for unchanged content. This is what makes the
// "no backend" story actually usable across sessions rather than a
// one-shot demo.

export interface CachedChunk {
  id: string;
  docHash: string;
  docName: string;
  text: string;
  tokenIds: number[];
  embedding: Float32Array;
  charStart: number;
  charEnd: number;
}

export interface CachedDocMeta {
  hash: string;
  name: string;
  chunkCount: number;
  addedAt: number;
}

const DB_NAME = "browsermind";
const DB_VERSION = 1;
const CHUNKS_STORE = "chunks";
const DOCS_STORE = "docs";

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(CHUNKS_STORE)) {
        const store = db.createObjectStore(CHUNKS_STORE, { keyPath: "id" });
        store.createIndex("docHash", "docHash", { unique: false });
      }
      if (!db.objectStoreNames.contains(DOCS_STORE)) {
        db.createObjectStore(DOCS_STORE, { keyPath: "hash" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

export async function hashText(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function getCachedDoc(hash: string): Promise<CachedDocMeta | null> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(DOCS_STORE, "readonly");
    const req = tx.objectStore(DOCS_STORE).get(hash);
    req.onsuccess = () => resolve(req.result ?? null);
    req.onerror = () => reject(req.error);
  });
}

export async function getCachedChunks(hash: string): Promise<CachedChunk[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CHUNKS_STORE, "readonly");
    const idx = tx.objectStore(CHUNKS_STORE).index("docHash");
    const req = idx.getAll(hash);
    req.onsuccess = () => resolve(req.result as CachedChunk[]);
    req.onerror = () => reject(req.error);
  });
}

export async function putDocument(meta: CachedDocMeta, chunks: CachedChunk[]): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([DOCS_STORE, CHUNKS_STORE], "readwrite");
    tx.objectStore(DOCS_STORE).put(meta);
    const chunkStore = tx.objectStore(CHUNKS_STORE);
    for (const c of chunks) chunkStore.put(c);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function listDocuments(): Promise<CachedDocMeta[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(DOCS_STORE, "readonly");
    const req = tx.objectStore(DOCS_STORE).getAll();
    req.onsuccess = () => resolve(req.result as CachedDocMeta[]);
    req.onerror = () => reject(req.error);
  });
}

export async function clearAll(): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([DOCS_STORE, CHUNKS_STORE], "readwrite");
    tx.objectStore(DOCS_STORE).clear();
    tx.objectStore(CHUNKS_STORE).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
