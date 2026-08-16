// Orchestrates the four WGSL compute kernels into the two pipelines the
// app actually runs:
//   embedBatch()  -- embedding_lookup.wgsl -> pooling.wgsl
//   search()      -- similarity.wgsl -> topk.wgsl (x k rounds)
//
// Buffers are created fresh per call rather than pooled/reused across
// calls, which is the right trade-off for this app's scale (ingestion and
// search are both "a few times per session" operations, not a hot loop),
// and it keeps the code honest about what each pass actually allocates --
// visible directly in the GPU Internals panel via bufferBytes.

import lookupSrc from "./shaders/embedding_lookup.wgsl?raw";
import poolingSrc from "./shaders/pooling.wgsl?raw";
import similaritySrc from "./shaders/similarity.wgsl?raw";
import topkSrc from "./shaders/topk.wgsl?raw";
import project2dSrc from "./shaders/project2d.wgsl?raw";

import type { BMindModel } from "./modelFormat";
import { GpuProfiler, type PassTiming } from "./profiler";

function specialize(src: string, hiddenDim: number): string {
  return src.replaceAll("__HIDDEN_DIM__", `${hiddenDim}u`);
}

function alignTo4(n: number): number {
  return (n + 3) & ~3;
}

function createStorageBuffer(
  device: GPUDevice,
  data: Float32Array | Uint32Array | Int8Array,
  label: string,
  extraUsage: GPUBufferUsageFlags = 0,
): GPUBuffer {
  const buf = device.createBuffer({
    label,
    size: Math.max(4, alignTo4(data.byteLength)),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
  });
  device.queue.writeBuffer(buf, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  return buf;
}

function createUniformBuffer(device: GPUDevice, data: Uint32Array): GPUBuffer {
  const buf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  return buf;
}

export interface EmbedResult {
  embeddings: Float32Array;
  buffer: GPUBuffer;
  timings: PassTiming[];
  bufferBytes: number;
}

export interface SearchResult {
  indices: Uint32Array;
  scores: Float32Array;
  timings: PassTiming[];
}

export interface DispatchInfo {
  kernel: string;
  workgroupCount: number;
  workgroupSize: number;
}

export class GpuPipeline {
  private device: GPUDevice;
  private model: BMindModel;
  private profiler: GpuProfiler;

  private quantMatrixBuf!: GPUBuffer;
  private rowScalesBuf!: GPUBuffer;
  private sifWeightsBuf!: GPUBuffer;
  private pcComponentBuf!: GPUBuffer;

  private lookupPipeline!: GPUComputePipeline;
  private poolingPipeline!: GPUComputePipeline;
  private similarityPipeline!: GPUComputePipeline;
  private topkBindGroupLayout!: GPUBindGroupLayout;
  private topkPartialPipeline!: GPUComputePipeline;
  private topkFinalPipeline!: GPUComputePipeline;
  private project2dPipeline!: GPUComputePipeline;

  readonly modelBufferBytes: number;
  lastDispatches: DispatchInfo[] = [];

  constructor(device: GPUDevice, model: BMindModel, timestampSupported: boolean) {
    this.device = device;
    this.model = model;
    this.profiler = new GpuProfiler(device, timestampSupported);

    this.quantMatrixBuf = createStorageBuffer(device, model.quantMatrix, "quant_matrix");
    this.rowScalesBuf = createStorageBuffer(device, model.rowScales, "row_scales");
    this.sifWeightsBuf = createStorageBuffer(device, model.sifWeights, "sif_weights");
    this.pcComponentBuf = createStorageBuffer(device, model.pcComponent, "pc_component");
    this.modelBufferBytes =
      alignTo4(model.quantMatrix.byteLength) + alignTo4(model.rowScales.byteLength) +
      alignTo4(model.sifWeights.byteLength) + alignTo4(model.pcComponent.byteLength);

    this.initPipelines();
  }

  private initPipelines() {
    const d = this.device;
    const hiddenDim = this.model.hiddenDim;

    this.lookupPipeline = d.createComputePipeline({
      label: "embedding_lookup",
      layout: "auto",
      compute: { module: d.createShaderModule({ label: "embedding_lookup", code: specialize(lookupSrc, hiddenDim) }), entryPoint: "main" },
    });
    this.poolingPipeline = d.createComputePipeline({
      label: "pooling",
      layout: "auto",
      compute: { module: d.createShaderModule({ label: "pooling", code: specialize(poolingSrc, hiddenDim) }), entryPoint: "main" },
    });
    this.similarityPipeline = d.createComputePipeline({
      label: "similarity",
      layout: "auto",
      compute: { module: d.createShaderModule({ label: "similarity", code: specialize(similaritySrc, hiddenDim) }), entryPoint: "main" },
    });

    // topk's two entry points (reduce_partial / reduce_final) must share
    // an explicit bind group layout: WebGPU's "auto" layout is opaque and
    // unique per-pipeline, so a bind group built against one pipeline's
    // auto layout cannot be reused with another pipeline even if the
    // bindings are structurally identical -- and this app deliberately
    // reuses one bind group across both passes, every round.
    const storageEntry = (binding: number): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    });
    this.topkBindGroupLayout = d.createBindGroupLayout({
      label: "topk",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        storageEntry(1), storageEntry(2), storageEntry(3), storageEntry(4), storageEntry(5),
      ],
    });
    const topkPipelineLayout = d.createPipelineLayout({ bindGroupLayouts: [this.topkBindGroupLayout] });
    const topkModule = d.createShaderModule({ label: "topk", code: topkSrc });
    this.topkPartialPipeline = d.createComputePipeline({ label: "topk_partial", layout: topkPipelineLayout, compute: { module: topkModule, entryPoint: "reduce_partial" } });
    this.topkFinalPipeline = d.createComputePipeline({ label: "topk_final", layout: topkPipelineLayout, compute: { module: topkModule, entryPoint: "reduce_final" } });

    this.project2dPipeline = d.createComputePipeline({
      label: "project2d",
      layout: "auto",
      compute: { module: d.createShaderModule({ label: "project2d", code: specialize(project2dSrc, hiddenDim) }), entryPoint: "main" },
    });
  }

  /** Uploads a precomputed (possibly cache-loaded) set of chunk embeddings as a fresh GPU buffer for search()/project2D(). */
  uploadEmbeddings(embeddings: Float32Array): GPUBuffer {
    return createStorageBuffer(this.device, embeddings, "chunk_embeddings_cached", GPUBufferUsage.COPY_SRC);
  }

  /** Projects every chunk embedding onto a 2D basis (see viz/layout.ts for how the basis is found). */
  async project2D(chunkEmbeddingsBuf: GPUBuffer, numChunks: number, axes: Float32Array): Promise<{ points: Float32Array; timings: PassTiming[] }> {
    const d = this.device;
    const hiddenDim = this.model.hiddenDim;

    const axesBuf = createStorageBuffer(d, axes, "axes");
    const pointsSize = Math.max(4, numChunks * 2 * 4);
    const pointsBuf = d.createBuffer({ label: "points", size: pointsSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const params = createUniformBuffer(d, new Uint32Array([hiddenDim, numChunks, 0, 0]));

    const bindGroup = d.createBindGroup({
      layout: this.project2dPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: chunkEmbeddingsBuf } },
        { binding: 2, resource: { buffer: axesBuf } },
        { binding: 3, resource: { buffer: pointsBuf } },
      ],
    });

    this.profiler.begin(numChunks > 0 ? 1 : 0);
    const encoder = d.createCommandEncoder({ label: "project2d" });
    if (numChunks > 0) {
      const pass = encoder.beginComputePass({ label: "project2d", timestampWrites: this.profiler.passTimestampWrites("project2d") });
      pass.setPipeline(this.project2dPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(numChunks / 64));
      pass.end();
    }
    const readback = d.createBuffer({ size: pointsSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(pointsBuf, 0, readback, 0, pointsSize);
    this.profiler.resolve(encoder);
    d.queue.submit([encoder.finish()]);

    const timings = await this.profiler.readResults();
    await readback.mapAsync(GPUMapMode.READ);
    const points = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();
    axesBuf.destroy();
    pointsBuf.destroy();

    return { points, timings };
  }

  /**
   * Embeds a ragged batch of token-id sequences (one per chunk, or a
   * single-item batch for a query) via embedding_lookup.wgsl ->
   * pooling.wgsl. Returns both a JS-side copy (for caching/CPU use) and a
   * live GPU buffer so search() can reuse it without re-uploading.
   */
  async embedBatch(tokenIdsPerItem: number[][]): Promise<EmbedResult> {
    const d = this.device;
    const hiddenDim = this.model.hiddenDim;
    const numItems = tokenIdsPerItem.length;
    const dispatches: DispatchInfo[] = [];

    const chunkOffsets = new Uint32Array(numItems + 1);
    let total = 0;
    for (let i = 0; i < numItems; i++) {
      chunkOffsets[i] = total;
      total += tokenIdsPerItem[i].length;
    }
    chunkOffsets[numItems] = total;

    const flatTokenIds = new Uint32Array(total);
    {
      let o = 0;
      for (const ids of tokenIdsPerItem) {
        flatTokenIds.set(ids, o);
        o += ids.length;
      }
    }

    const tokenIdsBuf = createStorageBuffer(d, flatTokenIds, "token_ids");
    const chunkOffsetsBuf = createStorageBuffer(d, chunkOffsets, "chunk_offsets");
    const weightedVecsBuf = d.createBuffer({
      label: "weighted_token_vecs",
      size: Math.max(4, total * hiddenDim * 4),
      usage: GPUBufferUsage.STORAGE,
    });
    const chunkEmbeddingsBuf = d.createBuffer({
      label: "chunk_embeddings",
      size: Math.max(4, numItems * hiddenDim * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const lookupParams = createUniformBuffer(d, new Uint32Array([this.model.vocabSize, hiddenDim, this.model.pcDim, total]));
    const lookupBindGroup = d.createBindGroup({
      layout: this.lookupPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: lookupParams } },
        { binding: 1, resource: { buffer: tokenIdsBuf } },
        { binding: 2, resource: { buffer: this.quantMatrixBuf } },
        { binding: 3, resource: { buffer: this.rowScalesBuf } },
        { binding: 4, resource: { buffer: this.sifWeightsBuf } },
        { binding: 5, resource: { buffer: weightedVecsBuf } },
      ],
    });

    const poolingParams = createUniformBuffer(d, new Uint32Array([hiddenDim, this.model.pcDim, numItems, 0]));
    const poolingBindGroup = d.createBindGroup({
      layout: this.poolingPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: poolingParams } },
        { binding: 1, resource: { buffer: chunkOffsetsBuf } },
        { binding: 2, resource: { buffer: tokenIdsBuf } },
        { binding: 3, resource: { buffer: this.sifWeightsBuf } },
        { binding: 4, resource: { buffer: weightedVecsBuf } },
        { binding: 5, resource: { buffer: this.pcComponentBuf } },
        { binding: 6, resource: { buffer: chunkEmbeddingsBuf } },
      ],
    });

    this.profiler.begin(total > 0 ? 2 : 0);
    const encoder = d.createCommandEncoder({ label: "embed-batch" });

    if (total > 0) {
      const quadsPerRow = hiddenDim / 4;
      const totalQuads = total * quadsPerRow;
      const lookupGroups = Math.ceil(totalQuads / 64);
      const lookupPass = encoder.beginComputePass({ label: "embedding_lookup", timestampWrites: this.profiler.passTimestampWrites("embedding_lookup") });
      lookupPass.setPipeline(this.lookupPipeline);
      lookupPass.setBindGroup(0, lookupBindGroup);
      lookupPass.dispatchWorkgroups(lookupGroups);
      lookupPass.end();
      dispatches.push({ kernel: "embedding_lookup", workgroupCount: lookupGroups, workgroupSize: 64 });

      const poolingPass = encoder.beginComputePass({ label: "pooling", timestampWrites: this.profiler.passTimestampWrites("pooling") });
      poolingPass.setPipeline(this.poolingPipeline);
      poolingPass.setBindGroup(0, poolingBindGroup);
      poolingPass.dispatchWorkgroups(numItems);
      poolingPass.end();
      dispatches.push({ kernel: "pooling", workgroupCount: numItems, workgroupSize: 128 });
    }

    const readbackSize = Math.max(4, numItems * hiddenDim * 4);
    const readbackBuf = d.createBuffer({ size: readbackSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(chunkEmbeddingsBuf, 0, readbackBuf, 0, readbackSize);
    this.profiler.resolve(encoder);
    d.queue.submit([encoder.finish()]);

    const timings = await this.profiler.readResults();

    await readbackBuf.mapAsync(GPUMapMode.READ);
    const embeddings = new Float32Array(readbackBuf.getMappedRange().slice(0));
    readbackBuf.unmap();
    readbackBuf.destroy();

    tokenIdsBuf.destroy();
    chunkOffsetsBuf.destroy();
    weightedVecsBuf.destroy();

    this.lastDispatches = dispatches;
    return { embeddings, buffer: chunkEmbeddingsBuf, timings, bufferBytes: numItems * hiddenDim * 4 };
  }

  /**
   * Scores a query embedding against every stored chunk embedding
   * (similarity.wgsl) and selects the top k (topk.wgsl), entirely on the
   * GPU -- only the final k (index, score) pairs are read back.
   */
  async search(queryEmbedding: Float32Array, chunkEmbeddingsBuf: GPUBuffer, numChunks: number, k: number): Promise<SearchResult> {
    const d = this.device;
    const hiddenDim = this.model.hiddenDim;
    const effectiveK = Math.max(0, Math.min(k, numChunks));
    const dispatches: DispatchInfo[] = [];

    const queryBuf = createStorageBuffer(d, queryEmbedding, "query_embedding");
    const scoresBuf = d.createBuffer({ label: "scores", size: Math.max(4, numChunks * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    const simParams = createUniformBuffer(d, new Uint32Array([hiddenDim, numChunks, 0, 0]));
    const simBindGroup = d.createBindGroup({
      layout: this.similarityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: simParams } },
        { binding: 1, resource: { buffer: queryBuf } },
        { binding: 2, resource: { buffer: chunkEmbeddingsBuf } },
        { binding: 3, resource: { buffer: scoresBuf } },
      ],
    });

    const numWorkgroups = Math.max(1, Math.ceil(numChunks / 256));
    const partialValsBuf = d.createBuffer({ size: Math.max(4, numWorkgroups * 4), usage: GPUBufferUsage.STORAGE });
    const partialIdxsBuf = d.createBuffer({ size: Math.max(4, numWorkgroups * 4), usage: GPUBufferUsage.STORAGE });
    const topkIndicesBuf = d.createBuffer({ size: Math.max(4, effectiveK * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const topkScoresBuf = d.createBuffer({ size: Math.max(4, effectiveK * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    this.profiler.begin(1 + effectiveK * 2);
    const encoder = d.createCommandEncoder({ label: "search" });

    if (numChunks > 0) {
      const simGroups = Math.ceil(numChunks / 64);
      const simPass = encoder.beginComputePass({ label: "similarity", timestampWrites: this.profiler.passTimestampWrites("similarity") });
      simPass.setPipeline(this.similarityPipeline);
      simPass.setBindGroup(0, simBindGroup);
      simPass.dispatchWorkgroups(simGroups);
      simPass.end();
      dispatches.push({ kernel: "similarity", workgroupCount: simGroups, workgroupSize: 64 });
    }

    for (let round = 0; round < effectiveK; round++) {
      const roundParams = createUniformBuffer(d, new Uint32Array([numChunks, numWorkgroups, round, 0]));
      const bindGroup = d.createBindGroup({
        layout: this.topkBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: roundParams } },
          { binding: 1, resource: { buffer: scoresBuf } },
          { binding: 2, resource: { buffer: partialValsBuf } },
          { binding: 3, resource: { buffer: partialIdxsBuf } },
          { binding: 4, resource: { buffer: topkIndicesBuf } },
          { binding: 5, resource: { buffer: topkScoresBuf } },
        ],
      });

      const partialPass = encoder.beginComputePass({ label: `topk_partial[${round}]`, timestampWrites: this.profiler.passTimestampWrites(`topk_partial[${round}]`) });
      partialPass.setPipeline(this.topkPartialPipeline);
      partialPass.setBindGroup(0, bindGroup);
      partialPass.dispatchWorkgroups(numWorkgroups);
      partialPass.end();
      dispatches.push({ kernel: `topk_partial[${round}]`, workgroupCount: numWorkgroups, workgroupSize: 256 });

      const finalPass = encoder.beginComputePass({ label: `topk_final[${round}]`, timestampWrites: this.profiler.passTimestampWrites(`topk_final[${round}]`) });
      finalPass.setPipeline(this.topkFinalPipeline);
      finalPass.setBindGroup(0, bindGroup);
      finalPass.dispatchWorkgroups(1);
      finalPass.end();
      dispatches.push({ kernel: `topk_final[${round}]`, workgroupCount: 1, workgroupSize: 256 });
    }

    const idxSize = Math.max(4, effectiveK * 4);
    const idxReadback = d.createBuffer({ size: idxSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const scoreReadback = d.createBuffer({ size: idxSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(topkIndicesBuf, 0, idxReadback, 0, idxSize);
    encoder.copyBufferToBuffer(topkScoresBuf, 0, scoreReadback, 0, idxSize);
    this.profiler.resolve(encoder);
    d.queue.submit([encoder.finish()]);

    const timings = await this.profiler.readResults();

    await Promise.all([idxReadback.mapAsync(GPUMapMode.READ), scoreReadback.mapAsync(GPUMapMode.READ)]);
    const indices = new Uint32Array(idxReadback.getMappedRange().slice(0)).slice(0, effectiveK);
    const scores = new Float32Array(scoreReadback.getMappedRange().slice(0)).slice(0, effectiveK);
    idxReadback.unmap();
    scoreReadback.unmap();
    idxReadback.destroy();
    scoreReadback.destroy();

    queryBuf.destroy();
    scoresBuf.destroy();
    partialValsBuf.destroy();
    partialIdxsBuf.destroy();
    topkIndicesBuf.destroy();
    topkScoresBuf.destroy();

    this.lastDispatches = dispatches;
    return { indices, scores, timings };
  }
}
