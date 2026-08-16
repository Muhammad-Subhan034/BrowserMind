// Wraps WebGPU timestamp queries around compute passes so the "GPU
// Internals" panel can show real, measured per-kernel GPU time -- not a
// JS-side wall-clock estimate. Falls back to null timings (and the UI
// falls back to labeling them "unavailable") on adapters that don't
// expose the `timestamp-query` feature.

export interface PassTiming {
  label: string;
  gpuMicroseconds: number | null;
}

export class GpuProfiler {
  private device: GPUDevice;
  private supported: boolean;
  private querySet: GPUQuerySet | null = null;
  private resolveBuffer: GPUBuffer | null = null;
  private readbackBuffer: GPUBuffer | null = null;
  private labels: string[] = [];
  private capacity = 0;

  constructor(device: GPUDevice, supported: boolean) {
    this.device = device;
    this.supported = supported;
  }

  /** Must be called once per encoder, before recording any passes, with the number of passes you intend to time. */
  begin(passCount: number) {
    this.labels = [];
    if (!this.supported || passCount === 0) {
      this.querySet?.destroy();
      this.querySet = null;
      return;
    }
    if (passCount > this.capacity) {
      this.querySet?.destroy();
      this.resolveBuffer?.destroy();
      this.readbackBuffer?.destroy();
      this.capacity = passCount;
      this.querySet = this.device.createQuerySet({ type: "timestamp", count: passCount * 2 });
      this.resolveBuffer = this.device.createBuffer({
        size: passCount * 2 * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.readbackBuffer = this.device.createBuffer({
        size: passCount * 2 * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
  }

  /** Returns timestampWrites to attach to a GPUComputePassDescriptor, or undefined if unsupported. */
  passTimestampWrites(label: string): GPUComputePassTimestampWrites | undefined {
    if (!this.supported || !this.querySet) return undefined;
    const idx = this.labels.length;
    this.labels.push(label);
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: idx * 2,
      endOfPassWriteIndex: idx * 2 + 1,
    };
  }

  resolve(encoder: GPUCommandEncoder) {
    if (!this.supported || !this.querySet || !this.resolveBuffer || !this.readbackBuffer) return;
    const count = this.labels.length;
    if (count === 0) return;
    encoder.resolveQuerySet(this.querySet, 0, count * 2, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readbackBuffer, 0, count * 2 * 8);
  }

  async readResults(): Promise<PassTiming[]> {
    if (!this.supported || !this.readbackBuffer || this.labels.length === 0) {
      return this.labels.map((label) => ({ label, gpuMicroseconds: null }));
    }
    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const data = new BigInt64Array(this.readbackBuffer.getMappedRange().slice(0));
    this.readbackBuffer.unmap();

    const out: PassTiming[] = [];
    for (let i = 0; i < this.labels.length; i++) {
      const start = data[i * 2];
      const end = data[i * 2 + 1];
      const ns = Number(end - start);
      out.push({ label: this.labels[i], gpuMicroseconds: ns / 1000 });
    }
    return out;
  }
}
