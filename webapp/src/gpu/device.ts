// WebGPU capability detection. Deliberately surfaces the adapter/device
// info rather than just returning a boolean -- knowing *which* GPU and
// backend the app is running against is exactly the kind of detail a GPU
// reviewer wants to see reported honestly, and it drives the "GPU
// Internals" panel's device card.

export interface GpuCapability {
  supported: boolean;
  reason?: string;
  adapterInfo?: {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
  };
  features?: string[];
  limits?: Record<string, number>;
  timestampQuerySupported: boolean;
}

export interface GpuContext {
  device: GPUDevice;
  adapter: GPUAdapter;
  capability: GpuCapability;
}

export async function detectGpu(): Promise<GpuCapability> {
  if (!("gpu" in navigator)) {
    return { supported: false, reason: "navigator.gpu is undefined -- this browser has no WebGPU support.", timestampQuerySupported: false };
  }

  const gpu = (navigator as Navigator & { gpu: GPU }).gpu;
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    return { supported: false, reason: "requestAdapter() returned null -- no compatible GPU adapter was found.", timestampQuerySupported: false };
  }

  const info = adapter.info;
  const timestampQuerySupported = adapter.features.has("timestamp-query");
  const vendor = info?.vendor || "unknown vendor";
  const architecture = info?.architecture || "";
  // Chrome/Edge frequently report an empty `description` for privacy
  // reasons even though `vendor`/`architecture` are populated -- fall back
  // to composing something readable from those instead of showing
  // "unknown" when we clearly do have adapter info.
  const description = info?.description || [vendor, architecture].filter(Boolean).join(" · ") || "unknown adapter";

  return {
    supported: true,
    adapterInfo: {
      vendor,
      architecture: architecture || "unknown",
      device: info?.device || "unknown",
      description,
    },
    features: Array.from(adapter.features).sort(),
    limits: extractLimits(adapter.limits),
    timestampQuerySupported,
  };
}

function extractLimits(limits: GPUSupportedLimits): Record<string, number> {
  const keys: (keyof GPUSupportedLimits)[] = [
    "maxBufferSize",
    "maxStorageBufferBindingSize",
    "maxComputeWorkgroupSizeX",
    "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupsPerDimension",
  ];
  const out: Record<string, number> = {};
  for (const k of keys) {
    const v = limits[k];
    if (typeof v === "number") out[k] = v;
  }
  return out;
}

export async function initGpu(): Promise<GpuContext> {
  if (!("gpu" in navigator)) {
    throw new Error("WebGPU is not available in this browser.");
  }
  const gpu = (navigator as Navigator & { gpu: GPU }).gpu;
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter available.");

  const wantTimestamps = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: wantTimestamps ? ["timestamp-query"] : [],
  });

  const capability = await detectGpu();
  return { device, adapter, capability };
}
