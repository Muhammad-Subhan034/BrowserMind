// Hand-written GLSL/WebGL2 embedding-space scatter plot. Deliberately a
// separate rendering pipeline from the WGSL compute work elsewhere in the
// app (see gpu/shaders/*.wgsl) -- this file only ever touches WebGL2's
// GLSL ES 3.00 shading language, to show both GPU shading languages the
// project targets are genuinely hand-written, not just one reused twice.

const VERTEX_SRC = `#version 300 es
precision highp float;

layout(location = 0) in vec2 aPosition;
layout(location = 1) in float aCluster;
layout(location = 2) in float aId;

uniform mat3 uTransform;
uniform float uPointSize;
uniform float uTime;
uniform float uSelectedId;
uniform vec3 uPalette[8];

out vec3 vColor;
out float vSelected;
out float vGlow;

void main() {
  vec3 pos = uTransform * vec3(aPosition, 1.0);
  gl_Position = vec4(pos.xy, 0.0, 1.0);

  int idx = int(mod(aCluster, 8.0));
  vColor = uPalette[idx];

  float isSelected = step(abs(aId - uSelectedId), 0.5);
  vSelected = isSelected;

  // Gentle per-point pulse, phase-offset by id so the whole field feels
  // alive rather than synchronized -- subtle, not distracting.
  float phase = fract(sin(aId * 12.9898) * 43758.5453);
  vGlow = 0.85 + 0.15 * sin(uTime * 1.3 + phase * 6.28318);

  gl_PointSize = uPointSize * (1.0 + isSelected * 0.9) * vGlow;
}
`;

const FRAGMENT_SRC = `#version 300 es
precision highp float;

in vec3 vColor;
in float vSelected;
in float vGlow;

out vec4 fragColor;

void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float d = length(c);
  if (d > 1.0) discard;

  float core = smoothstep(1.0, 0.35, d);
  float ring = vSelected * smoothstep(0.75, 0.62, d) * smoothstep(0.55, 0.68, d);

  vec3 color = mix(vColor, vec3(1.0), ring);
  float alpha = clamp(core * vGlow + ring, 0.0, 1.0);
  fragColor = vec4(color * alpha, alpha);
}
`;

const PALETTE: [number, number, number][] = [
  [0.42, 0.62, 1.0],
  [1.0, 0.55, 0.38],
  [0.45, 0.9, 0.68],
  [0.95, 0.45, 0.75],
  [0.85, 0.78, 0.35],
  [0.6, 0.5, 1.0],
  [0.35, 0.85, 0.9],
  [1.0, 0.4, 0.4],
];

export interface ScatterPoint {
  x: number;
  y: number;
  cluster: number;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`GLSL compile error: ${log}`);
  }
  return shader;
}

export class ScatterRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private positionBuf: WebGLBuffer;
  private clusterBuf: WebGLBuffer;
  private idBuf: WebGLBuffer;
  private uTransform: WebGLUniformLocation;
  private uPointSize: WebGLUniformLocation;
  private uTime: WebGLUniformLocation;
  private uSelectedId: WebGLUniformLocation;
  private uPalette: WebGLUniformLocation;

  private count = 0;
  private pan = { x: 0, y: 0 };
  private zoom = 1;
  private selectedId = -1;
  private rafHandle = 0;
  private points: ScatterPoint[] = [];
  private canvas: HTMLCanvasElement;
  private onPick: ((index: number) => void) | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { antialias: true, alpha: true, premultipliedAlpha: true });
    if (!gl) throw new Error("WebGL2 is not available in this browser.");
    this.gl = gl;

    const vs = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SRC);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SRC);
    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`GLSL link error: ${gl.getProgramInfoLog(program)}`);
    }
    this.program = program;

    this.vao = gl.createVertexArray()!;
    this.positionBuf = gl.createBuffer()!;
    this.clusterBuf = gl.createBuffer()!;
    this.idBuf = gl.createBuffer()!;

    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.clusterBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.idBuf);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    this.uTransform = gl.getUniformLocation(program, "uTransform")!;
    this.uPointSize = gl.getUniformLocation(program, "uPointSize")!;
    this.uTime = gl.getUniformLocation(program, "uTime")!;
    this.uSelectedId = gl.getUniformLocation(program, "uSelectedId")!;
    this.uPalette = gl.getUniformLocation(program, "uPalette")!;

    gl.useProgram(program);
    const flat = new Float32Array(PALETTE.flat());
    gl.uniform3fv(this.uPalette, flat);

    this.attachInteraction();
    this.loop(0);
  }

  setPoints(points: ScatterPoint[]) {
    this.points = points;
    const gl = this.gl;
    this.count = points.length;

    const positions = new Float32Array(this.count * 2);
    const clusters = new Float32Array(this.count);
    const ids = new Float32Array(this.count);

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of points) {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    }
    const spanX = Math.max(maxX - minX, 1e-6);
    const spanY = Math.max(maxY - minY, 1e-6);
    const span = Math.max(spanX, spanY);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;

    for (let i = 0; i < points.length; i++) {
      positions[i * 2] = ((points[i].x - cx) / span) * 1.7;
      positions[i * 2 + 1] = ((points[i].y - cy) / span) * 1.7;
      clusters[i] = points[i].cluster;
      ids[i] = i;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuf);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.clusterBuf);
    gl.bufferData(gl.ARRAY_BUFFER, clusters, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.idBuf);
    gl.bufferData(gl.ARRAY_BUFFER, ids, gl.DYNAMIC_DRAW);

    this.pan = { x: 0, y: 0 };
    this.zoom = 1;
  }

  setOnPick(cb: (index: number) => void) {
    this.onPick = cb;
  }

  setSelected(index: number) {
    this.selectedId = index;
  }

  private attachInteraction() {
    const canvas = this.canvas;
    let dragging = false;
    let lastX = 0, lastY = 0;

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const factor = Math.exp(-e.deltaY * 0.001);
      this.zoom = Math.min(40, Math.max(0.2, this.zoom * factor));
    }, { passive: false });

    canvas.addEventListener("pointerdown", (e) => {
      dragging = true;
      lastX = e.clientX; lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const rect = canvas.getBoundingClientRect();
      this.pan.x += ((e.clientX - lastX) / rect.width) * 2;
      this.pan.y -= ((e.clientY - lastY) / rect.height) * 2;
      lastX = e.clientX; lastY = e.clientY;
    });
    canvas.addEventListener("pointerup", (e) => {
      dragging = false;
      canvas.releasePointerCapture(e.pointerId);
    });
    canvas.addEventListener("click", (e) => {
      if (!this.onPick || this.points.length === 0) return;
      const rect = canvas.getBoundingClientRect();
      const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
      // Invert the (translate, scale) transform applied in the vertex shader.
      const dataX = (ndcX - this.pan.x) / this.zoom;
      const dataY = (ndcY - this.pan.y) / this.zoom;

      const positions = this.lastNormalizedPositions();
      let best = -1, bestDist = Infinity;
      for (let i = 0; i < positions.length / 2; i++) {
        const dx = positions[i * 2] - dataX;
        const dy = positions[i * 2 + 1] - dataY;
        const dist = dx * dx + dy * dy;
        if (dist < bestDist) { bestDist = dist; best = i; }
      }
      if (best >= 0 && bestDist < 0.01) this.onPick(best);
    });
  }

  private normalizedPositionsCache: Float32Array | null = null;
  private lastNormalizedPositions(): Float32Array {
    if (this.normalizedPositionsCache && this.normalizedPositionsCache.length === this.points.length * 2) {
      return this.normalizedPositionsCache;
    }
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of this.points) {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    }
    const span = Math.max(Math.max(maxX - minX, maxY - minY), 1e-6);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const out = new Float32Array(this.points.length * 2);
    for (let i = 0; i < this.points.length; i++) {
      out[i * 2] = ((this.points[i].x - cx) / span) * 1.7;
      out[i * 2 + 1] = ((this.points[i].y - cy) / span) * 1.7;
    }
    this.normalizedPositionsCache = out;
    return out;
  }

  private resize() {
    const gl = this.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
  }

  private loop = (t: number) => {
    this.resize();
    const gl = this.gl;
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (this.count > 0) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(this.program);
      gl.bindVertexArray(this.vao);

      const transform = new Float32Array([
        this.zoom, 0, 0,
        0, this.zoom, 0,
        this.pan.x, this.pan.y, 1,
      ]);
      gl.uniformMatrix3fv(this.uTransform, false, transform);
      gl.uniform1f(this.uPointSize, Math.max(4, Math.min(14, 400 / Math.sqrt(this.count))) * (window.devicePixelRatio || 1));
      gl.uniform1f(this.uTime, t / 1000);
      gl.uniform1f(this.uSelectedId, this.selectedId);

      gl.drawArrays(gl.POINTS, 0, this.count);
      gl.bindVertexArray(null);
    }

    this.rafHandle = requestAnimationFrame(this.loop);
  };

  dispose() {
    cancelAnimationFrame(this.rafHandle);
  }
}
