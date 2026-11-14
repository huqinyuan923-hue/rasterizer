/**
 * 纹理与采样器。约定（冻结）：
 * - UV 语义为 repeat 环绕：uu = u - floor(u)。
 * - nearest：tx = floor(uu * W)，钳位到 [0, W-1]。
 * - bilinear：fx = uu*W - 0.5，x0 = floor(fx)，权重 t = fx - x0，索引 repeat 环绕；
 *   先横向后纵向 lerp（a + (b-a)*t）。
 */

export interface Texture {
  width: number;
  height: number;
  data: Uint8ClampedArray; // RGBA
  /** mip 一档（半分辨率 box filter），由 buildMip1 构建；缺省 = 无 mip。 */
  mip?: Texture;
}

export type SamplerKind = "nearest" | "bilinear";

/** 棋盘格纹理：cells × cells 格，两色 a/b。 */
export function checkerTexture(size: number, cells: number, a: [number, number, number], b: [number, number, number]): Texture {
  const data = new Uint8ClampedArray(size * size * 4);
  const cell = size / cells;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cx = Math.min(cells - 1, Math.floor(x / cell));
      const cy = Math.min(cells - 1, Math.floor(y / cell));
      const c = (cx + cy) % 2 === 0 ? a : b;
      const o = (y * size + x) * 4;
      data[o] = c[0]; data[o + 1] = c[1]; data[o + 2] = c[2]; data[o + 3] = 255;
    }
  }
  return { width: size, height: size, data };
}

function texel(t: Texture, x: number, y: number, out: Float64Array): void {
  const o = (y * t.width + x) * 4;
  out[0] = t.data[o];
  out[1] = t.data[o + 1];
  out[2] = t.data[o + 2];
}

const T00 = new Float64Array(3);
const T10 = new Float64Array(3);
const T01 = new Float64Array(3);
const T11 = new Float64Array(3);

/** 最近邻采样（repeat 环绕）。 */
export function sampleNearest(t: Texture, u: number, v: number, out: Float64Array): void {
  const uu = u - Math.floor(u);
  const vv = v - Math.floor(v);
  let tx = Math.floor(uu * t.width);
  let ty = Math.floor(vv * t.height);
  if (tx >= t.width) tx = t.width - 1;
  if (ty >= t.height) ty = t.height - 1;
  texel(t, tx, ty, out);
}

/** 双线性采样（repeat 环绕）。 */
export function sampleBilinear(t: Texture, u: number, v: number, out: Float64Array): void {
  const uu = u - Math.floor(u);
  const vv = v - Math.floor(v);
  const fx = uu * t.width - 0.5;
  const fy = vv * t.height - 0.5;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;
  const W = t.width, H = t.height;
  const x0w = ((x0 % W) + W) % W;
  const x1w = (((x0 + 1) % W) + W) % W;
  const y0w = ((y0 % H) + H) % H;
  const y1w = (((y0 + 1) % H) + H) % H;
  texel(t, x0w, y0w, T00);
  texel(t, x1w, y0w, T10);
  texel(t, x0w, y1w, T01);
  texel(t, x1w, y1w, T11);
  for (let k = 0; k < 3; k++) {
    const row0 = T00[k] + (T10[k] - T00[k]) * tx;
    const row1 = T01[k] + (T11[k] - T01[k]) * tx;
    out[k] = row0 + (row1 - row0) * ty;
  }
}

/** 按采样器种类分发。 */
export function sample(t: Texture, kind: SamplerKind, u: number, v: number, out: Float64Array): void {
  if (kind === "bilinear") sampleBilinear(t, u, v, out);
  else sampleNearest(t, u, v, out);
}

/**
 * 构建 mip 一档（二期 M2）：半分辨率、2×2 box filter（RGBA 四通道都平均）。
 * 冻结公式：data = (a + b + c + d) / 4，直接赋给 Uint8ClampedArray（四舍五入量化）。
 * 奇数尺寸：丢弃末行/末列（mw = floor(w/2)、mh = floor(h/2），至少 1）。
 * 结果纹理的 mip 字段不设置（一层即到顶）。
 */
export function buildMip1(t: Texture): Texture {
  const mw = Math.max(1, Math.floor(t.width / 2));
  const mh = Math.max(1, Math.floor(t.height / 2));
  const data = new Uint8ClampedArray(mw * mh * 4);
  const tw = t.width;
  for (let y = 0; y < mh; y++) {
    for (let x = 0; x < mw; x++) {
      const o = (y * mw + x) * 4;
      const o00 = ((y * 2) * tw + (x * 2)) * 4;
      const o10 = ((y * 2) * tw + (x * 2 + 1)) * 4;
      const o01 = ((y * 2 + 1) * tw + (x * 2)) * 4;
      const o11 = ((y * 2 + 1) * tw + (x * 2 + 1)) * 4;
      for (let c = 0; c < 4; c++) {
        data[o + c] = (t.data[o00 + c] + t.data[o10 + c] + t.data[o01 + c] + t.data[o11 + c]) / 4;
      }
    }
  }
  return { width: mw, height: mh, data };
}
