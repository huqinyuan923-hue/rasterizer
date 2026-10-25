/**
 * M5 参考实现：透视校正纹理映射（独立编码）。
 * 场景：地面四边形（大掠射角，远端深度 8，近端 2）——透视校正的"swimming"检验场。
 * 参考实现自带：顶点投影（冻结公式）、逐像素重心 + (u/w, v/w, 1/w) 插值、独立采样器。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { mat4Perspective, mat4ApplyVec4 } from "../../src/math.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";

export const M5_W = 128;
export const M5_H = 128;

/** 相机：原点看 -Z，fov 60°，near 0.1 far 20。view 恒等。 */
export const M5_FOV = (60 * Math.PI) / 180;
const PROJ = mat4Perspective(M5_FOV, 1, 0.1, 20);

/** 地面四边形：y=-1，x∈[-2,2]，z∈[-2,-8]，uv [0,1]²。 */
export const M5_QUAD = {
  corners: [
    { x: -2, y: -1, z: -2, u: 0, v: 0 }, // A 近左
    { x: 2, y: -1, z: -2, u: 1, v: 0 },  // B 近右
    { x: 2, y: -1, z: -8, u: 1, v: 1 },  // C 远右
    { x: -2, y: -1, z: -8, u: 0, v: 1 }, // D 远左
  ],
  indices: [0, 1, 2, 0, 2, 3] as const,
};

export interface ProjVert { sx: number; sy: number; uOverW: number; vOverW: number; invW: number; dOverW: number }

/** 冻结的顶点投影公式（与 pipeline.ts 逐位一致）。 */
export function projectVertex(p: { x: number; y: number; z: number; u: number; v: number }): ProjVert {
  const c = mat4ApplyVec4(PROJ, { x: p.x, y: p.y, z: p.z, w: 1 });
  const w = c.w;
  return {
    sx: (c.x / w + 1) * 0.5 * M5_W,
    sy: (1 - c.y / w) * 0.5 * M5_H,
    uOverW: p.u / w,
    vOverW: p.v / w,
    invW: 1 / w,
    dOverW: -p.z / w,
  };
}

// ---- 参考采样器（独立实现，公式与规格冻结一致） ----

export function refSampleNearest(tw: number, th: number, data: Uint8ClampedArray, u: number, v: number, out: Float64Array): void {
  const uu = u - Math.floor(u);
  const vv = v - Math.floor(v);
  let tx = Math.floor(uu * tw);
  let ty = Math.floor(vv * th);
  if (tx >= tw) tx = tw - 1;
  if (ty >= th) ty = th - 1;
  const o = (ty * tw + tx) * 4;
  out[0] = data[o]; out[1] = data[o + 1]; out[2] = data[o + 2];
}

export function refSampleBilinear(tw: number, th: number, data: Uint8ClampedArray, u: number, v: number, out: Float64Array): void {
  const uu = u - Math.floor(u);
  const vv = v - Math.floor(v);
  const fx = uu * tw - 0.5;
  const fy = vv * th - 0.5;
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const tx = fx - x0, ty = fy - y0;
  const wrap = (n: number, m: number): number => ((n % m) + m) % m;
  const get = (x: number, y: number, k: number): number => data[(wrap(y, th) * tw + wrap(x, tw)) * 4 + k];
  for (let k = 0; k < 3; k++) {
    const c00 = get(x0, y0, k), c10 = get(x0 + 1, y0, k);
    const c01 = get(x0, y0 + 1, k), c11 = get(x0 + 1, y0 + 1, k);
    const row0 = c00 + (c10 - c00) * tx;
    const row1 = c01 + (c11 - c01) * tx;
    out[k] = row0 + (row1 - row0) * ty;
  }
}

/** 参考透视校正纹理三角形。 */
export function refTexturedTriangle(
  fb: Framebuffer,
  pv: [ProjVert, ProjVert, ProjVert],
  tex: { w: number; h: number; data: Uint8ClampedArray },
  bilinear: boolean,
): void {
  const [a0, b0, c0] = pv;
  const area = (b0.sx - a0.sx) * (c0.sy - a0.sy) - (b0.sy - a0.sy) * (c0.sx - a0.sx);
  if (!(Math.abs(area) > 1e-12)) return;
  let va = a0, vb = b0, vc = c0;
  if (area < 0) { vb = c0; vc = b0; }
  const A = area < 0 ? -area : area;

  const e0x = vb.sx - va.sx, e0y = vb.sy - va.sy;
  const e1x = vc.sx - vb.sx, e1y = vc.sy - vb.sy;
  const e2x = va.sx - vc.sx, e2y = va.sy - vc.sy;
  const tl = (dy: number, dx: number): boolean => dy < 0 || (dy === 0 && dx > 0);
  const tl0 = tl(e0y, e0x), tl1 = tl(e1y, e1x), tl2 = tl(e2y, e2x);

  const xs = Math.max(0, Math.ceil(Math.min(va.sx, vb.sx, vc.sx) - 0.5));
  const xe = Math.min(fb.width - 1, Math.floor(Math.max(va.sx, vb.sx, vc.sx) - 0.5));
  const ys = Math.max(0, Math.ceil(Math.min(va.sy, vb.sy, vc.sy) - 0.5));
  const ye = Math.min(fb.height - 1, Math.floor(Math.max(va.sy, vb.sy, vc.sy) - 0.5));

  const out = new Float64Array(3);
  for (let y = ys; y <= ye; y++) {
    for (let x = xs; x <= xe; x++) {
      const px = x + 0.5, py = y + 0.5;
      const E0 = e0x * (py - va.sy) - e0y * (px - va.sx);
      const E1 = e1x * (py - vb.sy) - e1y * (px - vb.sx);
      const E2 = e2x * (py - vc.sy) - e2y * (px - vc.sx);
      let inside: boolean;
      if (E0 > 0 && E1 > 0 && E2 > 0) inside = true;
      else if (E0 >= 0 && E1 >= 0 && E2 >= 0) inside = (E0 > 0 || tl0) && (E1 > 0 || tl1) && (E2 > 0 || tl2);
      else inside = false;
      if (!inside) continue;
      const l0 = E1 / A, l1 = E2 / A, l2 = E0 / A;
      const denom = l0 * va.invW + l1 * vb.invW + l2 * vc.invW;
      const u = (l0 * va.uOverW + l1 * vb.uOverW + l2 * vc.uOverW) / denom;
      const v = (l0 * va.vOverW + l1 * vb.vOverW + l2 * vc.vOverW) / denom;
      const d = (l0 * va.dOverW + l1 * vb.dOverW + l2 * vc.dOverW) / denom;
      const idx = y * fb.width + x;
      if (!(d < fb.depth[idx])) continue;
      if (bilinear) refSampleBilinear(tex.w, tex.h, tex.data, u, v, out);
      else refSampleNearest(tex.w, tex.h, tex.data, u, v, out);
      fb.depth[idx] = d;
      const o = idx * 4;
      fb.color[o] = out[0]; fb.color[o + 1] = out[1]; fb.color[o + 2] = out[2];
    }
  }
}

/** 8×8 棋盘 64px 纹理（测试与生成器共用夹具）。 */
export function makeM5Texture(): { w: number; h: number; data: Uint8ClampedArray } {
  const size = 64, cells = 8;
  const data = new Uint8ClampedArray(size * size * 4);
  const cell = size / cells;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cx = Math.min(cells - 1, Math.floor(x / cell));
      const cy = Math.min(cells - 1, Math.floor(y / cell));
      const c = (cx + cy) % 2 === 0 ? 235 : 45;
      const o = (y * size + x) * 4;
      data[o] = c; data[o + 1] = c; data[o + 2] = c; data[o + 3] = 255;
    }
  }
  return { w: size, h: size, data };
}

export function renderM5(bilinear: boolean): RawImage {
  const fb = new Framebuffer(M5_W, M5_H);
  const tex = makeM5Texture();
  const pv = M5_QUAD.corners.map(projectVertex);
  const idx = M5_QUAD.indices;
  refTexturedTriangle(fb, [pv[idx[0]], pv[idx[1]], pv[idx[2]]], tex, bilinear);
  refTexturedTriangle(fb, [pv[idx[3]], pv[idx[4]], pv[idx[5]]], tex, bilinear);
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("m5_tex_nearest.ppm", renderM5(false));
  writeGolden("m5_tex_bilinear.ppm", renderM5(true));
  console.log("m5_tex_nearest.ppm / m5_tex_bilinear.ppm written");
}
