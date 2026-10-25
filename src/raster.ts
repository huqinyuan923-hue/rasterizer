/**
 * 光栅化核心。
 *
 * 约定（冻结，勿改）：
 * - 像素 (i,j) 的采样点在 (i+0.5, j+0.5)。
 * - 屏幕坐标 y 向下；三角形先归一化到正有向面积（area2 > 0，视觉顺时针）。
 * - 边函数 E_k(p) = dxk*(p.y - v_k.y) - dyk*(p.x - v_k.x)，边依次 v0→v1, v1→v2, v2→v0。
 *   内部点满足全部 E > 0；重心坐标 λ0 = E1/A, λ1 = E2/A, λ2 = E0/A（A = |area2|）。
 * - top-left 规则：E == 0 的采样点仅当该边是 top（水平且 dx > 0）或 left（dy < 0）时包含。
 *   该规则对共享边"总和且互斥"（相邻三角形反向遍历共享边，恰好一方包含），保证无缝隙无重叠。
 * - 深度：d = (Σλᵢ dOverWᵢ) / (Σλᵢ invWᵢ)，直接除法（不用倒数乘法），invW≡1 时退化为屏幕空间线性 z。
 *   深度测试 d < depth[idx]（严格小于，同深度先画者赢），通过后写回。
 * - 所有插值一律"分子/分母"直接除法，保证与参考实现逐位一致。
 */

/** 最小像素写入接口（Framebuffer 满足它；测试可用记录器）。 */
export interface PixelSink {
  setPixel(x: number, y: number, r: number, g: number, b: number): void;
}

import type { Framebuffer } from "./framebuffer.js";
import type { Texture } from "./texture.js";
import { powInt, type LightSetup } from "./shading.js";
import type { Vec3 } from "./math.js";

/** Bresenham 直线，全象限整数实现，含边界 guard（越界像素由 setPixel 丢弃）。 */
export function drawLine(
  fb: PixelSink, x0: number, y0: number, x1: number, y1: number,
  r: number, g: number, b: number,
): void {
  let x = x0, y = y0;
  const dx = Math.abs(x1 - x0);
  const sx = x0 < x1 ? 1 : -1;
  const dy = -Math.abs(y1 - y0);
  const sy = y0 < y1 ? 1 : -1;
  let err = dx + dy;
  for (;;) {
    fb.setPixel(x, y, r, g, b);
    const e2 = 2 * err;
    if (e2 >= dy) {
      if (x === x1) break;
      err += dy;
      x += sx;
    }
    if (e2 <= dx) {
      if (y === y1) break;
      err += dx;
      y += sy;
    }
  }
}

/**
 * 屏幕空间顶点：位置 + 透视校正插值所需的预乘量。
 */
export interface ScreenVert {
  x: number;
  y: number;
  invW: number;         // 1/w_clip；仿射绘制（无透视）时为 1
  dOverW: number;       // 视距 d × invW；d = -z_view（正值，越小越近）
  aOverW: Float64Array; // varying × invW
}

export interface RasterOptions {
  depthTest: boolean;
  nearDist: number; // depthTest 时 d < nearDist 的片段被丢弃（防近平面泄漏）
}

export const MAX_ATTRS = 8;
const SCRATCH_ATTRS = new Float64Array(MAX_ATTRS);
const SCRATCH_RGB = new Float64Array(3);

export type FragmentShader = (
  lam0: number, lam1: number, lam2: number,
  denom: number,
  d: number,
  attrs: Float64Array, // 长度 attrCount，插值结果
  rgb: Float64Array,   // 出参：r,g,b ∈ [0,255]（float，核心负责量化）
) => void;

// ---- 着色专用快速路径（pipeline 使用）：公式与 texture.ts / shading.ts 同源冻结， ----
// ---- 采样/光照内联进循环体以消除逐像素调用开销；黄金图逐位仲裁一致性。       ----

export type ShadeKind = "textured" | "gouraud" | "phong";

export interface ShaderSpec {
  kind: ShadeKind;
  texture?: Texture;      // textured / phong+uv
  bilinear?: boolean;     // 采样器选择
  eye?: Vec3;             // phong
  lights?: LightSetup;    // phong
  ar: number; ag: number; ab: number;   // albedo 0..1
  specular: number;
  shininess: number;
}

/**
 * 光栅化单个三角形（屏幕空间）。返回写入的像素数（供覆盖率测试）。
 * 不做背面剔除 —— 剔除是调用方（pipeline）基于同一 area2 公式的决策。
 */
export function rasterizeTriangle(
  fb: Framebuffer,
  v0: ScreenVert, v1: ScreenVert, v2: ScreenVert,
  attrCount: number,
  opts: RasterOptions,
  shade: FragmentShader,
): number {
  const area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
  if (!(Math.abs(area) > 1e-12)) return 0;
  // 归一化到正面积；属性随顶点一起换序
  let va = v0, vb = v1, vc = v2;
  if (area < 0) { vb = v2; vc = v1; }
  const A = area < 0 ? -area : area;

  const e0x = vb.x - va.x, e0y = vb.y - va.y;
  const e1x = vc.x - vb.x, e1y = vc.y - vb.y;
  const e2x = va.x - vc.x, e2y = va.y - vc.y;
  const tl0 = e0y < 0 || (e0y === 0 && e0x > 0);
  const tl1 = e1y < 0 || (e1y === 0 && e1x > 0);
  const tl2 = e2y < 0 || (e2y === 0 && e2x > 0);

  const minX = Math.min(va.x, vb.x, vc.x);
  const maxX = Math.max(va.x, vb.x, vc.x);
  const minY = Math.min(va.y, vb.y, vc.y);
  const maxY = Math.max(va.y, vb.y, vc.y);
  if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) return 0;
  const xs = Math.max(0, Math.ceil(minX - 0.5));
  const xe = Math.min(fb.width - 1, Math.floor(maxX - 0.5));
  const ys = Math.max(0, Math.ceil(minY - 0.5));
  const ye = Math.min(fb.height - 1, Math.floor(maxY - 0.5));
  if (xs > xe || ys > ye) return 0;

  const needDenom = opts.depthTest || attrCount > 0;
  const vaW = va.invW, vbW = vb.invW, vcW = vc.invW;
  const vaD = va.dOverW, vbD = vb.dOverW, vcD = vc.dOverW;
  const vaA = va.aOverW, vbA = vb.aOverW, vcA = vc.aOverW;
  const fbColor = fb.color, fbDepth = fb.depth, fbW = fb.width;
  let count = 0;
  for (let y = ys; y <= ye; y++) {
    const py = y + 0.5;
    let idx = y * fbW + xs;
    for (let x = xs; x <= xe; x++, idx++) {
      const px = x + 0.5;
      // 逐像素直接求值（冻结语义）：黄金场景中存在精确 E==0 的平局像素，
      // 增量累积会破坏这些精确零 —— 性能优化不得改变求值方式。
      const E0 = e0x * (py - va.y) - e0y * (px - va.x);
      const E1 = e1x * (py - vb.y) - e1y * (px - vb.x);
      const E2 = e2x * (py - vc.y) - e2y * (px - vc.x);
      let inside: boolean;
      if (E0 > 0 && E1 > 0 && E2 > 0) {
        inside = true;
      } else if (E0 >= 0 && E1 >= 0 && E2 >= 0) {
        inside = (E0 > 0 || tl0) && (E1 > 0 || tl1) && (E2 > 0 || tl2);
      } else {
        inside = false;
      }
      if (!inside) continue;
      const lam0 = E1 / A;
      const lam1 = E2 / A;
      const lam2 = E0 / A;
      if (needDenom) {
        const denom = lam0 * vaW + lam1 * vbW + lam2 * vcW;
        if (opts.depthTest) {
          const d = (lam0 * vaD + lam1 * vbD + lam2 * vcD) / denom;
          if (!(d >= opts.nearDist) || !(d < fbDepth[idx])) continue;
          for (let j = 0; j < attrCount; j++) {
            SCRATCH_ATTRS[j] = (lam0 * vaA[j] + lam1 * vbA[j] + lam2 * vcA[j]) / denom;
          }
          shade(lam0, lam1, lam2, denom, d, SCRATCH_ATTRS, SCRATCH_RGB);
          fbDepth[idx] = d;
        } else {
          for (let j = 0; j < attrCount; j++) {
            SCRATCH_ATTRS[j] = (lam0 * vaA[j] + lam1 * vbA[j] + lam2 * vcA[j]) / denom;
          }
          shade(lam0, lam1, lam2, denom, 0, SCRATCH_ATTRS, SCRATCH_RGB);
        }
      } else {
        shade(lam0, lam1, lam2, 1, 0, SCRATCH_ATTRS, SCRATCH_RGB);
      }
      const o = idx * 4;
      fbColor[o] = SCRATCH_RGB[0];
      fbColor[o + 1] = SCRATCH_RGB[1];
      fbColor[o + 2] = SCRATCH_RGB[2];
      count++;
    }
  }
  return count;
}

const NO_ATTRS = new Float64Array(0);

function affineVert(x: number, y: number): ScreenVert {
  return { x, y, invW: 1, dOverW: 0, aOverW: NO_ATTRS };
}

/** 纯色填充（无深度）。z 参数不参与 —— 供 M3 黄金与覆盖率测试。 */
export function fillTriangleFlat(
  fb: Framebuffer,
  x0: number, y0: number, x1: number, y1: number, x2: number, y2: number,
  r: number, g: number, b: number,
): number {
  return rasterizeTriangle(
    fb, affineVert(x0, y0), affineVert(x1, y1), affineVert(x2, y2),
    0, { depthTest: false, nearDist: 0 },
    (_l0, _l1, _l2, _denom, _d, _attrs, rgb) => { rgb[0] = r; rgb[1] = g; rgb[2] = b; },
  );
}

/**
 * 屏幕空间线性 z 插值的深度填充（M4）。
 * z 为视空间距离 d = -z_view（正值，越小越近）；invW ≡ 1 ⟹ d = Σλᵢzᵢ（屏幕空间线性）。
 */
export function fillTriangleDepth(
  fb: Framebuffer,
  x0: number, y0: number, z0: number,
  x1: number, y1: number, z1: number,
  x2: number, y2: number, z2: number,
  r: number, g: number, b: number,
): number {
  const v0 = { x: x0, y: y0, invW: 1, dOverW: z0, aOverW: NO_ATTRS };
  const v1 = { x: x1, y: y1, invW: 1, dOverW: z1, aOverW: NO_ATTRS };
  const v2 = { x: x2, y: y2, invW: 1, dOverW: z2, aOverW: NO_ATTRS };
  return rasterizeTriangle(
    fb, v0, v1, v2,
    0, { depthTest: true, nearDist: 0 },
    (_l0, _l1, _l2, _denom, _d, _attrs, rgb) => { rgb[0] = r; rgb[1] = g; rgb[2] = b; },
  );
}

const INV255 = 1 / 255;

/**
 * 着色专用快速路径：与 rasterizeTriangle 完全相同的逐像素数学（逐位一致），
 * 但采样/光照公式手动内联进循环体 —— 消除每像素的回调、scratch 往返与跨模块调用。
 * 公式与 texture.ts / shading.ts 同源冻结，黄金图对拍仲裁一致性。
 * 返回写入的像素数。
 */
export function rasterizeShaded(
  fb: Framebuffer,
  v0: ScreenVert, v1: ScreenVert, v2: ScreenVert,
  opts: RasterOptions,
  spec: ShaderSpec,
): number {
  const area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
  if (!(Math.abs(area) > 1e-12)) return 0;
  let va = v0, vb = v1, vc = v2;
  if (area < 0) { vb = v2; vc = v1; }
  const A = area < 0 ? -area : area;

  const e0x = vb.x - va.x, e0y = vb.y - va.y;
  const e1x = vc.x - vb.x, e1y = vc.y - vb.y;
  const e2x = va.x - vc.x, e2y = va.y - vc.y;
  const tl0 = e0y < 0 || (e0y === 0 && e0x > 0);
  const tl1 = e1y < 0 || (e1y === 0 && e1x > 0);
  const tl2 = e2y < 0 || (e2y === 0 && e2x > 0);

  const minX = Math.min(va.x, vb.x, vc.x);
  const maxX = Math.max(va.x, vb.x, vc.x);
  const minY = Math.min(va.y, vb.y, vc.y);
  const maxY = Math.max(va.y, vb.y, vc.y);
  if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) return 0;
  const xs = Math.max(0, Math.ceil(minX - 0.5));
  const xe = Math.min(fb.width - 1, Math.floor(maxX - 0.5));
  const ys = Math.max(0, Math.ceil(minY - 0.5));
  const ye = Math.min(fb.height - 1, Math.floor(maxY - 0.5));
  if (xs > xe || ys > ye) return 0;

  const vaW = va.invW, vbW = vb.invW, vcW = vc.invW;
  const vaD = va.dOverW, vbD = vb.dOverW, vcD = vc.dOverW;
  const vaA = va.aOverW, vbA = vb.aOverW, vcA = vc.aOverW;
  const fbColor = fb.color, fbDepth = fb.depth, fbW = fb.width;
  const depthTest = opts.depthTest, nearDist = opts.nearDist;
  const kind = spec.kind;
  const tex = spec.texture, bilinear = spec.bilinear === true;
  const specAr = spec.ar, specAg = spec.ag, specAb = spec.ab;
  const hasTexAlbedo = kind === "phong" && tex !== undefined;
  const eye = spec.eye, lights = spec.lights;

  let count = 0;
  for (let y = ys; y <= ye; y++) {
    const py = y + 0.5;
    let idx = y * fbW + xs;
    for (let x = xs; x <= xe; x++, idx++) {
      const px = x + 0.5;
      const E0 = e0x * (py - va.y) - e0y * (px - va.x);
      const E1 = e1x * (py - vb.y) - e1y * (px - vb.x);
      const E2 = e2x * (py - vc.y) - e2y * (px - vc.x);
      let inside: boolean;
      if (E0 > 0 && E1 > 0 && E2 > 0) {
        inside = true;
      } else if (E0 >= 0 && E1 >= 0 && E2 >= 0) {
        inside = (E0 > 0 || tl0) && (E1 > 0 || tl1) && (E2 > 0 || tl2);
      } else {
        inside = false;
      }
      if (!inside) continue;
      const lam0 = E1 / A;
      const lam1 = E2 / A;
      const lam2 = E0 / A;
      const denom = lam0 * vaW + lam1 * vbW + lam2 * vcW;
      if (depthTest) {
        const d = (lam0 * vaD + lam1 * vbD + lam2 * vcD) / denom;
        if (!(d >= nearDist) || !(d < fbDepth[idx])) continue;
        fbDepth[idx] = d;
      }

      let r: number, g: number, b: number;
      if (kind === "gouraud") {
        r = (lam0 * vaA[0] + lam1 * vbA[0] + lam2 * vcA[0]) / denom;
        g = (lam0 * vaA[1] + lam1 * vbA[1] + lam2 * vcA[1]) / denom;
        b = (lam0 * vaA[2] + lam1 * vbA[2] + lam2 * vcA[2]) / denom;
      } else if (kind === "textured") {
        const u = (lam0 * vaA[0] + lam1 * vbA[0] + lam2 * vcA[0]) / denom;
        const v = (lam0 * vaA[1] + lam1 * vbA[1] + lam2 * vcA[1]) / denom;
        // 内联采样（与 texture.ts 冻结公式一致）
        const tw = tex!.width, th = tex!.height, tdata = tex!.data;
        const uu = u - Math.floor(u);
        const vv = v - Math.floor(v);
        if (bilinear) {
          const fx = uu * tw - 0.5;
          const fy = vv * th - 0.5;
          const x0 = Math.floor(fx), y0 = Math.floor(fy);
          const tx = fx - x0, ty = fy - y0;
          const x0w = ((x0 % tw) + tw) % tw;
          const x1w = (((x0 + 1) % tw) + tw) % tw;
          const y0w = ((y0 % th) + th) % th;
          const y1w = (((y0 + 1) % th) + th) % th;
          const o00 = (y0w * tw + x0w) * 4, o10 = (y0w * tw + x1w) * 4;
          const o01 = (y1w * tw + x0w) * 4, o11 = (y1w * tw + x1w) * 4;
          r = 0; g = 0; b = 0;
          const c00r = tdata[o00], c10r = tdata[o10], c01r = tdata[o01], c11r = tdata[o11];
          const row0r = c00r + (c10r - c00r) * tx;
          const row1r = c01r + (c11r - c01r) * tx;
          r = row0r + (row1r - row0r) * ty;
          const c00g = tdata[o00 + 1], c10g = tdata[o10 + 1], c01g = tdata[o01 + 1], c11g = tdata[o11 + 1];
          const row0g = c00g + (c10g - c00g) * tx;
          const row1g = c01g + (c11g - c01g) * tx;
          g = row0g + (row1g - row0g) * ty;
          const c00b = tdata[o00 + 2], c10b = tdata[o10 + 2], c01b = tdata[o01 + 2], c11b = tdata[o11 + 2];
          const row0b = c00b + (c10b - c00b) * tx;
          const row1b = c01b + (c11b - c01b) * tx;
          b = row0b + (row1b - row0b) * ty;
        } else {
          let tx = Math.floor(uu * tw);
          let ty = Math.floor(vv * th);
          if (tx >= tw) tx = tw - 1;
          if (ty >= th) ty = th - 1;
          const o = (ty * tw + tx) * 4;
          r = tdata[o]; g = tdata[o + 1]; b = tdata[o + 2];
        }
      } else {
        // phong：世界坐标 + 法线（+可选 uv）
        const pxx = (lam0 * vaA[0] + lam1 * vbA[0] + lam2 * vcA[0]) / denom;
        const pyy = (lam0 * vaA[1] + lam1 * vbA[1] + lam2 * vcA[1]) / denom;
        const pzz = (lam0 * vaA[2] + lam1 * vbA[2] + lam2 * vcA[2]) / denom;
        let nx = (lam0 * vaA[3] + lam1 * vbA[3] + lam2 * vcA[3]) / denom;
        let ny = (lam0 * vaA[4] + lam1 * vbA[4] + lam2 * vcA[4]) / denom;
        let nz = (lam0 * vaA[5] + lam1 * vbA[5] + lam2 * vcA[5]) / denom;
        let ar = specAr, ag = specAg, ab = specAb;
        if (hasTexAlbedo) {
          const u = (lam0 * vaA[6] + lam1 * vbA[6] + lam2 * vcA[6]) / denom;
          const v = (lam0 * vaA[7] + lam1 * vbA[7] + lam2 * vcA[7]) / denom;
          const tw = tex!.width, th = tex!.height, tdata = tex!.data;
          const uu = u - Math.floor(u);
          const vv = v - Math.floor(v);
          let tr: number, tg: number, tb: number;
          if (bilinear) {
            const fx = uu * tw - 0.5;
            const fy = vv * th - 0.5;
            const x0 = Math.floor(fx), y0 = Math.floor(fy);
            const tx = fx - x0, ty = fy - y0;
            const x0w = ((x0 % tw) + tw) % tw;
            const x1w = (((x0 + 1) % tw) + tw) % tw;
            const y0w = ((y0 % th) + th) % th;
            const y1w = (((y0 + 1) % th) + th) % th;
            const o00 = (y0w * tw + x0w) * 4, o10 = (y0w * tw + x1w) * 4;
            const o01 = (y1w * tw + x0w) * 4, o11 = (y1w * tw + x1w) * 4;
            const c00r = tdata[o00], c10r = tdata[o10], c01r = tdata[o01], c11r = tdata[o11];
            const row0r = c00r + (c10r - c00r) * tx;
            const row1r = c01r + (c11r - c01r) * tx;
            tr = row0r + (row1r - row0r) * ty;
            const c00g = tdata[o00 + 1], c10g = tdata[o10 + 1], c01g = tdata[o01 + 1], c11g = tdata[o11 + 1];
            const row0g = c00g + (c10g - c00g) * tx;
            const row1g = c01g + (c11g - c01g) * tx;
            tg = row0g + (row1g - row0g) * ty;
            const c00b = tdata[o00 + 2], c10b = tdata[o10 + 2], c01b = tdata[o01 + 2], c11b = tdata[o11 + 2];
            const row0b = c00b + (c10b - c00b) * tx;
            const row1b = c01b + (c11b - c01b) * tx;
            tb = row0b + (row1b - row0b) * ty;
          } else {
            let tx = Math.floor(uu * tw);
            let ty = Math.floor(vv * th);
            if (tx >= tw) tx = tw - 1;
            if (ty >= th) ty = th - 1;
            const o = (ty * tw + tx) * 4;
            tr = tdata[o]; tg = tdata[o + 1]; tb = tdata[o + 2];
          }
          ar = tr * INV255 * specAr;
          ag = tg * INV255 * specAg;
          ab = tb * INV255 * specAb;
        }
        // 内联 Blinn-Phong（与 shading.ts 冻结公式一致）
        let nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (nl === 0) nl = 1;
        nx /= nl; ny /= nl; nz /= nl;
        const ex = eye!.x - pxx, ey = eye!.y - pyy, ez = eye!.z - pzz;
        let vl = Math.sqrt(ex * ex + ey * ey + ez * ez);
        if (vl === 0) vl = 1;
        const vx = ex / vl, vy = ey / vl, vz = ez / vl;
        r = ar * lights!.ambient[0];
        g = ag * lights!.ambient[1];
        b = ab * lights!.ambient[2];
        const dir = lights!.dir;
        if (dir !== null) {
          const lx = dir.dir[0], ly = dir.dir[1], lz = dir.dir[2];
          const ndl = nx * lx + ny * ly + nz * lz;
          if (ndl > 0) {
            r += ar * dir.color[0] * ndl;
            g += ag * dir.color[1] * ndl;
            b += ab * dir.color[2] * ndl;
            const hx = lx + vx, hy = ly + vy, hz = lz + vz;
            const hl = Math.sqrt(hx * hx + hy * hy + hz * hz);
            if (hl > 0) {
              const ndh = (nx * hx + ny * hy + nz * hz) / hl;
              if (ndh > 0) {
                const s = spec.specular * powInt(ndh, spec.shininess);
                r += dir.color[0] * s;
                g += dir.color[1] * s;
                b += dir.color[2] * s;
              }
            }
          }
        }
        const pt = lights!.point;
        if (pt !== null) {
          const ldx = pt.pos.x - pxx, ldy = pt.pos.y - pyy, ldz = pt.pos.z - pzz;
          const dist = Math.sqrt(ldx * ldx + ldy * ldy + ldz * ldz);
          if (dist > 0) {
            const lx = ldx / dist, ly = ldy / dist, lz = ldz / dist;
            const atten = 1 / (1 + pt.k1 * dist + pt.k2 * dist * dist);
            const ndl = nx * lx + ny * ly + nz * lz;
            if (ndl > 0) {
              r += ar * pt.color[0] * ndl * atten;
              g += ag * pt.color[1] * ndl * atten;
              b += ab * pt.color[2] * ndl * atten;
              const hx = lx + vx, hy = ly + vy, hz = lz + vz;
              const hl = Math.sqrt(hx * hx + hy * hy + hz * hz);
              if (hl > 0) {
                const ndh = (nx * hx + ny * hy + nz * hz) / hl;
                if (ndh > 0) {
                  const s = spec.specular * powInt(ndh, spec.shininess);
                  r += pt.color[0] * s * atten;
                  g += pt.color[1] * s * atten;
                  b += pt.color[2] * s * atten;
                }
              }
            }
          }
        }
        // shadePhongPixel 语义：输出 0..255（此处补上量化前的放大）
        r *= 255; g *= 255; b *= 255;
      }

      const o = idx * 4;
      fbColor[o] = r;
      fbColor[o + 1] = g;
      fbColor[o + 2] = b;
      count++;
    }
  }
  return count;
}
