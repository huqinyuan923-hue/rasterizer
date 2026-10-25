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

/** 屏幕空间顶点：位置 + 透视校正插值所需的预乘量。 */
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
  let count = 0;
  for (let y = ys; y <= ye; y++) {
    const py = y + 0.5;
    for (let x = xs; x <= xe; x++) {
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
      const idx = y * fb.width + x;
      if (needDenom) {
        const denom = lam0 * va.invW + lam1 * vb.invW + lam2 * vc.invW;
        if (opts.depthTest) {
          const d = (lam0 * va.dOverW + lam1 * vb.dOverW + lam2 * vc.dOverW) / denom;
          if (!(d >= opts.nearDist) || !(d < fb.depth[idx])) continue;
          for (let j = 0; j < attrCount; j++) {
            SCRATCH_ATTRS[j] = (lam0 * va.aOverW[j] + lam1 * vb.aOverW[j] + lam2 * vc.aOverW[j]) / denom;
          }
          shade(lam0, lam1, lam2, denom, d, SCRATCH_ATTRS, SCRATCH_RGB);
          fb.depth[idx] = d;
        } else {
          for (let j = 0; j < attrCount; j++) {
            SCRATCH_ATTRS[j] = (lam0 * va.aOverW[j] + lam1 * vb.aOverW[j] + lam2 * vc.aOverW[j]) / denom;
          }
          shade(lam0, lam1, lam2, denom, 0, SCRATCH_ATTRS, SCRATCH_RGB);
        }
      } else {
        shade(lam0, lam1, lam2, 1, 0, SCRATCH_ATTRS, SCRATCH_RGB);
      }
      const o = idx * 4;
      fb.color[o] = SCRATCH_RGB[0];
      fb.color[o + 1] = SCRATCH_RGB[1];
      fb.color[o + 2] = SCRATCH_RGB[2];
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
