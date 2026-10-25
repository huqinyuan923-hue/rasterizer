/**
 * M4 参考实现：z-buffer（屏幕空间线性 z 插值）。
 * 场景用正交投影 —— 线性 z 在正交下是精确插值，参考与正式实现的语义完全对齐。
 * 参考实现自带独立编码的逐像素重心插值与深度测试。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { mat4Ortho, mat4ApplyVec4 } from "../../src/math.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";

export const M4_W = 96;
export const M4_H = 96;

/** 正交相机：原点看 -Z，范围 x/y ∈ [-2.4, 2.4]，near=1 far=9。 */
const ORTHO = mat4Ortho(-2.4, 2.4, -2.4, 2.4, 1, 9);

export interface DepthTri {
  /** 已投影的屏幕坐标 + 视距 d = -z_view（正值，越小越近）。 */
  sx: [number, number, number];
  sy: [number, number, number];
  d: [number, number, number];
  c: [number, number, number];
}

function project(p: [number, number, number]): { x: number; y: number; d: number } {
  const c = mat4ApplyVec4(ORTHO, { x: p[0], y: p[1], z: p[2], w: 1 });
  return {
    x: (c.x / c.w + 1) * 0.5 * M4_W,
    y: (1 - c.y / c.w) * 0.5 * M4_H,
    d: -p[2],
  };
}

function tri(p0: [number, number, number], p1: [number, number, number], p2: [number, number, number], c: [number, number, number]): DepthTri {
  const a = project(p0), b = project(p1), q = project(p2);
  return { sx: [a.x, b.x, q.x], sy: [a.y, b.y, q.y], d: [a.d, b.d, q.d], c };
}

/** 黄金场景：远/中/近三块遮挡 + 一对同深度平局（先画者赢）。 */
export const M4_TRIANGLES: DepthTri[] = [
  tri([-1.8, 1.6, -8], [1.4, 1.7, -8], [-0.2, 0.1, -8], [210, 70, 70]),   // 远·红
  tri([-1.6, -0.2, -5], [1.9, -0.4, -5], [0.1, -1.9, -5], [70, 200, 90]), // 中·绿
  tri([1.9, 1.9, -2], [-0.4, 1.2, -2], [1.2, -0.6, -2], [80, 110, 230]),  // 近·蓝
  tri([-1.9, -1.9, -6], [-0.3, -1.9, -6], [-1.2, -0.7, -6], [230, 220, 80]),  // 平局·黄（先画）
  tri([-0.5, -1.9, -6], [0.9, -1.2, -6], [-1.7, -0.9, -6], [190, 90, 220]),   // 平局·紫（后画，应被拒）
];

/** 参考深度填充：逐像素直接求值重心权重 + 线性 z + 深度测试。 */
export function refFillTriangleDepth(
  fb: Framebuffer,
  t: DepthTri,
): void {
  const x0 = t.sx[0], y0 = t.sy[0], z0 = t.d[0];
  const x1 = t.sx[1], y1 = t.sy[1], z1 = t.d[1];
  const x2 = t.sx[2], y2 = t.sy[2], z2 = t.d[2];
  const area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
  if (!(Math.abs(area) > 1e-12)) return;
  let ax = x0, ay = y0, az = z0, bx = x1, by = y1, bz = z1, cx = x2, cy = y2, cz = z2;
  if (area < 0) { bx = x2; by = y2; bz = z2; cx = x1; cy = y1; cz = z1; }
  const A = area < 0 ? -area : area;

  const xs = Math.max(0, Math.ceil(Math.min(ax, bx, cx) - 0.5));
  const xe = Math.min(fb.width - 1, Math.floor(Math.max(ax, bx, cx) - 0.5));
  const ys = Math.max(0, Math.ceil(Math.min(ay, by, cy) - 0.5));
  const ye = Math.min(fb.height - 1, Math.floor(Math.max(ay, by, cy) - 0.5));

  for (let y = ys; y <= ye; y++) {
    for (let x = xs; x <= xe; x++) {
      const px = x + 0.5, py = y + 0.5;
      const w0 = (cx - bx) * (py - by) - (cy - by) * (px - bx);
      const w1 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
      const w2 = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
      if (w0 < 0 || w1 < 0 || w2 < 0) continue;
      if (w0 === 0 || w1 === 0 || w2 === 0) {
        const tl = (dy: number, dx: number): boolean => dy < 0 || (dy === 0 && dx > 0);
        const ok = (w0 > 0 || tl(cy - by, cx - bx)) &&
                   (w1 > 0 || tl(ay - cy, ax - cx)) &&
                   (w2 > 0 || tl(by - ay, bx - ax));
        if (!ok) continue;
      }
      const l0 = w0 / A, l1 = w1 / A, l2 = w2 / A;
      const z = l0 * az + l1 * bz + l2 * cz; // invW ≡ 1：屏幕空间线性 z
      const idx = y * fb.width + x;
      if (z < fb.depth[idx]) {
        fb.depth[idx] = z;
        const o = idx * 4;
        fb.color[o] = t.c[0];
        fb.color[o + 1] = t.c[1];
        fb.color[o + 2] = t.c[2];
      }
    }
  }
}

export function renderM4(order: number[] = [0, 1, 2, 3, 4]): RawImage {
  const fb = new Framebuffer(M4_W, M4_H);
  for (const i of order) refFillTriangleDepth(fb, M4_TRIANGLES[i]);
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("m4_depth.ppm", renderM4());
  console.log("m4_depth.ppm written");
}
