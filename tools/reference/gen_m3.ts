/**
 * M3 参考实现：逐像素直接求值的三边形填充（独立于 src/raster.ts 的核心循环）。
 * 数学与正式实现同源（这是规格），但代码结构独立：
 * 不做回调/属性机制，直接逐像素算三条边函数与重心权重，写像素。
 * top-left 规则：E==0 仅当边为 top（水平 dx>0）或 left（dy<0）时包含 —— 由水密性扫描独立验证。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";

export function refFillTriangle(
  fb: Framebuffer,
  x0: number, y0: number, x1: number, y1: number, x2: number, y2: number,
  r: number, g: number, b: number,
): number {
  const area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
  if (!(Math.abs(area) > 1e-12)) return 0;
  let ax = x0, ay = y0, bx = x1, by = y1, cx = x2, cy = y2;
  if (area < 0) { bx = x2; by = y2; cx = x1; cy = y1; }

  const minX = Math.min(ax, bx, cx), maxX = Math.max(ax, bx, cx);
  const minY = Math.min(ay, by, cy), maxY = Math.max(ay, by, cy);
  if (![minX, maxX, minY, maxY].every(Number.isFinite)) return 0;
  const xs = Math.max(0, Math.ceil(minX - 0.5));
  const xe = Math.min(fb.width - 1, Math.floor(maxX - 0.5));
  const ys = Math.max(0, Math.ceil(minY - 0.5));
  const ye = Math.min(fb.height - 1, Math.floor(maxY - 0.5));

  let count = 0;
  for (let y = ys; y <= ye; y++) {
    for (let x = xs; x <= xe; x++) {
      const px = x + 0.5, py = y + 0.5;
      // λ0 ∝ 子三角形(b,c,p) 的有向面积，λ1 ∝ (c,a,p)，λ2 ∝ (a,b,p)
      const w0 = (cx - bx) * (py - by) - (cy - by) * (px - bx);
      const w1 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
      const w2 = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
      if (w0 < 0 || w1 < 0 || w2 < 0) continue;
      if (w0 === 0 || w1 === 0 || w2 === 0) {
        // 落边：top-left 规则
        const e0y = by - ay, e0x = bx - ax;
        const e1y = cy - by, e1x = cx - bx;
        const e2y = ay - cy, e2x = ax - cx;
        const isTopLeft = (dy: number, dx: number): boolean => dy < 0 || (dy === 0 && dx > 0);
        if ((w0 > 0 || isTopLeft(e1y, e1x)) &&
            (w1 > 0 || isTopLeft(e2y, e2x)) &&
            (w2 > 0 || isTopLeft(e0y, e0x))) {
          fb.setPixel(x, y, r, g, b);
          count++;
        }
        continue;
      }
      fb.setPixel(x, y, r, g, b);
      count++;
    }
  }
  return count;
}

export interface TriSpec { x0: number; y0: number; x1: number; y1: number; x2: number; y2: number; c: [number, number, number] }

/** 黄金场景（128²）：共享边特写（两条对角线方向）+ 扇形 + 细长条 + 越界 + 反向环绕。 */
export const M3_SCENE: TriSpec[] = [
  // 共享主对角线 (8,8)→(63,63)：对角线恰穿过像素中心，top-left 规则决定归属
  { x0: 8, y0: 8, x1: 63, y1: 8, x2: 63, y2: 63, c: [220, 60, 60] },
  { x0: 8, y0: 8, x1: 63, y1: 63, x2: 8, y2: 63, c: [60, 200, 60] },
  // 共享副对角线：归一化后由另一侧包含
  { x0: 65, y0: 8, x1: 120, y1: 8, x2: 65, y2: 63, c: [70, 90, 230] },
  { x0: 120, y0: 8, x1: 120, y1: 63, x2: 65, y2: 63, c: [230, 190, 60] },
  // 右侧穿出画布的三角形（guard）
  { x0: 110, y0: 70, x1: 140, y1: 95, x2: 95, y2: 95, c: [220, 70, 200] },
  // 细长条
  { x0: 8, y0: 100, x1: 60, y1: 100, x2: 8, y2: 103, c: [60, 210, 210] },
  // 反向环绕的大三角形（归一化路径）
  { x0: 64, y0: 70, x1: 100, y1: 122, x2: 40, y2: 122, c: [240, 150, 60] },
  // 扇形：围绕中心 8 个三角形，交替正反环绕
  ...Array.from({ length: 8 }, (_, k): TriSpec => {
    const ang0 = (k / 8) * Math.PI * 2;
    const ang1 = ((k + 1) / 8) * Math.PI * 2;
    const p0 = { x: 96, y: 104 };
    const p1 = { x: Math.round(96 + 22 * Math.cos(ang0)), y: Math.round(104 + 14 * Math.sin(ang0)) };
    const p2 = { x: Math.round(96 + 22 * Math.cos(ang1)), y: Math.round(104 + 14 * Math.sin(ang1)) };
    const c: [number, number, number] = k % 2 === 0 ? [170, 130, 240] : [110, 220, 150];
    return k % 2 === 0
      ? { x0: p0.x, y0: p0.y, x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y, c }
      : { x0: p0.x, y0: p0.y, x1: p2.x, y1: p2.y, x2: p1.x, y2: p1.y, c };
  }),
];

export function renderSceneWith(fill: typeof refFillTriangle): RawImage {
  const fb = new Framebuffer(128, 128);
  for (const t of M3_SCENE) fill(fb, t.x0, t.y0, t.x1, t.y1, t.x2, t.y2, t.c[0], t.c[1], t.c[2]);
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("m3_triangles.ppm", renderSceneWith(refFillTriangle));
  console.log("m3_triangles.ppm written");
}
