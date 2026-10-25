/**
 * M2 参考实现：经典"八分体归约"Bresenham（教科书版）。
 * 与正式实现（全象限误差累积变体）是**不同写法的两个正确实现**：
 * 在无平局的直线上二者必须产出完全相同的像素集合 —— 由测试的随机等价扫描保证。
 * 平局像素的选择是约定而非正确性，由钉死用例单独冻结正式实现的行为。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";

/** 判断一条整数端点直线是否"无平局"（中点永远不会恰好落在理想线上）。 */
export function isTieFree(x0: number, y0: number, x1: number, y1: number): boolean {
  const dx = Math.abs(x1 - x0), dy = Math.abs(y1 - y0);
  if (dx === 0 || dy === 0) return true;   // 轴对齐：无决策步进
  if (dx === dy) return true;              // 精确 45°：中点恒偏离 0.5，无平局
  return (dy > dx ? dy : dx) % 2 === 1;    // 步进轴的跨度为奇数 ⟹ 无平局
}

/** 参考直线：经典八分体归约 + 误差项 D（D > 0 才步进副轴）。 */
export function refDrawLine(
  fb: Framebuffer, x0: number, y0: number, x1: number, y1: number,
  r: number, g: number, b: number,
): void {
  const steep = Math.abs(y1 - y0) > Math.abs(x1 - x0);
  let ax0 = x0, ay0 = y0, ax1 = x1, ay1 = y1;
  if (steep) { ax0 = y0; ay0 = x0; ax1 = y1; ay1 = x1; }  // 换到主轴域
  if (ax0 > ax1) { // 保证主轴递增
    let t = ax0; ax0 = ax1; ax1 = t;
    t = ay0; ay0 = ay1; ay1 = t;
  }
  const dx = ax1 - ax0;
  const dy = Math.abs(ay1 - ay0);
  const ystep = ay0 < ay1 ? 1 : -1;
  let y = ay0;
  let D = 2 * dy - dx;
  const put = (px: number, py: number): void => {
    if (steep) fb.setPixel(py, px, r, g, b);
    else fb.setPixel(px, py, r, g, b);
  };
  for (let x = ax0; x <= ax1; x++) {
    put(x, y);
    if (D > 0) { y += ystep; D -= 2 * dx; }
    D += 2 * dy;
  }
}

export interface LineSpec { x0: number; y0: number; x1: number; y1: number; c: [number, number, number] }

/** 黄金场景：全八分族 + 轴对齐 + 45° + 越界 guard + 退化点，全部无平局。 */
export const M2_SCENE: LineSpec[] = [
  { x0: 64, y0: 64, x1: 121, y1: 75, c: [255, 80, 80] },    // 浅正
  { x0: 64, y0: 64, x1: 23, y1: 90, c: [80, 255, 80] },     // 浅负
  { x0: 64, y0: 64, x1: 64, y1: 10, c: [80, 80, 255] },     // 垂直向上
  { x0: 64, y0: 64, x1: 64, y1: 127, c: [255, 255, 80] },   // 垂直向下
  { x0: 64, y0: 64, x1: 10, y1: 64, c: [80, 255, 255] },    // 水平向左
  { x0: 64, y0: 64, x1: 127, y1: 64, c: [255, 80, 255] },   // 水平向右
  { x0: 64, y0: 64, x1: 100, y1: 100, c: [255, 160, 60] },  // 45° 正
  { x0: 64, y0: 64, x1: 8, y1: 8, c: [60, 200, 255] },      // 45° 负向
  { x0: 64, y0: 64, x1: 100, y1: 28, c: [200, 120, 255] },  // -45°
  { x0: 64, y0: 64, x1: 15, y1: 108, c: [120, 255, 160] },  // 浅负大跨
  { x0: 64, y0: 64, x1: 111, y1: 15, c: [255, 200, 120] },  // 陡正
  { x0: 64, y0: 64, x1: 17, y1: 13, c: [160, 255, 220] },   // 陡负
  { x0: 0, y0: 0, x1: 141, y1: 63, c: [255, 100, 100] },    // 越界（右）
  { x0: -31, y0: 101, x1: 150, y1: 40, c: [100, 255, 100] },// 双侧越界
  { x0: 64, y0: 64, x1: 64, y1: 64, c: [255, 255, 255] },   // 退化点
  { x0: 64, y0: 64, x1: 71, y1: 127, c: [130, 180, 255] },  // 越界（下）陡
  { x0: 64, y0: 64, x1: 3, y1: 120, c: [255, 130, 200] },   // 越界（左下）
  { x0: 10, y0: 10, x1: 120, y1: 10, c: [220, 220, 90] },   // 顶部水平
  { x0: 120, y0: 120, x1: 10, y1: 120, c: [90, 220, 220] }, // 底部水平（反向）
  { x0: 64, y0: 127, x1: 64, y1: 4, c: [220, 150, 90] },    // 垂直反向
  { x0: 64, y0: 64, x1: 40, y1: 127, c: [150, 150, 255] },  // 陡正越界
];

export function renderSceneWith(draw: typeof refDrawLine): RawImage {
  const fb = new Framebuffer(128, 128);
  for (const l of M2_SCENE) draw(fb, l.x0, l.y0, l.x1, l.y1, l.c[0], l.c[1], l.c[2]);
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("m2_lines.ppm", renderSceneWith(refDrawLine));
  console.log("m2_lines.ppm written");
}
