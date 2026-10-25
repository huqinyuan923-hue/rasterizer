/**
 * 光栅化核心。约定：直线端点是整数格点（像素索引本身），线段连接 (x0,y0) 与 (x1,y1)。
 * 平局约定（平局 = 中点恰好落在理想线上）：全象限误差累积变体，平局走对角步进 ——
 * 该约定已被 tests/m2_line.test.ts 中的"平局钉死用例"冻结，不要随意更改。
 */

/** 最小像素写入接口（Framebuffer 满足它；测试可用记录器）。 */
export interface PixelSink {
  setPixel(x: number, y: number, r: number, g: number, b: number): void;
}

/**
 * Bresenham 直线，全象限整数实现，含边界 guard（越界像素由 setPixel 丢弃）。
 */
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
