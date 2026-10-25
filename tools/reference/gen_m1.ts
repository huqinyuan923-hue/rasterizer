/**
 * M1 参考实现：黄金基线场景。
 * 内容：垂直渐变背景 + 种子 PRNG 撒点（mulberry32，种子 20261025）。
 * 本文件是"参考实现"：黄金图由此生成并冻结；正式实现以它为对拍基准。
 * 场景定义（渐变公式、点数、种子）是夹具，测试与生成器共用。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { mulberry32 } from "../../src/rng.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";

export const M1_W = 96;
export const M1_H = 96;
export const M1_SEED = 20261025;
export const M1_DOT_COUNT = 80;

/** 构建黄金基线场景（参考实现直接渲染）。 */
export function renderM1(): RawImage {
  const fb = new Framebuffer(M1_W, M1_H);
  // 背景：自顶向下垂直渐变（顶暗底亮）
  for (let y = 0; y < M1_H; y++) {
    const t = y / (M1_H - 1);
    const r = Math.round(18 + t * 40);
    const g = Math.round(24 + t * 90);
    const b = Math.round(48 + t * 150);
    for (let x = 0; x < M1_W; x++) fb.setPixel(x, y, r, g, b);
  }
  // 种子撒点：2×2 色块，颜色由 PRNG 决定
  const rnd = mulberry32(M1_SEED);
  for (let k = 0; k < M1_DOT_COUNT; k++) {
    const x = Math.floor(rnd() * (M1_W - 2));
    const y = Math.floor(rnd() * (M1_H - 2));
    const r = Math.floor(rnd() * 256);
    const g = Math.floor(rnd() * 256);
    const b = Math.floor(rnd() * 256);
    for (let dy = 0; dy < 2; dy++) {
      for (let dx = 0; dx < 2; dx++) fb.setPixel(x + dx, y + dy, r, g, b);
    }
  }
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("m1_baseline.ppm", renderM1());
  console.log("m1_baseline.ppm written");
}
