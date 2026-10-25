/**
 * 逐像素 diff 工具。黄金图对拍容差为 0：任何通道差异都算失败。
 */

import type { RawImage } from "./ppm.js";

export interface DiffResult {
  ok: boolean;
  message: string;
  numDiff: number;   // 有差异的像素数
  maxDiff: number;   // 最大通道差
  first: null | { x: number; y: number; a: [number, number, number]; b: [number, number, number] };
}

export function diffRGB(a: RawImage, b: RawImage): DiffResult {
  if (a.width !== b.width || a.height !== b.height) {
    return {
      ok: false,
      message: `dimensions differ: ${a.width}x${a.height} vs ${b.width}x${b.height}`,
      numDiff: -1, maxDiff: -1, first: null,
    };
  }
  let numDiff = 0;
  let maxDiff = 0;
  let first: DiffResult["first"] = null;
  const n = a.width * a.height;
  for (let i = 0; i < n; i++) {
    const dr = Math.abs(a.rgb[i * 3] - b.rgb[i * 3]);
    const dg = Math.abs(a.rgb[i * 3 + 1] - b.rgb[i * 3 + 1]);
    const db = Math.abs(a.rgb[i * 3 + 2] - b.rgb[i * 3 + 2]);
    const d = Math.max(dr, dg, db);
    if (d > 0) {
      numDiff++;
      if (d > maxDiff) maxDiff = d;
      if (first === null) {
        const x = i % a.width, y = (i / a.width) | 0;
        first = {
          x, y,
          a: [a.rgb[i * 3], a.rgb[i * 3 + 1], a.rgb[i * 3 + 2]],
          b: [b.rgb[i * 3], b.rgb[i * 3 + 1], b.rgb[i * 3 + 2]],
        };
      }
    }
  }
  return {
    ok: numDiff === 0,
    message: numDiff === 0 ? "identical" : `${numDiff} pixels differ, max channel delta ${maxDiff}`,
    numDiff, maxDiff, first,
  };
}
