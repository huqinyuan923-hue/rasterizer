import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Framebuffer } from "../src/framebuffer.js";
import { fillTriangleFlat } from "../src/raster.js";
import { decodePPM, framebufferToRaw, type RawImage } from "../src/ppm.js";
import { diffRGB } from "../src/diff.js";
import { mulberry32 } from "../src/rng.js";
import { renderSceneWith } from "../tools/reference/gen_m3.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "..", "..", "golden");

function loadGolden(name: string): RawImage {
  return decodePPM(new Uint8Array(readFileSync(join(GOLDEN_DIR, name))));
}

// ---------- 黄金对拍 ----------

test("M3 黄金: 正式 edge-function 填充与参考实现黄金图逐像素一致", () => {
  const golden = loadGolden("m3_triangles.ppm");
  const live = renderSceneWith((fb, x0, y0, x1, y1, x2, y2, r, g, b) =>
    fillTriangleFlat(fb, x0, y0, x1, y1, x2, y2, r, g, b));
  const r = diffRGB(golden, live);
  assert.ok(r.ok, `golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

// ---------- 水密性扫描（无参考实现、独立的不变量） ----------

/**
 * 两个相邻三角形拼成矩形时：矩形内每个像素中心恰好被覆盖一次（无缝隙、无重叠），
 * 矩形外零覆盖。top-left 规则的"总和且互斥"性质由此独立验证。
 */
test("M3 水密性: 600 对相邻三角形拼矩形，逐像素覆盖计数恰为 1", () => {
  const W = 128, H = 128;
  const fbA = new Framebuffer(W, H);
  const fbB = new Framebuffer(W, H);
  const rnd = mulberry32(777);
  for (let iter = 0; iter < 600; iter++) {
    const w = 1 + Math.floor(rnd() * 40);
    const h = 1 + Math.floor(rnd() * 40);
    const rx = Math.floor(rnd() * (W - w));
    const ry = Math.floor(rnd() * (H - h));
    const x0 = rx, y0 = ry, x1 = rx + w, y1 = ry + h;
    fbA.clear(0, 0, 0);
    fbB.clear(0, 0, 0);
    let a: [number, number, number, number, number, number][];
    let b: [number, number, number, number, number, number][];
    if (iter % 2 === 0) {
      // 主对角线拆分
      a = [[x0, y0, x1, y0, x1, y1]];
      b = [[x0, y0, x1, y1, x0, y1]];
    } else {
      // 副对角线拆分
      a = [[x0, y0, x1, y0, x0, y1]];
      b = [[x1, y0, x1, y1, x0, y1]];
    }
    for (const t of a) fillTriangleFlat(fbA, t[0], t[1], t[2], t[3], t[4], t[5], 255, 255, 255);
    for (const t of b) fillTriangleFlat(fbB, t[0], t[1], t[2], t[3], t[4], t[5], 255, 255, 255);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const o = (y * W + x) * 4;
        const inA = fbA.color[o] === 255;
        const inB = fbB.color[o] === 255;
        assert.ok(!(inA && inB), `overlap at (${x},${y}) rect=(${rx},${ry},${w},${h}) diag=${iter % 2}`);
        const inRect = x >= rx && x < x1 && y >= ry && y < y1;
        if (inRect) {
          assert.ok(inA || inB, `gap at (${x},${y}) rect=(${rx},${ry},${w},${h}) diag=${iter % 2}`);
        } else {
          assert.ok(!inA && !inB, `leak outside rect at (${x},${y})`);
        }
      }
    }
  }
});

// ---------- 覆盖计数（解析已知答案） ----------

test("M3 覆盖计数: 直角三角形 (0,0)(8,0)(0,8) 恰覆盖 28 个像素中心", () => {
  // 斜边 x+y=8 穿过 i+j=7 的像素中心；top-left 规则下斜边像素归属他者 ⟹ 覆盖 i+j ≤ 6 ⟹ 28
  const fb = new Framebuffer(16, 16);
  const n = fillTriangleFlat(fb, 0, 0, 8, 0, 0, 8, 255, 0, 0);
  assert.equal(n, 28);
});

// ---------- 环绕归一化 ----------

test("M3 环绕: 同一三角形 CW/CCW 两种顶点序产出相同图像", () => {
  const fbA = new Framebuffer(64, 64);
  const fbB = new Framebuffer(64, 64);
  fillTriangleFlat(fbA, 10, 10, 50, 18, 22, 50, 200, 100, 50);
  fillTriangleFlat(fbB, 10, 10, 22, 50, 50, 18, 200, 100, 50);
  const d = diffRGB(framebufferToRaw(fbA), framebufferToRaw(fbB));
  assert.ok(d.ok, `winding normalization failed: ${d.message}`);
});

// ---------- 退化与 guard ----------

test("M3 退化: 零面积三角形返回 0 且不崩溃", () => {
  const fb = new Framebuffer(8, 8);
  assert.equal(fillTriangleFlat(fb, 3, 3, 3, 3, 3, 3, 255, 0, 0), 0);
  assert.equal(fillTriangleFlat(fb, 1, 1, 5, 1, 9, 1, 255, 0, 0), 0); // 共线
});

test("M3 guard: 完全画布外的三角形返回 0；部分出界只画界内", () => {
  const fb = new Framebuffer(32, 32);
  assert.equal(fillTriangleFlat(fb, 100, 100, 140, 100, 120, 140, 255, 0, 0), 0);
  // 半出界：底边 y=30 在界内，顶点 y=50 出界
  const n = fillTriangleFlat(fb, 0, 30, 31, 30, 15, 50, 255, 0, 0);
  assert.ok(n > 0 && n < 32 * 2, `partial triangle should draw some pixels, got ${n}`);
});
