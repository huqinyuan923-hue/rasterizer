import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Framebuffer } from "../src/framebuffer.js";
import { drawLine, type PixelSink } from "../src/raster.js";
import { decodePPM, framebufferToRaw, type RawImage } from "../src/ppm.js";
import { diffRGB } from "../src/diff.js";
import { mulberry32 } from "../src/rng.js";
import { refDrawLine, isTieFree, renderSceneWith, M2_SCENE } from "../tools/reference/gen_m2.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "..", "..", "golden");

function loadGolden(name: string): RawImage {
  return decodePPM(new Uint8Array(readFileSync(join(GOLDEN_DIR, name))));
}

/** 按写入顺序记录像素的记录器。 */
class Recorder implements PixelSink {
  pixels: Array<[number, number]> = [];
  setPixel(x: number, y: number): void {
    this.pixels.push([x, y]);
  }
}

function distPointSegment(px: number, py: number, x0: number, y0: number, x1: number, y1: number): number {
  const dx = x1 - x0, dy = y1 - y0;
  const l2 = dx * dx + dy * dy;
  if (l2 === 0) return Math.hypot(px - x0, py - y0);
  let t = ((px - x0) * dx + (py - y0) * dy) / l2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (x0 + t * dx), py - (y0 + t * dy));
}

// ---------- 黄金对拍 ----------

test("M2 黄金: 正式 Bresenham 与参考实现生成的黄金图逐像素一致", () => {
  const golden = loadGolden("m2_lines.ppm");
  const live = renderSceneWith((fb, x0, y0, x1, y1, r, g, b) => drawLine(fb, x0, y0, x1, y1, r, g, b));
  const r = diffRGB(golden, live);
  assert.ok(r.ok, `golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

// ---------- 双实现等价扫描（无平局直线上必须逐像素相同） ----------

test("M2 等价扫描: 800 条种子随机无平局直线上，经典八分体版与全象限版像素集合完全一致", () => {
  const W = 128, H = 128;
  const fbA = new Framebuffer(W, H);
  const fbB = new Framebuffer(W, H);
  const rnd = mulberry32(2026);
  let tested = 0;
  for (let i = 0; i < 800; i++) {
    const x0 = Math.floor(rnd() * 152) - 12;
    const y0 = Math.floor(rnd() * 152) - 12;
    const x1 = Math.floor(rnd() * 152) - 12;
    const y1 = Math.floor(rnd() * 152) - 12;
    if (!isTieFree(x0, y0, x1, y1)) continue;
    tested++;
    fbA.clear(0, 0, 0);
    fbB.clear(0, 0, 0);
    refDrawLine(fbA, x0, y0, x1, y1, 255, 255, 255);
    drawLine(fbB, x0, y0, x1, y1, 255, 255, 255);
    const d = diffRGB(framebufferToRaw(fbA), framebufferToRaw(fbB));
    assert.ok(d.ok, `line (${x0},${y0})->(${x1},${y1}) differs: ${d.message} first=${JSON.stringify(d.first)}`);
  }
  assert.ok(tested > 300, `sweep should exercise enough tie-free lines, got ${tested}`);
});

// ---------- 性质测试 ----------

test("M2 性质: 端点必画、按序 8-连通、每点到线段距离 ≤ 0.51", () => {
  for (const l of M2_SCENE) {
    const rec = new Recorder();
    drawLine(rec, l.x0, l.y0, l.x1, l.y1, 255, 255, 255);
    const ps = rec.pixels;
    assert.ok(ps.length >= 1, "at least one pixel");
    assert.deepEqual(ps[0], [l.x0, l.y0], "start endpoint drawn first");
    assert.deepEqual(ps[ps.length - 1], [l.x1, l.y1], "end endpoint drawn last");
    for (let i = 1; i < ps.length; i++) {
      const ddx = Math.abs(ps[i][0] - ps[i - 1][0]);
      const ddy = Math.abs(ps[i][1] - ps[i - 1][1]);
      assert.ok(ddx <= 1 && ddy <= 1 && ddx + ddy >= 1, `not 8-connected at index ${i} of (${l.x0},${l.y0})->(${l.x1},${l.y1})`);
    }
    for (const [px, py] of ps) {
      const d = distPointSegment(px, py, l.x0, l.y0, l.x1, l.y1);
      assert.ok(d <= 0.51, `pixel (${px},${py}) too far (${d}) from segment (${l.x0},${l.y0})->(${l.x1},${l.y1})`);
    }
  }
});

// ---------- 平局约定钉死（全象限变体：平局走对角步进） ----------

test("M2 平局约定: 正向与反向的平局像素选择被冻结", () => {
  const rec1 = new Recorder();
  drawLine(rec1, 0, 0, 4, 2, 0, 0, 0);
  assert.deepEqual(rec1.pixels, [[0, 0], [1, 1], [2, 1], [3, 2], [4, 2]]);

  const rec2 = new Recorder();
  drawLine(rec2, 0, 0, 2, 1, 0, 0, 0);
  assert.deepEqual(rec2.pixels, [[0, 0], [1, 1], [2, 1]]);

  const rec3 = new Recorder();
  drawLine(rec3, 4, 2, 0, 0, 0, 0, 0);
  assert.deepEqual(rec3.pixels, [[4, 2], [3, 1], [2, 1], [1, 0], [0, 0]]);
});

// ---------- guard 与退化 ----------

test("M2 guard: 完全画布外的直线不崩不画；画布外端点被截断", () => {
  const fb = new Framebuffer(32, 32);
  drawLine(fb, 100, 100, 200, 150, 255, 255, 255); // 完全在外
  const img = framebufferToRaw(fb);
  const d = diffRGB(img, { width: 32, height: 32, rgb: new Uint8Array(32 * 32 * 3) });
  assert.ok(d.ok, "off-canvas line must draw nothing");

  drawLine(fb, -10, 5, 40, 5, 255, 255, 255); // 横穿，两端越界
  const img2 = framebufferToRaw(fb);
  let count = 0;
  for (let i = 0; i < 32 * 32; i++) if (img2.rgb[i * 3] === 255) count++;
  assert.equal(count, 32, "row 5 must be fully covered");
});

test("M2 退化: 点线只画一个像素", () => {
  const rec = new Recorder();
  drawLine(rec, 7, 9, 7, 9, 255, 255, 255);
  assert.deepEqual(rec.pixels, [[7, 9]]);
});
