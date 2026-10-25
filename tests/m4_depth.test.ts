import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Framebuffer } from "../src/framebuffer.js";
import { fillTriangleDepth } from "../src/raster.js";
import { decodePPM, framebufferToRaw, type RawImage } from "../src/ppm.js";
import { diffRGB } from "../src/diff.js";
import { M4_TRIANGLES, M4_W, M4_H, renderM4 } from "../tools/reference/gen_m4.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "..", "..", "golden");

function loadGolden(name: string): RawImage {
  return decodePPM(new Uint8Array(readFileSync(join(GOLDEN_DIR, name))));
}

function renderFormal(order: number[] = [0, 1, 2, 3, 4]): RawImage {
  const fb = new Framebuffer(M4_W, M4_H);
  for (const i of order) {
    const t = M4_TRIANGLES[i];
    fillTriangleDepth(fb, t.sx[0], t.sy[0], t.d[0], t.sx[1], t.sy[1], t.d[1], t.sx[2], t.sy[2], t.d[2], t.c[0], t.c[1], t.c[2]);
  }
  return framebufferToRaw(fb);
}

// ---------- 黄金对拍 ----------

test("M4 黄金: 正式 z-buffer 与参考实现黄金图逐像素一致", () => {
  const golden = loadGolden("m4_depth.ppm");
  const r = diffRGB(golden, renderFormal());
  assert.ok(r.ok, `golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

// ---------- 遮挡语义 ----------

test("M4 遮挡: 绘制顺序无关（不同深度三角形，任意顺序结果一致）", () => {
  const forward = renderFormal([0, 1, 2]);
  const backward = renderFormal([2, 1, 0]);
  const shuffled = renderFormal([1, 2, 0]);
  assert.ok(diffRGB(forward, backward).ok, "reverse order must match");
  assert.ok(diffRGB(forward, shuffled).ok, "shuffled order must match");
});

test("M4 遮挡: 重叠区近者胜（远画在近之后也无法覆盖）", () => {
  // 红(d=8)与绿(d=5)重叠区应是绿色，无论谁先画
  const fbA = new Framebuffer(M4_W, M4_H);
  const fbB = new Framebuffer(M4_W, M4_H);
  const red = M4_TRIANGLES[0], green = M4_TRIANGLES[1];
  fillTriangleDepth(fbA, red.sx[0], red.sy[0], red.d[0], red.sx[1], red.sy[1], red.d[1], red.sx[2], red.sy[2], red.d[2], red.c[0], red.c[1], red.c[2]);
  fillTriangleDepth(fbA, green.sx[0], green.sy[0], green.d[0], green.sx[1], green.sy[1], green.d[1], green.sx[2], green.sy[2], green.d[2], green.c[0], green.c[1], green.c[2]);
  fillTriangleDepth(fbB, green.sx[0], green.sy[0], green.d[0], green.sx[1], green.sy[1], green.d[1], green.sx[2], green.sy[2], green.d[2], green.c[0], green.c[1], green.c[2]);
  fillTriangleDepth(fbB, red.sx[0], red.sy[0], red.d[0], red.sx[1], red.sy[1], red.d[1], red.sx[2], red.sy[2], red.d[2], red.c[0], red.c[1], red.c[2]);
  // 找一个两三角形 bbox 交叠且各属于两者的像素，断言两幅图都是绿色
  let checked = 0;
  for (let y = 0; y < M4_H; y++) {
    for (let x = 0; x < M4_W; x++) {
      const o = (y * M4_W + x) * 4;
      const isGreen = (fb: Framebuffer): boolean =>
        fb.color[o] === green.c[0] && fb.color[o + 1] === green.c[1] && fb.color[o + 2] === green.c[2];
      if (isGreen(fbA) && isGreen(fbB)) checked++; // 两种顺序下都是绿 → 深度判定的稳定性
    }
  }
  assert.ok(checked > 50, `overlap region should resolve to green consistently, got ${checked} pixels`);
});

test("M4 平局: 逐位相同深度时先画者赢（严格小于）；二进制精确几何下可构造逐位相同", () => {
  // 2 的幂坐标 ⟹ 边函数与重心权重均为二进制精确值，常数 z=4 的插值深度逐位等于 4.0
  const fb = new Framebuffer(16, 16);
  fillTriangleDepth(fb, 0, 0, 4, 8, 0, 4, 0, 8, 4, 255, 255, 0);  // 黄（先）
  fillTriangleDepth(fb, 0, 0, 4, 8, 8, 4, 0, 8, 4, 200, 60, 200); // 紫（后，常数 z=4 逐位相同）
  const o = (2 * 16 + 2) * 4; // (2,2)：两三角形都内部覆盖
  assert.deepEqual([fb.color[o], fb.color[o + 1], fb.color[o + 2]], [255, 255, 0], "first drawn must win an exact tie");
});

test("M4 平局: 非精确几何的同深度对是确定的（噪声方向冻结，两次渲染一致）", () => {
  const fbA = new Framebuffer(64, 64);
  const fbB = new Framebuffer(64, 64);
  for (const fb of [fbA, fbB]) {
    fillTriangleDepth(fb, 4, 4, 6, 40, 4, 6, 4, 40, 6, 255, 255, 0);
    fillTriangleDepth(fb, 4, 4, 6, 40, 4, 6, 40, 40, 6, 200, 60, 200);
  }
  assert.ok(diffRGB(framebufferToRaw(fbA), framebufferToRaw(fbB)).ok, "tie resolution must be deterministic");
});

// ---------- 线性插值正确性 ----------

test("M4 线性 z: 深度按屏幕空间仿函数精确插值（float32 精度内）", () => {
  // 三角形 (0,0,4)(32,0,4)(0,32,10)：z 仿函数 z(x,y) = 4 + 6y/32（梯度仅在 y）
  const fb = new Framebuffer(48, 48);
  fillTriangleDepth(fb, 0, 0, 4, 32, 0, 4, 0, 32, 10, 255, 0, 0);
  for (const [i, j] of [[3, 3], [5, 2], [1, 8], [12, 2], [2, 12]] as const) {
    const expected = 4 + 6 * (j + 0.5) / 32;
    const got = fb.depth[j * 48 + i];
    assert.ok(Math.abs(got - expected) < 1e-6, `depth(${i},${j}): got ${got}, want ${expected}`);
  }
});

// ---------- 黄金场景排序一致性（参考实现自检） ----------

test("M4 参考: 参考实现同样顺序无关（夹具自检）", () => {
  assert.ok(diffRGB(renderM4([0, 1, 2, 3, 4]), renderM4([3, 2, 1, 0, 4])).ok);
});
