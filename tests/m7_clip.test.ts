import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Framebuffer } from "../src/framebuffer.js";
import { Renderer, CLIP_EPS, type DrawItem, type Mesh } from "../src/pipeline.js";
import { clipTriangleNear, type ClipVert } from "../src/clip.js";
import { decodePPM, framebufferToRaw, type RawImage } from "../src/ppm.js";
import { diffRGB } from "../src/diff.js";
import { M7_CAMERA, M7_LIGHTS, M7_TRIANGLES, renderM7 } from "../tools/reference/gen_m7.js";
import { mat4Identity } from "../src/math.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "..", "..", "golden");

function loadGolden(name: string): RawImage {
  return decodePPM(new Uint8Array(readFileSync(join(GOLDEN_DIR, name))));
}

// ---------- clipTriangleNear 单元 ----------

function cv(x: number, y: number, z: number, w: number, payload: number[] = []): ClipVert {
  return { x, y, z, w, d: -z, a: Float64Array.from(payload) };
}

test("M7 裁剪: 全部在前 → 原样返回 3 顶点（1 个三角形，顶点序不变）", () => {
  const a = cv(0, 0, -1, 1), b = cv(1, 0, -1, 1), c = cv(0, 1, -1, 1);
  const out = clipTriangleNear(a, b, c, 1e-6);
  assert.equal(out.length, 3);
  assert.equal(out[0], a);
  assert.equal(out[1], b);
  assert.equal(out[2], c);
});

test("M7 裁剪: 全部在后 → 0 顶点", () => {
  const out = clipTriangleNear(cv(0, 0, 1, -1), cv(1, 0, 1, -1), cv(0, 1, 2, -1), 1e-6);
  assert.equal(out.length, 0);
});

test("M7 裁剪: 单顶点在后 → 4 顶点（扇出 2 个三角形）", () => {
  // a 在后（w<eps），b/c 在前
  const a = cv(0, 0, 0.5, -0.5, [1, 2, 3]);
  const b = cv(1, 0, -1, 1, [4, 5, 6]);
  const c = cv(0, 1, -1, 1, [7, 8, 9]);
  const out = clipTriangleNear(a, b, c, 1e-6);
  assert.equal(out.length, 4, `expected quad, got ${out.length}`);
  // 两个新顶点都恰好在近平面上
  for (const v of out) {
    if (v !== a && v !== b && v !== c) {
      assert.ok(Math.abs(v.w - CLIP_EPS) < 1e-12, `clipped vertex w=${v.w} should equal eps`);
    }
  }
});

test("M7 裁剪: 双顶点在后 → 3 顶点（1 个三角形）", () => {
  const a = cv(0, 0, 0.5, -0.5);
  const b = cv(1, 0, 0.5, -0.5);
  const c = cv(0, 1, -1, 1);
  const out = clipTriangleNear(a, b, c, 1e-6);
  assert.equal(out.length, 3);
});

test("M7 裁剪: 交点插值精确 —— w 落在 eps，payload 线性", () => {
  // a: w=0.5（前），b: w=-0.3（后）→ t = (0.5-eps)/(0.5-(-0.3)) ≈ 0.62499875
  const a = cv(0, 0, -0.5, 0.5, [10]);
  const b = cv(2, 4, 0.3, -0.3, [20]);
  const out = clipTriangleNear(a, b, cv(0, 1, -1, 1), 1e-6);
  const v = out.find((x) => x !== a && x !== cv(0, 1, -1, 1))!;
  assert.ok(v, "crossing vertex exists");
  const t = (0.5 - 1e-6) / (0.5 - -0.3);
  assert.ok(Math.abs(v.x - 2 * t) < 1e-15, `x=${v.x}, want ${2 * t}`);
  assert.ok(Math.abs(v.a[0] - (10 + (20 - 10) * t)) < 1e-12, `payload=${v.a[0]}`);
  assert.ok(Math.abs(v.w - 1e-6) < 1e-15, `w=${v.w}`);
});

// ---------- 黄金对拍 ----------

function triMesh(p: [number, number, number][]): Mesh {
  return {
    positions: p.flat(),
    normals: p.map(() => 0).flatMap(() => [0, 0, 1]),
    uvs: p.map(() => 0).flatMap(() => [0, 0]),
    indices: [0, 1, 2],
  };
}

test("M7 黄金: 正式管线（真裁剪）与参考实现逐像素一致", () => {
  const golden = loadGolden("m7_clip.ppm");
  const items: DrawItem[] = M7_TRIANGLES.map((t) => ({
    mesh: triMesh(t.p.map((v) => [v.x, v.y, v.z])),
    model: mat4Identity(),
    material: { albedo: [t.c[0] / 255, t.c[1] / 255, t.c[2] / 255], specular: 0, shininess: 1, doubleSided: true },
  }));
  const fb = new Framebuffer(128, 128);
  new Renderer(fb).render(M7_CAMERA, items, "gouraud", M7_LIGHTS);
  const r = diffRGB(golden, framebufferToRaw(fb));
  assert.ok(r.ok, `golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

test("M7 场景: 三种颜色都有产出（含裁剪楔形）", () => {
  const golden = loadGolden("m7_clip.ppm");
  const has = (c: [number, number, number]): number => {
    let n = 0;
    for (let i = 0; i < 128 * 128; i++) {
      if (Math.abs(golden.rgb[i * 3] - c[0]) <= 1 && Math.abs(golden.rgb[i * 3 + 1] - c[1]) <= 1 && Math.abs(golden.rgb[i * 3 + 2] - c[2]) <= 1) n++;
    }
    return n;
  };
  assert.ok(has([210, 60, 60]) > 200, "red guard triangle visible");
  assert.ok(has([60, 200, 90]) > 100, "green near-clipped wedge visible");
  assert.ok(has([230, 210, 70]) > 30, "yellow two-behind wedge visible");
});

// ---------- 屏幕 guard ----------

test("M7 guard: 完全在相机后的三角形不产出任何像素", () => {
  const behind: DrawItem = {
    mesh: triMesh([[0.5, 0.2, 1], [-0.5, 0.2, 1], [0, -0.4, 2]]),
    model: mat4Identity(),
    material: { albedo: [1, 0, 0], specular: 0, shininess: 1, doubleSided: true },
  };
  const fb = new Framebuffer(64, 64);
  new Renderer(fb).render(M7_CAMERA, [behind], "gouraud", M7_LIGHTS);
  let nonBg = 0;
  for (let i = 0; i < 64 * 64; i++) {
    if (fb.color[i * 4] !== 0 || fb.color[i * 4 + 1] !== 0 || fb.color[i * 4 + 2] !== 0) nonBg++;
  }
  assert.equal(nonBg, 0, "fully-behind triangle must draw nothing");
});

test("M7 参考: 参考实现自检与黄金图一致", () => {
  assert.ok(diffRGB(loadGolden("m7_clip.ppm"), renderM7()).ok);
});
