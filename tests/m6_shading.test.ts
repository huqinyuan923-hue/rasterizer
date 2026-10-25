import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Framebuffer } from "../src/framebuffer.js";
import { Renderer, type DrawItem } from "../src/pipeline.js";
import { shadePhongPixel } from "../src/shading.js";
import { makeSphere, makeCube } from "../src/scene.js";
import { decodePPM, framebufferToRaw, type RawImage } from "../src/ppm.js";
import { diffRGB } from "../src/diff.js";
import { M6_CAMERA, M6_LIGHTS, M6_ITEMS } from "../tools/reference/gen_m6.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "..", "..", "golden");

function loadGolden(name: string): RawImage {
  return decodePPM(new Uint8Array(readFileSync(join(GOLDEN_DIR, name))));
}

function formalItems(): DrawItem[] {
  return M6_ITEMS.map((it) => ({
    mesh: it.mesh,
    model: it.model,
    material: { albedo: it.albedo, specular: it.specular, shininess: it.shininess },
  }));
}

function renderFormal(mode: "gouraud" | "phong"): RawImage {
  const fb = new Framebuffer(128, 128);
  new Renderer(fb).render(M6_CAMERA, formalItems(), mode, M6_LIGHTS);
  return framebufferToRaw(fb);
}

// ---------- 黄金对拍 ----------

test("M6 黄金: Gouraud 模式与参考实现逐像素一致", () => {
  const r = diffRGB(loadGolden("m6_gouraud.ppm"), renderFormal("gouraud"));
  assert.ok(r.ok, `gouraud golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

test("M6 黄金: Phong 模式与参考实现逐像素一致", () => {
  const r = diffRGB(loadGolden("m6_phong.ppm"), renderFormal("phong"));
  assert.ok(r.ok, `phong golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

test("M6 两档确实不同: Gouraud 与 Phong 黄金图存在可见差异（逐顶点 vs 逐像素光照）", () => {
  assert.ok(!diffRGB(loadGolden("m6_gouraud.ppm"), loadGolden("m6_phong.ppm")).ok);
});

// ---------- 手工锚定：轴对齐配置的精确期望值 ----------

test("M6 手工锚定: N=L=V=+Z 的 Blinn-Phong 数值精确可预言", () => {
  // ambient 0.1；方向光色 0.7 (L=+Z)；点光位于 (0,0,10) 色 0.4 无衰减；albedo (0.3,0.2,0.1)；spec 0.2 shin 8
  const setup = {
    ambient: [0.1, 0.1, 0.1] as [number, number, number],
    dir: { dir: [0, 0, 1] as [number, number, number], color: [0.7, 0.7, 0.7] as [number, number, number] },
    point: { pos: { x: 0, y: 0, z: 10 }, color: [0.4, 0.4, 0.4] as [number, number, number], k1: 0, k2: 0 },
  };
  const out = new Float64Array(3);
  shadePhongPixel(0, 0, 1, 0, 0, 0, { x: 0, y: 0, z: 5 }, setup, 0.3, 0.2, 0.1, 0.2, 8, out);
  // r = 0.3*0.1 + 0.3*0.7 + 0.7*0.2 + 0.3*0.4 + 0.4*0.2 = 0.58 → 147.9
  assert.ok(Math.abs(out[0] - 147.9) < 1e-9, `r: ${out[0]}`);
  assert.ok(Math.abs(out[1] - 117.3) < 1e-9, `g: ${out[1]}`);
  assert.ok(Math.abs(out[2] - 86.7) < 1e-9, `b: ${out[2]}`);
  // 量化（Uint8ClampedArray 写入语义：四舍五入）
  const q = new Uint8ClampedArray(3);
  q[0] = out[0]; q[1] = out[1]; q[2] = out[2];
  assert.deepEqual([...q], [148, 117, 87]);
});

test("M6 手工锚定: 半向量高光 off-axis（N·L>0 但偏离镜面方向）", () => {
  // L=(0.6,0,0.8)（单位），V=+Z：N·L=0.8 有漫反射；H=L+V=(0.6,0,1.8)，N·H=1.8/√3.6=3/√10
  // (3/√10)^8 = 6561/10000 = 0.6561；spec=0.2 → 总 r = 0.8 + 0.2·0.6561 = 0.93122 → ×255 = 237.4611
  const setup = {
    ambient: [0, 0, 0] as [number, number, number],
    dir: { dir: [0.6, 0, 0.8] as [number, number, number], color: [1, 1, 1] as [number, number, number] },
    point: null,
  };
  const out = new Float64Array(3);
  shadePhongPixel(0, 0, 1, 0, 0, 0, { x: 0, y: 0, z: 1 }, setup, 1, 1, 1, 0.2, 8, out);
  assert.ok(Math.abs(out[0] - 237.4611) < 1e-6, `diffuse+spec: ${out[0]}`);
});

test("M6 约定: N·L == 0 时整个光源无贡献（含高光，冻结约定）", () => {
  const setup = {
    ambient: [0, 0, 0] as [number, number, number],
    dir: { dir: [1, 0, 0] as [number, number, number], color: [1, 1, 1] as [number, number, number] },
    point: null,
  };
  const out = new Float64Array(3);
  shadePhongPixel(0, 0, 1, 0, 0, 0, { x: 0, y: 0, z: 1 }, setup, 1, 1, 1, 1, 8, out);
  assert.ok(out[0] === 0 && out[1] === 0 && out[2] === 0);
});

test("M6 手工锚定: 点光二次衰减 atten = 1/(1+k1·d+k2·d²)", () => {
  const setup = {
    ambient: [0, 0, 0] as [number, number, number],
    dir: null,
    point: { pos: { x: 0, y: 0, z: 2 }, color: [1, 1, 1] as [number, number, number], k1: 0.5, k2: 0.5 },
  };
  const out = new Float64Array(3);
  shadePhongPixel(0, 0, 1, 0, 0, 0, { x: 0, y: 0, z: 5 }, setup, 1, 1, 1, 0, 8, out);
  // dist=2 → atten = 1/(1+1+2) = 0.25 → 0.25*255 = 63.75
  assert.ok(Math.abs(out[0] - 63.75) < 1e-9, `atten: ${out[0]}`);
});

test("M6 背光: N·L ≤ 0 时该光源无贡献", () => {
  const setup = {
    ambient: [0, 0, 0] as [number, number, number],
    dir: { dir: [0, 0, -1] as [number, number, number], color: [1, 1, 1] as [number, number, number] },
    point: { pos: { x: 0, y: 0, z: -5 }, color: [1, 1, 1] as [number, number, number], k1: 0, k2: 0 },
  };
  const out = new Float64Array(3);
  shadePhongPixel(0, 0, 1, 0, 0, 0, { x: 0, y: 0, z: 5 }, setup, 1, 1, 1, 0.5, 8, out);
  assert.ok(out[0] === 0 && out[1] === 0 && out[2] === 0);
});

// ---------- 几何生成器 ----------

test("M6 几何: 球/立方体法线单位长、面向外侧、索引合法", () => {
  const meshes = [makeSphere(1, 16, 8), makeCube(1)];
  for (const m of meshes) {
    const vc = m.positions.length / 3;
    for (let t = 0; t < m.indices.length / 3; t++) {
      for (const k of [0, 1, 2]) {
        const i = m.indices[t * 3 + k];
        assert.ok(i >= 0 && i < vc, "index in range");
        const nl = Math.hypot(m.normals[i * 3], m.normals[i * 3 + 1], m.normals[i * 3 + 2]);
        assert.ok(Math.abs(nl - 1) < 1e-9, `normal unit length, got ${nl}`);
      }
      // 面法线朝外：cross(v1-v0, v2-v0) 与 (质心-原点) 同向（网格都在原点附近）
      const i0 = m.indices[t * 3], i1 = m.indices[t * 3 + 1], i2 = m.indices[t * 3 + 2];
      const ax = m.positions[i1 * 3] - m.positions[i0 * 3];
      const ay = m.positions[i1 * 3 + 1] - m.positions[i0 * 3 + 1];
      const az = m.positions[i1 * 3 + 2] - m.positions[i0 * 3 + 2];
      const bx = m.positions[i2 * 3] - m.positions[i0 * 3];
      const by = m.positions[i2 * 3 + 1] - m.positions[i0 * 3 + 1];
      const bz = m.positions[i2 * 3 + 2] - m.positions[i0 * 3 + 2];
      const cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
      const gx = (m.positions[i0 * 3] + m.positions[i1 * 3] + m.positions[i2 * 3]) / 3;
      const gy = (m.positions[i0 * 3 + 1] + m.positions[i1 * 3 + 1] + m.positions[i2 * 3 + 1]) / 3;
      const gz = (m.positions[i0 * 3 + 2] + m.positions[i1 * 3 + 2] + m.positions[i2 * 3 + 2]) / 3;
      const dot = cx * gx + cy * gy + cz * gz;
      const area2 = Math.hypot(cx, cy, cz);
      if (area2 > 1e-9) assert.ok(dot > 0, `face ${t} not outward (dot=${dot})`);
    }
  }
});

test("M6 场景: 正式渲染非空（球与立方体都贡献像素）", () => {
  const fb = new Framebuffer(128, 128);
  new Renderer(fb).render(M6_CAMERA, formalItems(), "phong", M6_LIGHTS);
  let nonBg = 0;
  for (let i = 0; i < 128 * 128; i++) {
    if (fb.color[i * 4] !== 0 || fb.color[i * 4 + 1] !== 0 || fb.color[i * 4 + 2] !== 0) nonBg++;
  }
  assert.ok(nonBg > 3000, `scene should draw plenty of pixels, got ${nonBg}`);
});
