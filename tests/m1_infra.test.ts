import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { mulberry32 } from "../src/rng.js";
import { Framebuffer } from "../src/framebuffer.js";
import { encodePPM, decodePPM, framebufferToRaw, type RawImage } from "../src/ppm.js";
import { diffRGB } from "../src/diff.js";
import { mat4Perspective, mat4Ortho, mat4LookAt, mat4ApplyVec4, mat4Mul, mat4Translation, mat4RotY } from "../src/math.js";
import { renderM1 } from "../tools/reference/gen_m1.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "..", "..", "golden");

function loadGolden(name: string): RawImage {
  const bytes = readFileSync(join(GOLDEN_DIR, name));
  return decodePPM(new Uint8Array(bytes));
}

// ---------- 种子 PRNG ----------

test("mulberry32: 同种子序列确定，异种子序列不同", () => {
  const a1 = mulberry32(42);
  const a2 = mulberry32(42);
  for (let i = 0; i < 1000; i++) assert.equal(a1(), a2());
  const b = mulberry32(43);
  let differs = false;
  const a3 = mulberry32(42);
  for (let i = 0; i < 100; i++) if (a3() !== b()) { differs = true; break; }
  assert.ok(differs, "different seeds must diverge");
});

test("mulberry32: 输出落在 [0,1)", () => {
  const r = mulberry32(7);
  for (let i = 0; i < 10000; i++) {
    const v = r();
    assert.ok(v >= 0 && v < 1, `out of range: ${v}`);
  }
});

// ---------- PPM 编解码 ----------

test("PPM: encode→decode 往返无损", () => {
  const W = 37, H = 23;
  const fb = new Framebuffer(W, H);
  const rnd = mulberry32(99);
  for (let i = 0; i < W * H; i++) {
    fb.color[i * 4] = Math.floor(rnd() * 256);
    fb.color[i * 4 + 1] = Math.floor(rnd() * 256);
    fb.color[i * 4 + 2] = Math.floor(rnd() * 256);
  }
  const img = framebufferToRaw(fb);
  const decoded = decodePPM(encodePPM(img));
  assert.equal(decoded.width, W);
  assert.equal(decoded.height, H);
  assert.ok(diffRGB(img, decoded).ok, "roundtrip must be lossless");
});

test("PPM: 解析器容忍头部注释与多余空白", () => {
  const w = 2, h = 2;
  const header = "P6 # comment line\n# another\n 2\t 2\n# before maxval\n255\n";
  const hb = Uint8Array.from(header, (c) => c.charCodeAt(0));
  const pixels = Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const bytes = new Uint8Array(hb.length + pixels.length);
  bytes.set(hb, 0); bytes.set(pixels, hb.length);
  const img = decodePPM(bytes);
  assert.equal(img.width, w);
  assert.equal(img.height, h);
  assert.deepEqual([...img.rgb], [...pixels]);
});

test("PPM: 损坏输入抛错", () => {
  assert.throws(() => decodePPM(Uint8Array.from("P5\n1 1\n255\n".concat("x"), (c) => c.charCodeAt(0))));
  assert.throws(() => decodePPM(Uint8Array.from("P6\n10 10\n255\n".concat("x"), (c) => c.charCodeAt(0)))); // 数据不足
});

// ---------- diff 工具 ----------

test("diff: 相同图通过；单像素差异被精确定位", () => {
  const img: RawImage = { width: 4, height: 3, rgb: new Uint8Array(4 * 3 * 3).fill(100) };
  assert.ok(diffRGB(img, img).ok);

  const mutated: RawImage = { width: 4, height: 3, rgb: img.rgb.slice() };
  mutated.rgb[(1 * 4 + 2) * 3 + 1] = 101; // pixel (2,1) green +1
  const r = diffRGB(img, mutated);
  assert.ok(!r.ok);
  assert.equal(r.numDiff, 1);
  assert.equal(r.maxDiff, 1);
  assert.ok(r.first);
  assert.equal(r.first.x, 2);
  assert.equal(r.first.y, 1);
  assert.deepEqual(r.first.b, [100, 101, 100]);
});

test("diff: 尺寸不同直接失败", () => {
  const a: RawImage = { width: 4, height: 3, rgb: new Uint8Array(36) };
  const b: RawImage = { width: 3, height: 3, rgb: new Uint8Array(27) };
  assert.ok(!diffRGB(a, b).ok);
});

// ---------- 数学库抽查 ----------

test("透视矩阵: near/far 平面映射到 ndc z = -1/+1，w = -z_view", () => {
  const P = mat4Perspective(Math.PI / 3, 1, 0.5, 20);
  const atNear = mat4ApplyVec4(P, { x: 0, y: 0, z: -0.5, w: 1 });
  assert.equal(atNear.w, 0.5);
  assert.ok(Math.abs(atNear.z / atNear.w - -1) < 1e-12);
  const atFar = mat4ApplyVec4(P, { x: 0, y: 0, z: -20, w: 1 });
  assert.equal(atFar.w, 20);
  assert.ok(Math.abs(atFar.z / atFar.w - 1) < 1e-12);
});

test("正交矩阵: w 恒为 1（屏幕空间线性 z 在正交下精确）", () => {
  const O = mat4Ortho(-2, 2, -2, 2, 0.5, 20);
  const p = mat4ApplyVec4(O, { x: 1, y: -1, z: -5, w: 1 });
  assert.equal(p.w, 1);
  assert.ok(Math.abs(p.x - 0.5) < 1e-12);   // x=1 ∈ [-2,2] → ndc 0.5
  assert.ok(Math.abs(p.y - -0.5) < 1e-12);  // y=-1 ∈ [-2,2] → ndc -0.5
});

test("lookAt: 相机 (0,0,5) 看原点，原点视空间 z = -5", () => {
  const V = mat4LookAt({ x: 0, y: 0, z: 5 }, { x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 0 });
  const v = mat4ApplyVec4(V, { x: 0, y: 0, z: 0, w: 1 });
  assert.ok(Math.abs(v.z - -5) < 1e-12);
  assert.ok(Math.abs(v.x) < 1e-12 && Math.abs(v.y) < 1e-12);
});

test("mat4Mul: T*R 先旋转后平移（列向量右乘）", () => {
  const quarter = Math.PI / 2;
  const T = mat4Translation({ x: 1, y: 0, z: 0 });
  const R = mat4RotY(quarter);
  const TR = mat4Mul(T, R);
  // 点 (1,0,0)：R 绕 Y 转 90° → (0,0,-1)（右手系），再平移 → (1,0,-1)
  const p = mat4ApplyVec4(TR, { x: 1, y: 0, z: 0, w: 1 });
  assert.ok(Math.abs(p.x - 1) < 1e-12);
  assert.ok(Math.abs(p.y - 0) < 1e-12);
  assert.ok(Math.abs(p.z - -1) < 1e-12);
});

// ---------- M1 黄金基线 ----------

test("M1 黄金基线: 正式渲染与黄金图逐像素一致（容差 0）", () => {
  const golden = loadGolden("m1_baseline.ppm");
  assert.equal(golden.width, 96);
  assert.equal(golden.height, 96);
  const live = renderM1();
  const r = diffRGB(golden, live);
  assert.ok(r.ok, `golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

test("M1 黄金基线: 抽查背景渐变与确定性", () => {
  // 背景渐变公式抽查：y=0 → (18,24,48)，y=95 → (58,114,198)
  const fb = new Framebuffer(96, 96);
  for (let y = 0; y < 96; y++) {
    const t = y / 95;
    for (let x = 0; x < 96; x++) {
      fb.setPixel(x, y, Math.round(18 + t * 40), Math.round(24 + t * 90), Math.round(48 + t * 150));
    }
  }
  const img = framebufferToRaw(fb);
  assert.deepEqual([img.rgb[0], img.rgb[1], img.rgb[2]], [18, 24, 48]);
  const last = (95 * 96 + 0) * 3;
  assert.deepEqual([img.rgb[last], img.rgb[last + 1], img.rgb[last + 2]], [58, 114, 198]);
});
