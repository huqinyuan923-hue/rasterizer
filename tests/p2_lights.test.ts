import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Framebuffer } from "../src/framebuffer.js";
import { Renderer, type DrawItem } from "../src/pipeline.js";
import { shadePhongPixel, legacyLights, NO_LIGHTS, type LightSetup } from "../src/shading.js";
import { decodePPM, framebufferToRaw, type RawImage } from "../src/ppm.js";
import { diffRGB } from "../src/diff.js";
import {
  P2L_CAMERA, P2L_ITEMS, P2L_LIGHTS_MULTI, P2L_LIGHTS_SPOT, p2lMaterial,
  type P2LSetup,
} from "../tools/reference/gen_p2_lights.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "..", "..", "golden");

function loadGolden(name: string): RawImage {
  return decodePPM(new Uint8Array(readFileSync(join(GOLDEN_DIR, name))));
}

/** 参考侧 setup(P2LSetup)→ 正式实现 LightSetup(结构相同,数值引用一致)。 */
function toLightSetup(s: P2LSetup): LightSetup {
  return { ambient: s.ambient, dirs: s.dirs, points: s.points, spots: s.spots };
}

function formalItems(): DrawItem[] {
  return P2L_ITEMS.map((it) => ({ mesh: it.mesh, model: it.model, material: p2lMaterial(it) }));
}

function renderFormal(lights: LightSetup): RawImage {
  const fb = new Framebuffer(128, 128);
  new Renderer(fb).render(P2L_CAMERA, formalItems(), "phong", lights);
  return framebufferToRaw(fb);
}

// ---------- 黄金对拍 ----------

test("P2 光照: 多光源(2 方向 + 2 点光)与独立参考实现逐像素一致(容差 0)", () => {
  const r = diffRGB(loadGolden("p2_lights_multi.ppm"), renderFormal(toLightSetup(P2L_LIGHTS_MULTI)));
  assert.ok(r.ok, `multi-light golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

test("P2 光照: 聚光(2 软锥)与独立参考实现逐像素一致(容差 0)", () => {
  const r = diffRGB(loadGolden("p2_lights_spot.ppm"), renderFormal(toLightSetup(P2L_LIGHTS_SPOT)));
  assert.ok(r.ok, `spot golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

test("P2 光照: 两张黄金图确实不同(聚光锥形光斑可见)", () => {
  assert.ok(!diffRGB(loadGolden("p2_lights_multi.ppm"), loadGolden("p2_lights_spot.ppm")).ok);
});

test("P2 光照: Lambert 材质(立方体)在多光源下无高光 —— 与参考一致已隐含;此处直接断言蓝色通道形态", () => {
  // 立方体为 lambert;若高光未关闭,其面向点光的面会出现亮斑。取立方体区域像素,
  // 断言不存在"高光泛白"(r/g/b 全部 > 240)的像素。
  const img = loadGolden("p2_lights_multi.ppm");
  let blown = 0;
  for (let y = 40; y < 128; y++) {
    for (let x = 75; x < 128; x++) {
      const i = (y * 128 + x) * 3;
      if (img.rgb[i] > 240 && img.rgb[i + 1] > 240 && img.rgb[i + 2] > 240) blown++;
    }
  }
  assert.equal(blown, 0, "lambert cube must not show specular blowout");
});

// ---------- 聚光公式手工锚定 ----------

test("P2 光照 锚定: 轴上点 cosΘ=1 → t=1(钳位),漫反射+高光数值精确", () => {
  // 光 (0,0,2) 轴 (0,0,-1),cos30/cos60 软锥,k1=0.1;N=+Z 平面原点;eye (0,0,5)
  // dist=2, atten=1/1.2;ndl=1;V=(0,0,1),H=(0,0,2),ndh=1 → s=0.4
  // r = 0.5*1*1*(1/1.2) + 1*0.4*(1/1.2) = 0.9/1.2 = 0.75 → 191.25
  const setup = legacyLights([0, 0, 0], null, null);
  setup.spots = [{
    pos: { x: 0, y: 0, z: 2 },
    dir: [0, 0, -1],
    color: [1, 1, 1],
    cosInner: Math.cos(30 * Math.PI / 180),
    cosOuter: Math.cos(60 * Math.PI / 180),
    k1: 0.1, k2: 0,
  }];
  const out = new Float64Array(3);
  shadePhongPixel(0, 0, 1, 0, 0, 0, { x: 0, y: 0, z: 5 }, setup, 0.5, 0.5, 0.5, 0.4, 8, out);
  assert.ok(Math.abs(out[0] - 191.25) < 1e-9, `r: ${out[0]}`);
  assert.ok(Math.abs(out[2] - 191.25) < 1e-9);
});

test("P2 光照 锚定: 45° 离轴点 → t=(cos45−cos60)/(cos30−cos60),pow(1/√2,8)=1/16", () => {
  // 光 (0,0,10) 轴 (0,0,-1);P=(5,0,5),N=+Z,eye=(0,0,10)
  // L=normalize((-5,0,5)) → cosΘ = -(L·axis) = 1/√2;t = (1/√2 − 0.5)/(cos30 − 0.5)
  // ndl = 1/√2;V=L(对称),H=L+V 方向 ndh=1/√2 → pow = 1/16
  const ci = Math.cos(30 * Math.PI / 180), co = 0.5;
  const t = (Math.SQRT1_2 - co) / (ci - co);
  const setup = legacyLights([0.1, 0.1, 0.1], null, null);
  setup.spots = [{
    pos: { x: 0, y: 0, z: 10 },
    dir: [0, 0, -1],
    color: [1, 1, 1],
    cosInner: ci, cosOuter: co,
    k1: 0, k2: 0,
  }];
  const out = new Float64Array(3);
  shadePhongPixel(0, 0, 1, 5, 0, 5, { x: 0, y: 0, z: 10 }, setup, 0.5, 0.5, 0.5, 0.4, 8, out);
  const expect = 0.5 * 0.1 + 0.5 * 1 * Math.SQRT1_2 * t + 1 * (0.4 / 16) * t;
  assert.ok(Math.abs(out[0] - expect * 255) < 1e-9, `r: ${out[0]} vs ${expect * 255}`);
});

test("P2 光照 锚定: 锥外(cosΘ ≤ cosOuter)无贡献;零轴无贡献;退化硬锥(cosInner=cosOuter)in-cone 全亮", () => {
  const mk = (cosInner: number, cosOuter: number, dir: [number, number, number]): LightSetup => {
    const s = legacyLights([0, 0, 0], null, null);
    s.spots = [{ pos: { x: 0, y: 0, z: 2 }, dir, color: [1, 1, 1], cosInner, cosOuter, k1: 0, k2: 0 }];
    return s;
  };
  const out = new Float64Array(3);
  // 轴上但 cosOuter=1(半角 0°):cosΘ=1 不满足 > 1 → 无贡献
  shadePhongPixel(0, 0, 1, 0, 0, 0, { x: 0, y: 0, z: 5 }, mk(1, 1, [0, 0, -1]), 1, 1, 1, 0, 8, out);
  assert.ok(out[0] === 0, `hard edge must be empty, got ${out[0]}`);
  // 零轴:无贡献
  shadePhongPixel(0, 0, 1, 0, 0, 0, { x: 0, y: 0, z: 5 }, mk(0.9, 0.5, [0, 0, 0]), 1, 1, 1, 0, 8, out);
  assert.ok(out[0] === 0);
  // 退化硬锥:cosInner=cosOuter=0.5,轴上 cosΘ=1 > 0.5 → 全亮(t=1)
  shadePhongPixel(0, 0, 1, 0, 0, 0, { x: 0, y: 0, z: 5 }, mk(0.5, 0.5, [0, 0, -1]), 1, 1, 1, 0, 8, out);
  assert.ok(Math.abs(out[0] - 255) < 1e-9, `hard cone on-axis full: ${out[0]}`);
});

// ---------- Lambert / Blinn 开关与累积顺序 ----------

test("P2 光照: lambert(去高光)手工锚定 —— ambient+两光源漫反射,无 spec 项", () => {
  // albedo (0.3,0.2,0.1);ambient 0.1;dir L=+Z 色 0.7;点光 (0,0,10) 色 0.4 无衰减;N=+Z
  // lambert:r = 0.3*(0.1+0.7+0.4) = 0.36 → 91.8;g = 0.2*1.2 = 0.24 → 61.2;b = 0.1*1.2 = 0.12 → 30.6
  // (一期同场景 blinn 还要加 0.7*0.2 与 0.4*0.2 两项高光 → lambert 必须更暗)
  const setup = legacyLights(
    [0.1, 0.1, 0.1],
    { dir: [0, 0, 1], color: [0.7, 0.7, 0.7] },
    { pos: { x: 0, y: 0, z: 10 }, color: [0.4, 0.4, 0.4], k1: 0, k2: 0 },
  );
  const out = new Float64Array(3);
  shadePhongPixel(0, 0, 1, 0, 0, 0, { x: 0, y: 0, z: 5 }, setup, 0.3, 0.2, 0.1, 0.2, 8, out, false);
  assert.ok(Math.abs(out[0] - 91.8) < 1e-9, `lambert r: ${out[0]}`);
  assert.ok(Math.abs(out[1] - 61.2) < 1e-9, `lambert g: ${out[1]}`);
  assert.ok(Math.abs(out[2] - 30.6) < 1e-9, `lambert b: ${out[2]}`);
  // 同场景 blinn(r = 0.36 + 0.7*0.2 + 0.4*0.2 = 0.58 → 147.9,一期 m6 锚定)必须更亮 —— 开关确实生效
  const out2 = new Float64Array(3);
  shadePhongPixel(0, 0, 1, 0, 0, 0, { x: 0, y: 0, z: 5 }, setup, 0.3, 0.2, 0.1, 0.2, 8, out2, true);
  assert.ok(Math.abs(out2[0] - 147.9) < 1e-9, `blinn r: ${out2[0]}`);
  assert.ok(out2[0] > out[0], "blinn must be brighter than lambert here");
});

test("P2 光照: 多光源累积顺序冻结 —— 2 方向光的和与逐个累加一致(次序不敏感的量验证数组循环)", () => {
  const setup = legacyLights([0, 0, 0], null, null);
  setup.dirs = [
    { dir: [0, 0, 1], color: [0.5, 0, 0] },
    { dir: [0, 0, 1], color: [0, 0.5, 0] },
  ];
  const out = new Float64Array(3);
  shadePhongPixel(0, 0, 1, 0, 0, 0, { x: 0, y: 0, z: 5 }, setup, 1, 1, 1, 0, 8, out);
  assert.ok(Math.abs(out[0] - 127.5) < 1e-9 && Math.abs(out[1] - 127.5) < 1e-9 && out[2] === 0);
});

test("P2 光照: NO_LIGHTS 输出全黑(零环境、零光源)", () => {
  const out = new Float64Array(3);
  shadePhongPixel(0.3, 0.4, 0.5, 0, 0, 0, { x: 0, y: 0, z: 1 }, NO_LIGHTS, 0.5, 0.5, 0.5, 0.5, 8, out);
  assert.ok(out[0] === 0 && out[1] === 0 && out[2] === 0);
});
