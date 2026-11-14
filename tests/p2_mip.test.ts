import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Framebuffer } from "../src/framebuffer.js";
import { Renderer, mipSelect, type DrawItem, type Mesh } from "../src/pipeline.js";
import { checkerTexture, buildMip1 } from "../src/texture.js";
import { mat4Identity } from "../src/math.js";
import { decodePPM, framebufferToRaw, type RawImage } from "../src/ppm.js";
import { diffRGB } from "../src/diff.js";
import { P2M_CAMERA, P2M_GROUND_TEX, P2M_CUBE_TEX, makeP2MItems } from "../tools/reference/gen_p2_mip.js";

/** 正对相机、距离 2 的满屏四边形用的相机(看向 -Z,无视差)。 */
const ORTHOFACE_CAM = {
  eye: { x: 0, y: 0, z: 2 },
  target: { x: 0, y: 0, z: 0 },
  up: { x: 0, y: 1, z: 0 },
  fovYRad: (55 * Math.PI) / 180,
  aspect: 1,
  near: 0.1,
  far: 30,
};

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "..", "..", "golden");

function loadGolden(name: string): RawImage {
  return decodePPM(new Uint8Array(readFileSync(join(GOLDEN_DIR, name))));
}

/** 正式实现的场景装配(buildMip1 为被测对象)。 */
function formalItems(mip: boolean): DrawItem[] {
  const groundTex = checkerTexture(P2M_GROUND_TEX.size, P2M_GROUND_TEX.cells, P2M_GROUND_TEX.a, P2M_GROUND_TEX.b);
  const cubeTex = checkerTexture(P2M_CUBE_TEX.size, P2M_CUBE_TEX.cells, P2M_CUBE_TEX.a, P2M_CUBE_TEX.b);
  if (mip) {
    groundTex.mip = buildMip1(groundTex);
    cubeTex.mip = buildMip1(cubeTex);
  }
  return makeP2MItems(mip, groundTex, cubeTex);
}

function renderFormal(mip: boolean): RawImage {
  const fb = new Framebuffer(128, 128);
  new Renderer(fb).render(P2M_CAMERA, formalItems(mip), "textured");
  return framebufferToRaw(fb);
}

// ---------- 黄金对拍 ----------

test("P2 mip: mip 开(逐三角形选层)与独立参考实现逐像素一致(容差 0)", () => {
  const r = diffRGB(loadGolden("p2_mip_on.ppm"), renderFormal(true));
  assert.ok(r.ok, `mip-on golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

test("P2 mip: mip 关(恒 mip0)与独立参考实现逐像素一致(容差 0)", () => {
  const r = diffRGB(loadGolden("p2_mip_off.ppm"), renderFormal(false));
  assert.ok(r.ok, `mip-off golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

test("P2 mip: 放大四边形(低密度)density ≪ 1 → mip 开/关渲染逐像素相同(恒 mip0)", () => {
  const quad: Mesh = {
    positions: [-3, -3, -2, 3, -3, -2, 3, 3, -2, -3, 3, -2],
    normals: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    uvs: [0, 0, 1, 0, 1, 1, 0, 1],
    indices: [0, 2, 1, 0, 3, 2],
  };
  const items = (mip: boolean): DrawItem[] => {
    const tex = checkerTexture(32, 4, [200, 100, 60], [40, 30, 20]);
    if (mip) tex.mip = buildMip1(tex);
    return [{ mesh: quad, model: mat4Identity(), material: { albedo: [1, 1, 1], specular: 0, shininess: 1, texture: tex, sampler: "bilinear", mip, doubleSided: true } }];
  };
  const fbA = new Framebuffer(128, 128);
  const fbB = new Framebuffer(128, 128);
  new Renderer(fbA).render(ORTHOFACE_CAM, items(true), "textured");
  new Renderer(fbB).render(ORTHOFACE_CAM, items(false), "textured");
  const r = diffRGB(framebufferToRaw(fbA), framebufferToRaw(fbB));
  assert.ok(r.ok, `magnified quad must stay mip0: ${r.message}`);
  // 且确实画了东西(防止两边都空导致恒真)
  let nonBg = 0;
  for (let i = 0; i < 128 * 128; i++) {
    if (fbA.color[i * 4] !== 0 || fbA.color[i * 4 + 1] !== 0 || fbA.color[i * 4 + 2] !== 0) nonBg++;
  }
  assert.ok(nonBg > 10000, `quad should fill screen, got ${nonBg}`);
});

test("P2 mip: 缩小四边形(高密度,uv 8×8 重复)density ≥ 1 → mip 开/关渲染必须不同(选了 mip1)", () => {
  const quad: Mesh = {
    positions: [-3, -3, -2, 3, -3, -2, 3, 3, -2, -3, 3, -2],
    normals: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    uvs: [0, 0, 8, 0, 8, 8, 0, 8],
    indices: [0, 2, 1, 0, 3, 2],
  };
  const items = (mip: boolean): DrawItem[] => {
    const tex = checkerTexture(32, 4, [200, 100, 60], [40, 30, 20]);
    if (mip) tex.mip = buildMip1(tex);
    return [{ mesh: quad, model: mat4Identity(), material: { albedo: [1, 1, 1], specular: 0, shininess: 1, texture: tex, sampler: "bilinear", mip, doubleSided: true } }];
  };
  const fbA = new Framebuffer(128, 128);
  const fbB = new Framebuffer(128, 128);
  new Renderer(fbA).render(ORTHOFACE_CAM, items(true), "textured");
  new Renderer(fbB).render(ORTHOFACE_CAM, items(false), "textured");
  const r = diffRGB(framebufferToRaw(fbA), framebufferToRaw(fbB));
  assert.ok(!r.ok, `minified quad must switch to mip1: ${r.message} (${r.numDiff} px)`);
});

// ---------- 选层阈值锚定 ----------

test("P2 mip: mipSelect 阈值锚定 —— density 恰为 1 → mip1;略小于 1 → mip0", () => {
  assert.equal(mipSelect(1, 64, 64, 4096), true, "density exactly 1 → mip1");
  assert.equal(mipSelect(1, 64, 64, 4097), false, "density just below 1 → mip0");
  assert.equal(mipSelect(0, 64, 64, 1), false, "zero uv area → mip0");
  assert.equal(mipSelect(2, 32, 32, 2048), true, "density 1 → mip1");
});

test("P2 mip: buildMip1 box filter 锚定 —— 2×2 均值、奇数尺寸丢末行列", () => {
  // 2×2 纹理,四 texel 值 (10, 30, 50, 90) → mip1 单 texel = 45(R/G/B 同)
  const t = { width: 2, height: 2, data: new Uint8ClampedArray([10, 10, 10, 255, 30, 30, 30, 255, 50, 50, 50, 255, 90, 90, 90, 255]) };
  const m = buildMip1(t);
  assert.equal(m.width, 1);
  assert.equal(m.height, 1);
  assert.deepEqual([...m.data.slice(0, 4)], [45, 45, 45, 255]);
  // 3×3 → mip1 丢末行/末列:取左上 2×2
  const t3 = { width: 3, height: 3, data: new Uint8ClampedArray(3 * 3 * 4) };
  const put = (x: number, y: number, v: number): void => {
    const o = (y * 3 + x) * 4;
    t3.data[o] = v; t3.data[o + 1] = v; t3.data[o + 2] = v; t3.data[o + 3] = 255;
  };
  put(0, 0, 0); put(1, 0, 40); put(2, 0, 200);
  put(0, 1, 80); put(1, 1, 120); put(2, 1, 200);
  put(0, 2, 200); put(2, 2, 200); put(1, 2, 200);
  const m3 = buildMip1(t3);
  assert.equal(m3.width, 1);
  assert.equal(m3.height, 1);
  assert.deepEqual([...m3.data.slice(0, 4)], [60, 60, 60, 255]); // (0+40+80+120)/4 = 60
});
