import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Framebuffer } from "../src/framebuffer.js";
import { Renderer, type DrawItem } from "../src/pipeline.js";
import { mat4Ortho, mat4ApplyVec4 } from "../src/math.js";
import { decodePPM, framebufferToRaw, type RawImage } from "../src/ppm.js";
import { diffRGB } from "../src/diff.js";
import { P2K_CAMERA, P2K_LIGHTS, P2K_ITEMS } from "../tools/reference/gen_p2_camera.js";
import { P2S_CAMERA, P2S_LIGHTS, P2S_ITEMS } from "../tools/reference/gen_p2_sort.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "..", "..", "golden");

function loadGolden(name: string): RawImage {
  return decodePPM(new Uint8Array(readFileSync(join(GOLDEN_DIR, name))));
}

type RefLightsLike = typeof P2K_LIGHTS;
function lightsOf(l: RefLightsLike) {
  return { ambient: l.ambient, dirs: l.dirs, points: l.points, spots: [] as never[] };
}

// ---------- 正交相机 ----------

test("P2 相机: mat4Ortho 锚定 —— 已知点投影落点精确", () => {
  // 视景体 [-2,2]×[-2,2],near 0.1 far 30:点 (1, 0.5, -10) 应投到 NDC (0.5, 0.25)
  const proj = mat4Ortho(-2, 2, -2, 2, 0.1, 30);
  const p = mat4ApplyVec4(proj, { x: 1, y: 0.5, z: -10, w: 1 });
  assert.ok(Math.abs(p.x / p.w - 0.5) < 1e-12, `x: ${p.x}`);
  assert.ok(Math.abs(p.y / p.w - 0.25) < 1e-12, `y: ${p.y}`);
  assert.ok(p.w === 1, "ortho w must be 1");
});

test("P2 相机: 正交投影黄金对拍 —— 正式管线 projection='ortho' 与参考逐像素一致(容差 0)", () => {
  const fb = new Framebuffer(128, 128);
  const items: DrawItem[] = P2K_ITEMS.map((it) => ({
    mesh: it.mesh,
    model: it.model,
    material: { ...it.material, cullBackface: true },
  }));
  const cam = {
    ...P2K_CAMERA,
    projection: "ortho" as const,
    left: P2K_CAMERA.ortho!.left,
    right: P2K_CAMERA.ortho!.right,
    bottom: P2K_CAMERA.ortho!.bottom,
    top: P2K_CAMERA.ortho!.top,
  };
  new Renderer(fb).render(cam, items, "phong", lightsOf(P2K_LIGHTS));
  const r = diffRGB(loadGolden("p2_ortho.ppm"), framebufferToRaw(fb));
  assert.ok(r.ok, `ortho golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

// ---------- 不透明排序 ----------

test("P2 排序: sortOpaque 开启后像素与黄金图逐像素一致(深度仲裁,顺序无关)", () => {
  const fb = new Framebuffer(128, 128);
  const items: DrawItem[] = P2S_ITEMS.map((it) => ({
    mesh: it.mesh,
    model: it.model,
    material: { ...it.material, cullBackface: true },
  }));
  new Renderer(fb).render(P2S_CAMERA, items, "phong", lightsOf(P2S_LIGHTS), { sortOpaque: true });
  const r = diffRGB(loadGolden("p2_sort.ppm"), framebufferToRaw(fb));
  assert.ok(r.ok, `sort golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

test("P2 排序: sortOpaque 开/关在无平局场景下像素一致;排序结果为距离升序且稳定", () => {
  const renderWith = (sort: boolean): RawImage => {
    const fb = new Framebuffer(128, 128);
    const items: DrawItem[] = P2S_ITEMS.map((it) => ({
      mesh: it.mesh,
      model: it.model,
      material: { ...it.material, cullBackface: true },
    }));
    new Renderer(fb).render(P2S_CAMERA, items, "phong", lightsOf(P2S_LIGHTS), { sortOpaque: sort });
    return framebufferToRaw(fb);
  };
  const r = diffRGB(renderWith(true), renderWith(false));
  assert.ok(r.ok, `sorted vs unsorted must be identical (depth-tested): ${r.message}`);
  // 排序键本身:红球最近、绿球最远(距离平方升序)
  const eye = P2S_CAMERA.eye;
  const dist = (m: { x: number; y: number; z: number }): number => {
    const dx = m.x - eye.x, dy = m.y - eye.y, dz = m.z - eye.z;
    return dx * dx + dy * dy + dz * dz;
  };
  const dRed = dist({ x: -0.45, y: 0, z: 1 });
  const dBlue = dist({ x: 0, y: 0, z: 0 });
  const dGreen = dist({ x: 0.45, y: 0, z: -1 });
  assert.ok(dRed < dBlue && dBlue < dGreen, "front-to-back ordering expected");
});

// ---------- 视锥剔除开关回归(默认开,可关) ----------

test("P2 管线: cullFrustum=false 渲染与默认(开)一致 —— 无出视锥物体时剔除不改变像素", () => {
  const renderWith = (cull: boolean): RawImage => {
    const fb = new Framebuffer(128, 128);
    const items: DrawItem[] = P2S_ITEMS.map((it) => ({
      mesh: it.mesh,
      model: it.model,
      material: { ...it.material, cullBackface: true },
    }));
    new Renderer(fb).render(P2S_CAMERA, items, "phong", lightsOf(P2S_LIGHTS), { cullFrustum: cull });
    return framebufferToRaw(fb);
  };
  const r = diffRGB(renderWith(true), renderWith(false));
  assert.ok(r.ok, `cull on/off must not change pixels: ${r.message}`);
});
