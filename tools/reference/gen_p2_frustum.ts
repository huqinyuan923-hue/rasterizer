/**
 * M3(二期)参考实现:视锥剔除像素安全性。
 * 场景 5 物体:中心球、右侧立方体、左缘跨骑球(部分可见)、左外球(完全出视锥,应被剔除
 * 但参考实现照画 —— 屏幕边界 guard 使其产出 0 像素)、远平面外小球(本管线不裁远平面,
 * 仍被绘制 —— 钉死"远平面不裁剪"语义)。
 * 黄金图:p2_frustum.ppm = 参考实现全量渲染(无剔除)。正式管线 cullFrustum 默认开,
 * 剔除只跳过 0 像素项 → 渲染必须与黄金图逐像素一致。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { makeCube, makeSphere } from "../../src/scene.js";
import { mat4Translation, type Vec3 } from "../../src/math.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";
import { refRenderMesh, type RefCamera, type RefLights, type RefMaterial } from "./ref_render.js";

const W = 128;
const H = 128;

export const P2F_CAMERA: RefCamera = {
  eye: { x: 0, y: 0, z: 5 } as Vec3,
  target: { x: 0, y: 0, z: 0 } as Vec3,
  up: { x: 0, y: 1, z: 0 } as Vec3,
  fovYRad: (50 * Math.PI) / 180,
  aspect: 1,
  near: 0.1,
  far: 20,
};

export const P2F_LIGHTS: RefLights = {
  ambient: [0.1, 0.1, 0.12],
  dirs: [{ dir: [0.45, 0.8, 0.5], color: [0.85, 0.82, 0.75] }],
  points: [{ pos: { x: 2.0, y: 2.0, z: 2.0 }, color: [1.0, 0.75, 0.5], k1: 0.1, k2: 0.03 }],
};
{
  const d = P2F_LIGHTS.dirs[0].dir;
  const l = Math.hypot(d[0], d[1], d[2]);
  d[0] /= l; d[1] /= l; d[2] /= l;
}

/** 场景清单(mesh/model/material);参考实现全量渲染,正式实现默认视锥剔除。 */
export const P2F_ITEMS: Array<{ mesh: ReturnType<typeof makeSphere>; model: ReturnType<typeof mat4Translation>; material: RefMaterial }> = [
  { mesh: makeSphere(0.9, 32, 16), model: mat4Translation({ x: 0, y: 0, z: 0 }), material: { albedo: [0.85, 0.3, 0.22], specular: 0.55, shininess: 32 } },
  { mesh: makeCube(0.8), model: mat4Translation({ x: 1.9, y: -0.3, z: -0.5 }), material: { albedo: [0.25, 0.55, 0.85], specular: 0.3, shininess: 16 } },
  { mesh: makeSphere(0.9, 32, 16), model: mat4Translation({ x: -2.9, y: 0.1, z: -1.0 }), material: { albedo: [0.2, 0.75, 0.65], specular: 0.4, shininess: 24 } },
  { mesh: makeSphere(0.6, 24, 12), model: mat4Translation({ x: -6.8, y: 0, z: -1.0 }), material: { albedo: [0.9, 0.85, 0.2], specular: 0.2, shininess: 8 } },
  { mesh: makeSphere(0.5, 24, 12), model: mat4Translation({ x: 0.7, y: -0.2, z: -24 }), material: { albedo: [0.7, 0.4, 0.85], specular: 0.2, shininess: 8 } },
];

export function renderP2F(): RawImage {
  const fb = new Framebuffer(W, H);
  for (const it of P2F_ITEMS) {
    refRenderMesh(fb, it.mesh, it.model, it.material, P2F_CAMERA, P2F_LIGHTS, true);
  }
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("p2_frustum.ppm", renderP2F());
  console.log("p2_frustum.ppm written");
}

if (process.argv[1] && process.argv[1].endsWith("gen_p2_frustum.js")) {
  run();
}
