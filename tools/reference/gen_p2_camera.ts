/**
 * M4(二期)参考实现:正交投影相机。
 * 场景:三球一盒;相机为正交投影(orthographic half-height 2.2,俯视 30°)。
 * 黄金图:p2_ortho.ppm —— 参考实现自带 mat4Ortho 公式副本 + 顶点阶段副本;
 * 正式管线 camera.projection = "ortho" 必须逐像素一致。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { makeCube, makeSphere } from "../../src/scene.js";
import { mat4Translation, type Vec3 } from "../../src/math.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";
import { refRenderMesh, type RefCamera, type RefLights, type RefMaterial } from "./ref_render.js";

const W = 128;
const H = 128;

export const P2K_CAMERA: RefCamera = {
  eye: { x: 3.5, y: 3.0, z: 3.5 } as Vec3,
  target: { x: 0, y: 0, z: 0 } as Vec3,
  up: { x: 0, y: 1, z: 0 } as Vec3,
  fovYRad: 0, // 正交模式下不使用
  aspect: 1,
  near: 0.1,
  far: 30,
  ortho: { left: -2.6, right: 2.6, bottom: -2.6, top: 2.6 },
};

export const P2K_LIGHTS: RefLights = {
  ambient: [0.1, 0.1, 0.12],
  dirs: [{ dir: [0.45, 0.8, 0.5], color: [0.85, 0.82, 0.75] }],
  points: [{ pos: { x: 2.5, y: 3.0, z: 2.0 }, color: [1.0, 0.8, 0.55], k1: 0.08, k2: 0.02 }],
};
{
  const d = P2K_LIGHTS.dirs[0].dir;
  const l = Math.hypot(d[0], d[1], d[2]);
  d[0] /= l; d[1] /= l; d[2] /= l;
}

/** 场景清单(与正式测试共用)。 */
export const P2K_ITEMS: Array<{ mesh: ReturnType<typeof makeSphere>; model: ReturnType<typeof mat4Translation>; material: RefMaterial }> = [
  { mesh: makeSphere(0.7, 32, 16), model: mat4Translation({ x: -0.9, y: 0, z: 0 }), material: { albedo: [0.85, 0.3, 0.22], specular: 0.5, shininess: 32 } },
  { mesh: makeSphere(0.7, 32, 16), model: mat4Translation({ x: 0.9, y: 0, z: 0.4 }), material: { albedo: [0.25, 0.55, 0.85], specular: 0.5, shininess: 32 } },
  { mesh: makeCube(0.9), model: mat4Translation({ x: 0, y: 0, z: -1.3 }), material: { albedo: [0.3, 0.7, 0.4], specular: 0.3, shininess: 16 } },
  { mesh: makeSphere(0.45, 24, 12), model: mat4Translation({ x: 0, y: 1.2, z: 0.2 }), material: { albedo: [0.9, 0.8, 0.3], specular: 0.4, shininess: 24 } },
];

export function renderP2K(): RawImage {
  const fb = new Framebuffer(W, H);
  for (const it of P2K_ITEMS) refRenderMesh(fb, it.mesh, it.model, it.material, P2K_CAMERA, P2K_LIGHTS, true);
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("p2_ortho.ppm", renderP2K());
  console.log("p2_ortho.ppm written");
}

if (process.argv[1] && process.argv[1].endsWith("gen_p2_camera.js")) {
  run();
}
