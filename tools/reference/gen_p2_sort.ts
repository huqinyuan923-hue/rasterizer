/**
 * M4(二期)参考实现:不透明物体 front-to-back 排序(像素不变性)。
 * 场景:三个相互重叠的球,固定绘制顺序为红→蓝→绿(参考实现按此顺序渲染,深度缓冲仲裁)。
 * 黄金图:p2_sort.ppm = 参考实现按固定顺序渲染。
 * 正式管线 opts.sortOpaque = true 时按距离平方升序稳定排序 —— 深度测试使像素与
 * 绘制顺序无关(无共面平局场景) → 必须与黄金图逐像素一致。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { makeSphere } from "../../src/scene.js";
import { mat4Translation, type Vec3 } from "../../src/math.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";
import { refRenderMesh, type RefCamera, type RefLights, type RefMaterial } from "./ref_render.js";

const W = 128;
const H = 128;

export const P2S_CAMERA: RefCamera = {
  eye: { x: 0, y: 0, z: 5 } as Vec3,
  target: { x: 0, y: 0, z: 0 } as Vec3,
  up: { x: 0, y: 1, z: 0 } as Vec3,
  fovYRad: (50 * Math.PI) / 180,
  aspect: 1,
  near: 0.1,
  far: 30,
};

export const P2S_LIGHTS: RefLights = {
  ambient: [0.1, 0.1, 0.12],
  dirs: [{ dir: [0.45, 0.8, 0.5], color: [0.85, 0.82, 0.75] }],
  points: [],
};
{
  const d = P2S_LIGHTS.dirs[0].dir;
  const l = Math.hypot(d[0], d[1], d[2]);
  d[0] /= l; d[1] /= l; d[2] /= l;
}

/** 三球重叠场景:红(近)在 z=1、蓝(中) z=0、绿(远) z=-1,x 上互相咬合。 */
export const P2S_ITEMS: Array<{ mesh: ReturnType<typeof makeSphere>; model: ReturnType<typeof mat4Translation>; material: RefMaterial }> = [
  { mesh: makeSphere(0.9, 32, 16), model: mat4Translation({ x: -0.45, y: 0, z: 1 }), material: { albedo: [0.85, 0.25, 0.2], specular: 0.5, shininess: 32 } },
  { mesh: makeSphere(0.9, 32, 16), model: mat4Translation({ x: 0, y: 0, z: 0 }), material: { albedo: [0.25, 0.45, 0.85], specular: 0.5, shininess: 32 } },
  { mesh: makeSphere(0.9, 32, 16), model: mat4Translation({ x: 0.45, y: 0, z: -1 }), material: { albedo: [0.25, 0.75, 0.35], specular: 0.5, shininess: 32 } },
];

export function renderP2S(): RawImage {
  const fb = new Framebuffer(W, H);
  for (const it of P2S_ITEMS) refRenderMesh(fb, it.mesh, it.model, it.material, P2S_CAMERA, P2S_LIGHTS, true);
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("p2_sort.ppm", renderP2S());
  console.log("p2_sort.ppm written");
}

if (process.argv[1] && process.argv[1].endsWith("gen_p2_sort.js")) {
  run();
}
