/**
 * M3(二期)参考实现:背面剔除开关。
 * 场景:一个"开口盒"(立方体去掉朝向相机的 +Z 面)+ 其后方的红球;相机从开口正对看入。
 * - cull ON:盒的所有剩余面(内壁)都是背面 → 全被剔除 → 只见红球
 * - cull OFF:内壁双面绘制 → 盒内壁可见
 * 黄金图:p2_cull_on.ppm / p2_cull_off.ppm(两图必须不同)。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { makeCube, makeSphere } from "../../src/scene.js";
import { checkerTexture } from "../../src/texture.js";
import { mat4Translation, type Vec3 } from "../../src/math.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";
import { refRenderMesh, type RefCamera, type RefLights, type RefMaterial } from "./ref_render.js";

const W = 128;
const H = 128;

export const P2C_CAMERA: RefCamera = {
  eye: { x: 0, y: 0.1, z: 3.6 } as Vec3,
  target: { x: 0, y: 0, z: 0 } as Vec3,
  up: { x: 0, y: 1, z: 0 } as Vec3,
  fovYRad: (50 * Math.PI) / 180,
  aspect: 1,
  near: 0.1,
  far: 30,
};

export const P2C_LIGHTS: RefLights = {
  ambient: [0.1, 0.1, 0.12],
  dirs: [{ dir: [0.45, 0.8, 0.5], color: [0.85, 0.82, 0.75] }],
  points: [{ pos: { x: 2.0, y: 2.0, z: 2.0 }, color: [1.0, 0.75, 0.5], k1: 0.1, k2: 0.03 }],
};
{
  const d = P2C_LIGHTS.dirs[0].dir;
  const l = Math.hypot(d[0], d[1], d[2]);
  d[0] /= l; d[1] /= l; d[2] /= l;
}

/** 开口盒:makeCube(1.2) 去掉 +Z 面的 2 个三角形(索引 0..5)。 */
export function openBoxMesh() {
  const cube = makeCube(1.2);
  // 面片顺序:+Z 面先入(索引 0..5),其余面保留
  return {
    positions: cube.positions.slice(12), // 前 4 顶点属于 +Z 面(每面 4 顶点)
    normals: cube.normals.slice(12),
    uvs: cube.uvs.slice(8),
    indices: cube.indices.slice(6).map((i: number) => i - 4),
  };
}

export const P2C_BOX_MATERIAL: RefMaterial = {
  albedo: [1, 1, 1],
  specular: 0.3,
  shininess: 16,
  texture: checkerTexture(32, 4, [205, 150, 80], [90, 60, 40]),
  doubleSided: false,
};

export const P2C_SPHERE_MATERIAL: RefMaterial = {
  albedo: [0.85, 0.3, 0.22],
  specular: 0.55,
  shininess: 32,
};

export function renderP2C(cullOn: boolean): RawImage {
  const fb = new Framebuffer(W, H);
  refRenderMesh(fb, openBoxMesh(), mat4Translation({ x: 0, y: 0, z: 0 }), P2C_BOX_MATERIAL, P2C_CAMERA, P2C_LIGHTS, cullOn);
  refRenderMesh(fb, makeSphere(0.8, 32, 16), mat4Translation({ x: 0, y: 0, z: -2 }), P2C_SPHERE_MATERIAL, P2C_CAMERA, P2C_LIGHTS, true);
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("p2_cull_on.ppm", renderP2C(true));
  writeGolden("p2_cull_off.ppm", renderP2C(false));
  console.log("p2_cull_on.ppm / p2_cull_off.ppm written");
}

if (process.argv[1] && process.argv[1].endsWith("gen_p2_cull.js")) {
  run();
}
