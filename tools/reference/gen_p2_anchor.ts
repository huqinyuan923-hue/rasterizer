/**
 * 二期锚点黄金图:p2_scenegraph_anchor.ppm。
 *
 * 生成时机:场景图迁移**之前**（2026-11-15,升级会话开工第一步）。
 * 内容:升级前的演示场景(buildDemoScene 的当时构造,此处**硬编码冻结**,不引用
 * buildDemoScene —— 即使将来 scene.ts 改写,本夹具仍渲染升级前的几何/材质/光照),
 * 在固定相机与固定动画时刻 t=1.25 下的一帧 phong 渲染。
 *
 * 用途:验收 #2「场景图等价」—— 迁移后经场景图展平渲染的旧 demo 必须与
 * 本锚点逐像素一致(容差 0)。本脚本**不加入 gen_all**:重新生成锚点会使
 * 对拍退化为自我引用(若迁移被改坏,重跑会悄悄祝福坏结果)。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { Renderer, type Mesh, type DrawItem } from "../../src/pipeline.js";
import { makeCube, makeSphere, makeTorus, stripeTexture } from "../../src/scene.js";
import { checkerTexture } from "../../src/texture.js";
import {
  mat4Mul, mat4RotX, mat4RotY, mat4Translation, mat4Identity,
  type Mat4,
} from "../../src/math.js";
import { framebufferToRaw } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";

const W = 256;
const H = 256;
const ANIM_T = 1.25;

// ---- 硬编码的升级前场景构造(与 2026-10-25 版 buildDemoScene 逐值一致) ----

function compose(...ms: Mat4[]): Mat4 {
  let out = ms[0];
  for (let i = 1; i < ms.length; i++) out = mat4Mul(out, ms[i]);
  return out;
}

function anchorItems(): { items: DrawItem[]; lights: Parameters<Renderer["render"]>[3] } {
  const groundTex = checkerTexture(128, 12, [190, 184, 172], [70, 76, 82]);
  const crateTex = checkerTexture(64, 4, [205, 150, 80], [138, 94, 50]);
  const torusTex = stripeTexture(64, 8, [36, 158, 148], [232, 230, 224]);

  const cubeMesh = makeCube(0.9);
  const sphereMesh = makeSphere(0.95, 48, 24);
  const torusMesh = makeTorus(0.85, 0.38, 48, 24);
  const groundMesh: Mesh = {
    positions: [-15, -1.3, -15, 15, -1.3, -15, 15, -1.3, 15, -15, -1.3, 15],
    normals: [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    uvs: [0, 0, 1, 0, 1, 1, 0, 1],
    indices: [0, 2, 1, 0, 3, 2],
  };

  const cube: DrawItem = {
    mesh: cubeMesh, model: mat4Translation({ x: -1.7, y: -0.85, z: 0.2 }),
    material: { albedo: [1, 1, 1], specular: 0.35, shininess: 24, texture: crateTex, sampler: "nearest" },
  };
  const sphere: DrawItem = {
    mesh: sphereMesh, model: mat4Translation({ x: 0.05, y: -0.35, z: 0 }),
    material: { albedo: [0.85, 0.3, 0.22], specular: 0.6, shininess: 48 },
  };
  const torus: DrawItem = {
    mesh: torusMesh, model: mat4Translation({ x: 1.9, y: -0.92, z: -0.1 }),
    material: { albedo: [1, 1, 1], specular: 0.85, shininess: 64, texture: torusTex, sampler: "bilinear" },
  };
  const ground: DrawItem = {
    mesh: groundMesh, model: mat4Identity(),
    material: { albedo: [1, 1, 1], specular: 0.06, shininess: 8, texture: groundTex, sampler: "bilinear", doubleSided: true },
  };

  const lights = {
    ambient: [0.12, 0.12, 0.14] as [number, number, number],
    dirs: [{ dir: [0.45, 0.8, 0.5] as [number, number, number], color: [0.85, 0.82, 0.75] as [number, number, number] }],
    points: [{ pos: { x: 3.2, y: 2.4, z: 0 }, color: [1.0, 0.78, 0.55] as [number, number, number], k1: 0.08, k2: 0.02 }],
    spots: [] as never[],
  };
  {
    const d = lights.dirs[0].dir;
    const l = Math.hypot(d[0], d[1], d[2]);
    d[0] /= l; d[1] /= l; d[2] /= l;
  }

  // animate(1.25) 冻结副本(升级前逐值一致)
  cube.model = compose(
    mat4Translation({ x: -1.7, y: -0.85, z: 0.2 }),
    mat4RotY(ANIM_T * 0.5),
  );
  torus.model = compose(
    mat4Translation({ x: 1.9, y: -0.92, z: -0.1 }),
    mat4RotY(ANIM_T * 0.7),
    mat4RotX(-0.14),
  );
  lights.points[0].pos.x = 3.2 * Math.cos(ANIM_T * 0.6);
  lights.points[0].pos.y = 2.4 + 0.5 * Math.sin(ANIM_T * 0.9);
  lights.points[0].pos.z = 3.2 * Math.sin(ANIM_T * 0.6);

  return { items: [ground, cube, sphere, torus], lights };
}

// ---- 相机(demoOrbitCamera(0.55, 0.34, 6.6) 冻结副本) ----

function anchorCamera() {
  const yaw = 0.55, pitch = 0.34, dist = 6.6;
  const target = { x: 0, y: -0.3, z: 0 };
  const cp = Math.cos(pitch);
  return {
    eye: {
      x: target.x + dist * Math.sin(yaw) * cp,
      y: target.y + dist * Math.sin(pitch),
      z: target.z + dist * Math.cos(yaw) * cp,
    },
    target,
    up: { x: 0, y: 1, z: 0 },
    fovYRad: (55 * Math.PI) / 180,
    aspect: 1,
    near: 0.1,
    far: 60,
  };
}

export function renderAnchor() {
  const fb = new Framebuffer(W, H);
  const { items, lights } = anchorItems();
  new Renderer(fb).render(anchorCamera(), items, "phong", lights);
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("p2_scenegraph_anchor.ppm", renderAnchor());
  console.log("p2_scenegraph_anchor.ppm written");
}

if (process.argv[1] && process.argv[1].endsWith("gen_p2_anchor.js")) {
  run();
}
