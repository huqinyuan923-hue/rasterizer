/**
 * 程序化几何生成器（确定性，无随机）。网格约定：
 * - 三角形顶点序从外部看为逆时针（cross(v1-v0, v2-v0) 指向外侧）—— 与背面剔除约定配套。
 * - 法线为平滑（球）或面法线（立方体），单位长度。
 * - uv ∈ [0,1]。
 */

import type { Mesh, DrawItem, Camera } from "./pipeline.js";
import type { LightSetup } from "./shading.js";
import { mat4Mul, mat4RotY, mat4RotX, mat4Translation, mat4Identity, type Mat4 } from "./math.js";
import { checkerTexture, type Texture } from "./texture.js";

/** 立方体：size 为边长，中心在原点。24 顶点（每面独立法线与 uv），12 三角形。 */
export function makeCube(size = 1): Mesh {
  const h = size / 2;
  const faces: Array<{ n: [number, number, number]; verts: [number, number, number][] }> = [
    { n: [0, 0, 1], verts: [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]] },     // +Z
    { n: [0, 0, -1], verts: [[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]] }, // -Z
    { n: [1, 0, 0], verts: [[h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h]] },     // +X
    { n: [-1, 0, 0], verts: [[-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h]] }, // -X
    { n: [0, 1, 0], verts: [[-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h]] },     // +Y
    { n: [0, -1, 0], verts: [[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]] }, // -Y
  ];
  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  const faceUV: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];
  for (const f of faces) {
    const base = positions.length / 3;
    for (let k = 0; k < 4; k++) {
      positions.push(f.verts[k][0], f.verts[k][1], f.verts[k][2]);
      normals.push(f.n[0], f.n[1], f.n[2]);
      uvs.push(faceUV[k][0], faceUV[k][1]);
    }
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { positions, normals, uvs, indices };
}

/**
 * UV 球：segU 经线 × segV 纬度带。θ ∈ [0,π] 自 +Y 极点起，φ ∈ [0,2π)。
 * pos = (r·sinθ·cosφ, r·cosθ, r·sinθ·sinφ)，normal = pos/r，uv = (φ/2π, θ/π)。
 * 极点行会生成零面积三角形，由光栅核心的退化守卫自然跳过。
 */
export function makeSphere(radius: number, segU: number, segV: number): Mesh {
  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  const vid = (i: number, j: number): number => i * segU + (j % segU);
  for (let i = 0; i <= segV; i++) {
    const theta = (Math.PI * i) / segV;
    const st = Math.sin(theta), ct = Math.cos(theta);
    for (let j = 0; j < segU; j++) {
      const phi = (2 * Math.PI * j) / segU;
      const x = radius * st * Math.cos(phi);
      const y = radius * ct;
      const z = radius * st * Math.sin(phi);
      positions.push(x, y, z);
      normals.push(x / radius, y / radius, z / radius);
      uvs.push(j / segU, i / segV);
    }
  }
  for (let i = 0; i < segV; i++) {
    for (let j = 0; j < segU; j++) {
      const A = vid(i, j), B = vid(i + 1, j), C = vid(i + 1, j + 1), D = vid(i, j + 1);
      // (A,D,B) 与 (D,C,B)：cross(t_φ, t_θ) 指向外侧
      indices.push(A, D, B, D, C, B);
    }
  }
  return { positions, normals, uvs, indices };
}

/**
 * 环面：主环半径 R、管半径 r，主环躺在 XZ 平面（洞朝 +Y）。
 * pos = ((R + r·cos v)·cos u, r·sin v, (R + r·cos v)·sin u)
 * normal = (cos v·cos u, sin v, cos v·sin u)，uv = (u/2π, v/2π)。
 * 三角形 (A,D,B),(D,C,B) 使 cross(t_v, t_u) 指向外侧（管外表面）。
 */
export function makeTorus(R: number, r: number, segU: number, segV: number): Mesh {
  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  const vid = (i: number, j: number): number => i * segV + (j % segV);
  // u 方向 [0..segU] 共 segU+1 列（末列重复首列位置、uv u=1，闭合主环）；v 方向环绕闭合
  for (let i = 0; i <= segU; i++) {
    const u = (2 * Math.PI * i) / segU;
    const cu = Math.cos(u), su = Math.sin(u);
    for (let j = 0; j < segV; j++) {
      const v = (2 * Math.PI * j) / segV;
      const cv = Math.cos(v), sv = Math.sin(v);
      const w = R + r * cv;
      positions.push(w * cu, r * sv, w * su);
      normals.push(cv * cu, sv, cv * su);
      uvs.push(i / segU, j / segV);
    }
  }
  for (let i = 0; i < segU; i++) {
    for (let j = 0; j < segV; j++) {
      const A = vid(i, j), B = vid(i + 1, j), C = vid(i + 1, j + 1), D = vid(i, j + 1);
      indices.push(A, D, B, D, C, B);
    }
  }
  return { positions, normals, uvs, indices };
}

/** 竖条纹纹理（repeat 环绕语义）。 */
export function stripeTexture(size: number, stripes: number, a: [number, number, number], b: [number, number, number]): Texture {
  const data = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cx = Math.min(stripes - 1, Math.floor((x / size) * stripes));
      const c = cx % 2 === 0 ? a : b;
      const o = (y * size + x) * 4;
      data[o] = c[0]; data[o + 1] = c[1]; data[o + 2] = c[2]; data[o + 3] = 255;
    }
  }
  return { width: size, height: size, data };
}

// ---------- 演示场景（可无头渲染，浏览器与测试共用） ----------

export interface DemoScene {
  items: DrawItem[];
  lights: LightSetup;
  /** 推进动画：立方体自转、环面旋转、点光轨道（t 单位秒）。 */
  animate(tSec: number): void;
}

export const DEMO_TARGET = { x: 0, y: -0.3, z: 0 };
export const DEMO_FOV = (55 * Math.PI) / 180;

/** 轨道相机。 */
export function demoOrbitCamera(yaw: number, pitch: number, dist: number): Camera {
  const cp = Math.cos(pitch);
  return {
    eye: {
      x: DEMO_TARGET.x + dist * Math.sin(yaw) * cp,
      y: DEMO_TARGET.y + dist * Math.sin(pitch),
      z: DEMO_TARGET.z + dist * Math.cos(yaw) * cp,
    },
    target: DEMO_TARGET,
    up: { x: 0, y: 1, z: 0 },
    fovYRad: DEMO_FOV,
    aspect: 1,
    near: 0.1,
    far: 60,
  };
}

function compose(...ms: Mat4[]): Mat4 {
  let out = ms[0];
  for (let i = 1; i < ms.length; i++) out = mat4Mul(out, ms[i]);
  return out;
}

export function buildDemoScene(): DemoScene {
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

  const lights: LightSetup = {
    ambient: [0.12, 0.12, 0.14],
    dir: { dir: [0.45, 0.8, 0.5], color: [0.85, 0.82, 0.75] },
    point: { pos: { x: 3.2, y: 2.4, z: 0 }, color: [1.0, 0.78, 0.55], k1: 0.08, k2: 0.02 },
  };
  {
    const d = lights.dir!.dir;
    const l = Math.hypot(d[0], d[1], d[2]);
    d[0] /= l; d[1] /= l; d[2] /= l;
  }

  return {
    items: [ground, cube, sphere, torus],
    lights,
    animate(tSec: number): void {
      cube.model = compose(
        mat4Translation({ x: -1.7, y: -0.85, z: 0.2 }),
        mat4RotY(tSec * 0.5),
      );
      torus.model = compose(
        mat4Translation({ x: 1.9, y: -0.92, z: -0.1 }),
        mat4RotY(tSec * 0.7),
        mat4RotX(-0.14),
      );
      lights.point!.pos.x = 3.2 * Math.cos(tSec * 0.6);
      lights.point!.pos.y = 2.4 + 0.5 * Math.sin(tSec * 0.9);
      lights.point!.pos.z = 3.2 * Math.sin(tSec * 0.6);
    },
  };
}
