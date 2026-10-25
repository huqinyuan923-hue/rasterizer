/**
 * 渲染管线：网格 → 模型/视图/投影变换 → （M7 起近平面裁剪）→ 投影到屏幕 →
 * 透视校正插值光栅化（走 rasterizeTriangle 核心）。
 *
 * 顶点级冻结公式（参考实现逐位复刻）：
 *   wp  = model·p        （模型矩阵视为仿射）
 *   vp  = view·wp
 *   clip = proj·vp
 *   w = clip.w；invW = 1/w；sx = (clip.x/w + 1)*0.5*W；sy = (1 - clip.y/w)*0.5*H
 *   d = -vp.z（视距）；dOverW = d/w；attrOverW = attr/w（一律直接除法）
 *
 * 透视校正插值由核心完成：attr = (Σλᵢ attrᵢ/wᵢ) / (Σλᵢ 1/wᵢ) —— 一次插值同时得到
 * 正确的 uv 与深度。M5 提供 textured（无光照贴图）模式；M6 增加 gouraud / phong。
 */

import type { Framebuffer } from "./framebuffer.js";
import {
  mat4LookAt, mat4Perspective, mat4ApplyPoint, mat4ApplyDir, mat4ApplyVec4,
  type Mat4, type Vec3,
} from "./math.js";
import { rasterizeTriangle, type ScreenVert, type FragmentShader, MAX_ATTRS } from "./raster.js";
import { sample, type Texture, type SamplerKind } from "./texture.js";

export interface Mesh {
  positions: number[]; // xyz*vc
  normals: number[];   // xyz*vc
  uvs: number[];       // uv*vc
  indices: number[];   // 三角形索引（长度 3*tc）
}

export interface Camera {
  eye: Vec3;
  target: Vec3;
  up: Vec3;
  fovYRad: number;
  aspect: number;
  near: number;
  far: number;
}

export interface Material {
  albedo: [number, number, number]; // 0..1，M6 起参与光照
  specular: number;                 // 高光强度 0..1
  shininess: number;
  texture?: Texture;
  sampler?: SamplerKind;            // 默认 nearest
  doubleSided?: boolean;
}

export interface DrawItem {
  mesh: Mesh;
  model: Mat4;
  material: Material;
}

export type ShadingMode = "textured" | "gouraud" | "phong";

/** 近平面 guard（M7 将替换为 Sutherland–Hodgman 裁剪，常数保留）。 */
export const CLIP_EPS = 1e-6;

// ---- 顶点阶段 scratch（按需增长） ----
let cap = 0;
let sWorld = new Float64Array(0);
let sD = new Float64Array(0);
let sClip = new Float64Array(0);
let sNrm = new Float64Array(0);

function ensureCapacity(n: number): void {
  if (n <= cap) return;
  cap = Math.max(n, 1024);
  sWorld = new Float64Array(cap * 3);
  sD = new Float64Array(cap);
  sClip = new Float64Array(cap * 4);
  sNrm = new Float64Array(cap * 3);
}

// ---- ScreenVert 池（每三角形复用，rasterizeTriangle 同步消费） ----
const pool: ScreenVert[] = [0, 1, 2].map(() => ({
  x: 0, y: 0, invW: 0, dOverW: 0, aOverW: new Float64Array(MAX_ATTRS),
}));

function texturedShader(material: Material): FragmentShader {
  const texture = material.texture!;
  const kind: SamplerKind = material.sampler ?? "nearest";
  const out = new Float64Array(3);
  return (_l0, _l1, _l2, _denom, _d, attrs, rgb) => {
    sample(texture, kind, attrs[0], attrs[1], out);
    rgb[0] = out[0];
    rgb[1] = out[1];
    rgb[2] = out[2];
  };
}

export class Renderer {
  constructor(private readonly fb: Framebuffer) {}

  render(camera: Camera, items: DrawItem[], mode: ShadingMode): void {
    const { fb } = this;
    const view = mat4LookAt(camera.eye, camera.target, camera.up);
    const proj = mat4Perspective(camera.fovYRad, camera.aspect, camera.near, camera.far);

    for (const item of items) {
      const { mesh, model, material } = item;
      const vc = mesh.positions.length / 3;
      ensureCapacity(vc);

      // ---- 顶点阶段 ----
      for (let i = 0; i < vc; i++) {
        const px = mesh.positions[i * 3], py = mesh.positions[i * 3 + 1], pz = mesh.positions[i * 3 + 2];
        const wp = mat4ApplyPoint(model, { x: px, y: py, z: pz });
        const vp = mat4ApplyPoint(view, wp);
        const clip = mat4ApplyVec4(proj, { x: vp.x, y: vp.y, z: vp.z, w: 1 });
        sWorld[i * 3] = wp.x; sWorld[i * 3 + 1] = wp.y; sWorld[i * 3 + 2] = wp.z;
        sD[i] = -vp.z;
        sClip[i * 4] = clip.x; sClip[i * 4 + 1] = clip.y; sClip[i * 4 + 2] = clip.z; sClip[i * 4 + 3] = clip.w;
        const nx = mesh.normals[i * 3], ny = mesh.normals[i * 3 + 1], nz = mesh.normals[i * 3 + 2];
        const wn = mat4ApplyDir(model, { x: nx, y: ny, z: nz });
        const nl = Math.hypot(wn.x, wn.y, wn.z) || 1;
        sNrm[i * 3] = wn.x / nl; sNrm[i * 3 + 1] = wn.y / nl; sNrm[i * 3 + 2] = wn.z / nl;
      }

      // ---- 着色器与属性布局 ----
      let attrCount = 0;
      let shade: FragmentShader;
      if (mode === "textured") {
        attrCount = 2;
        shade = texturedShader(material);
      } else {
        throw new Error(`mode ${mode} arrives in M6`);
      }

      // ---- 三角形阶段 ----
      const tc = mesh.indices.length / 3;
      for (let t = 0; t < tc; t++) {
        const i0 = mesh.indices[t * 3], i1 = mesh.indices[t * 3 + 1], i2 = mesh.indices[t * 3 + 2];
        const w0 = sClip[i0 * 4 + 3], w1 = sClip[i1 * 4 + 3], w2 = sClip[i2 * 4 + 3];
        // M5/M6 近平面 guard：任一顶点在近平面后即放弃（M7 换成真裁剪）
        if (w0 < CLIP_EPS || w1 < CLIP_EPS || w2 < CLIP_EPS) continue;

        const ids = [i0, i1, i2];
        for (let k = 0; k < 3; k++) {
          const i = ids[k];
          const w = sClip[i * 4 + 3];
          const invW = 1 / w;
          const sv = pool[k];
          sv.x = (sClip[i * 4] / w + 1) * 0.5 * fb.width;
          sv.y = (1 - sClip[i * 4 + 1] / w) * 0.5 * fb.height;
          sv.invW = invW;
          sv.dOverW = sD[i] / w;
          const a = sv.aOverW;
          if (mode === "textured") {
            a[0] = mesh.uvs[i * 2] / w;
            a[1] = mesh.uvs[i * 2 + 1] / w;
          }
        }

        // 背面剔除：归一化前的原始有向面积（与 rasterizeTriangle 同一公式）
        const p0 = pool[0], p1 = pool[1], p2 = pool[2];
        const area2 = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
        if (!(Math.abs(area2) > 1e-12)) continue;
        if (area2 < 0 && !material.doubleSided) continue;

        rasterizeTriangle(fb, p0, p1, p2, attrCount, { depthTest: true, nearDist: CLIP_EPS }, shade);
      }
    }
  }
}
