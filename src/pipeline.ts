/**
 * 渲染管线：网格 → 模型/视图/投影变换 → 近平面 Sutherland–Hodgman 裁剪 →
 * 投影到屏幕 → 背面剔除 → 透视校正插值光栅化（rasterizeTriangle 核心）。
 *
 * 顶点级冻结公式（参考实现逐位复刻）：
 *   wp  = model·p        （模型矩阵视为仿射）
 *   vp  = view·wp
 *   clip = proj·vp
 *   w = clip.w；invW = 1/w；sx = (clip.x/w + 1)*0.5*W；sy = (1 - clip.y/w)*0.5*H
 *   d = -vp.z（视距）；dOverW = d/w；attrOverW = attr/w（一律直接除法）
 *
 * 裁剪在 clip 空间进行（对 w >= CLIP_EPS 半空间做 Sutherland–Hodgman，输出 0/1/2 个
 * 三角形）；payload 在 clip 空间线性插值，投影时统一除 w —— 透视校正语义保持不变。
 *
 * 模式：textured（无光照贴图）/ gouraud（逐顶点光照）/ phong（逐像素光照，可带纹理）。
 */

import type { Framebuffer } from "./framebuffer.js";
import {
  mat4LookAt, mat4Perspective, mat4ApplyPoint, mat4ApplyDir, mat4ApplyVec4,
  type Mat4, type Vec3,
} from "./math.js";
import { rasterizeTriangle, type ScreenVert, type FragmentShader, MAX_ATTRS } from "./raster.js";
import { sample, type Texture, type SamplerKind } from "./texture.js";
import { shadePhongPixel, NO_LIGHTS, type LightSetup } from "./shading.js";
import { clipTriangleNear, type ClipVert } from "./clip.js";

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
  albedo: [number, number, number]; // 0..1
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

/** 近平面裁剪位置（clip 空间 w = CLIP_EPS），与片段深度丢弃阈值一致。 */
export const CLIP_EPS = 1e-6;

// ---- 顶点阶段 scratch（按需增长） ----
let cap = 0;
let sWorld = new Float64Array(0); // xyz
let sD = new Float64Array(0);     // 视距
let sClip = new Float64Array(0);  // xyzw
let sNrm = new Float64Array(0);   // xyz（模型变换并归一化）
let sLit = new Float64Array(0);   // gouraud 逐顶点光照色（0..255）
let sPay = new Float64Array(0);   // payload * 8（裁剪期线性插值的 varying）
const LIT_OUT = new Float64Array(3);

function ensureCapacity(n: number): void {
  if (n <= cap) return;
  cap = Math.max(n, 1024);
  sWorld = new Float64Array(cap * 3);
  sD = new Float64Array(cap);
  sClip = new Float64Array(cap * 4);
  sNrm = new Float64Array(cap * 3);
  sLit = new Float64Array(cap * 3);
  sPay = new Float64Array(cap * 8);
}

// ---- ScreenVert 池（每次最多 4 个裁剪输出顶点；rasterizeTriangle 同步消费） ----
const pool: ScreenVert[] = [0, 1, 2, 3, 4].map(() => ({
  x: 0, y: 0, invW: 0, dOverW: 0, aOverW: new Float64Array(MAX_ATTRS),
}));

// ---- ClipVert 池（每三角形 3 个输入） ----
const clipPool: ClipVert[] = [0, 1, 2].map(() => ({
  x: 0, y: 0, z: 0, w: 0, d: 0, a: new Float64Array(MAX_ATTRS),
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

function gouraudShader(): FragmentShader {
  return (_l0, _l1, _l2, _denom, _d, attrs, rgb) => {
    rgb[0] = attrs[0];
    rgb[1] = attrs[1];
    rgb[2] = attrs[2];
  };
}

function phongShader(material: Material, eye: Vec3, lights: LightSetup): FragmentShader {
  const tex = material.texture;
  const kind: SamplerKind = material.sampler ?? "nearest";
  const smp = new Float64Array(3);
  const ar0 = material.albedo[0], ag0 = material.albedo[1], ab0 = material.albedo[2];
  const spec = material.specular, shin = material.shininess;
  return (_l0, _l1, _l2, _denom, _d, attrs, rgb) => {
    // 插值后的法线不再是单位长度，逐像素归一化（sqrt(x²+y²+z²)，0 时置 1）
    let nx = attrs[3], ny = attrs[4], nz = attrs[5];
    let nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (nl === 0) nl = 1;
    nx /= nl; ny /= nl; nz /= nl;
    let ar = ar0, ag = ag0, ab = ab0;
    if (tex !== undefined) {
      sample(tex, kind, attrs[6], attrs[7], smp);
      ar = (smp[0] / 255) * ar0;
      ag = (smp[1] / 255) * ag0;
      ab = (smp[2] / 255) * ab0;
    }
    shadePhongPixel(nx, ny, nz, attrs[0], attrs[1], attrs[2], eye, lights, ar, ag, ab, spec, shin, rgb);
  };
}

export class Renderer {
  constructor(private readonly fb: Framebuffer) {}

  render(camera: Camera, items: DrawItem[], mode: ShadingMode, lights: LightSetup = NO_LIGHTS): void {
    const { fb } = this;
    const view = mat4LookAt(camera.eye, camera.target, camera.up);
    const proj = mat4Perspective(camera.fovYRad, camera.aspect, camera.near, camera.far);

    for (const item of items) {
      const { mesh, model, material } = item;
      const vc = mesh.positions.length / 3;
      ensureCapacity(vc);
      const hasTex = material.texture !== undefined;

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
        let nl = Math.hypot(wn.x, wn.y, wn.z);
        if (nl === 0) nl = 1;
        sNrm[i * 3] = wn.x / nl; sNrm[i * 3 + 1] = wn.y / nl; sNrm[i * 3 + 2] = wn.z / nl;

        // payload：裁剪期线性插值的 varying（投影时统一除 w）
        const o = i * 8;
        if (mode === "textured") {
          sPay[o] = mesh.uvs[i * 2];
          sPay[o + 1] = mesh.uvs[i * 2 + 1];
        } else if (mode === "gouraud") {
          shadePhongPixel(
            sNrm[i * 3], sNrm[i * 3 + 1], sNrm[i * 3 + 2],
            wp.x, wp.y, wp.z,
            camera.eye, lights,
            material.albedo[0], material.albedo[1], material.albedo[2],
            material.specular, material.shininess,
            LIT_OUT,
          );
          sLit[i * 3] = LIT_OUT[0]; sLit[i * 3 + 1] = LIT_OUT[1]; sLit[i * 3 + 2] = LIT_OUT[2];
          sPay[o] = LIT_OUT[0]; sPay[o + 1] = LIT_OUT[1]; sPay[o + 2] = LIT_OUT[2];
        } else {
          sPay[o] = wp.x; sPay[o + 1] = wp.y; sPay[o + 2] = wp.z;
          sPay[o + 3] = sNrm[i * 3]; sPay[o + 4] = sNrm[i * 3 + 1]; sPay[o + 5] = sNrm[i * 3 + 2];
          if (hasTex) {
            sPay[o + 6] = mesh.uvs[i * 2];
            sPay[o + 7] = mesh.uvs[i * 2 + 1];
          }
        }
      }

      // ---- 着色器与属性布局 ----
      let attrCount = 0;
      let shade: FragmentShader;
      if (mode === "textured") {
        attrCount = 2;
        shade = texturedShader(material);
      } else if (mode === "gouraud") {
        attrCount = 3;
        shade = gouraudShader();
      } else {
        attrCount = hasTex ? 8 : 6;
        shade = phongShader(material, camera.eye, lights);
      }

      // ---- 三角形阶段：裁剪 → 扇形化 → 投影 → 剔除 → 光栅化 ----
      const tc = mesh.indices.length / 3;
      for (let t = 0; t < tc; t++) {
        const ids = [mesh.indices[t * 3], mesh.indices[t * 3 + 1], mesh.indices[t * 3 + 2]];
        for (let k = 0; k < 3; k++) {
          const i = ids[k];
          const cv = clipPool[k];
          cv.x = sClip[i * 4]; cv.y = sClip[i * 4 + 1]; cv.z = sClip[i * 4 + 2]; cv.w = sClip[i * 4 + 3];
          cv.d = sD[i];
          for (let j = 0; j < attrCount; j++) cv.a[j] = sPay[i * 8 + j];
        }
        const poly = clipTriangleNear(clipPool[0], clipPool[1], clipPool[2], CLIP_EPS);
        const n = poly.length;
        if (n < 3) continue;

        // 投影（冻结公式）
        for (let k = 0; k < n; k++) {
          const cv = poly[k];
          const w = cv.w;
          const invW = 1 / w;
          const sv = pool[k];
          sv.x = (cv.x / w + 1) * 0.5 * fb.width;
          sv.y = (1 - cv.y / w) * 0.5 * fb.height;
          sv.invW = invW;
          sv.dOverW = cv.d / w;
          const a = sv.aOverW;
          for (let j = 0; j < attrCount; j++) a[j] = cv.a[j] / w;
        }

        // 扇形化 + 背面剔除 + 光栅化
        for (let i = 1; i + 1 < n; i++) {
          const p0 = pool[0], p1 = pool[i], p2 = pool[i + 1];
          // 与 rasterizeTriangle 同一 area2 公式。
          // 约定推导：视空间（y 上）CCW 的前向面，经投影 y 翻转后在 y-down 屏幕坐标中
          // area2 < 0，因此前向面 = area2 < 0，剔除 area2 > 0（背面）。
          const area2 = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
          if (!(Math.abs(area2) > 1e-12)) continue;
          if (area2 > 0 && !material.doubleSided) continue;
          rasterizeTriangle(fb, p0, p1, p2, attrCount, { depthTest: true, nearDist: CLIP_EPS }, shade);
        }
      }
    }
  }
}
