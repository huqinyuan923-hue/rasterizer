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
  mat4LookAt, mat4Perspective, mat4Ortho,
  type Mat4, type Vec3,
} from "./math.js";
import { rasterizeShaded, type ScreenVert, type ShaderSpec, MAX_ATTRS } from "./raster.js";
import type { Texture, SamplerKind } from "./texture.js";
import { shadePhongPixel, NO_LIGHTS, type LightSetup } from "./shading.js";
import { clipTriangleNear, type ClipVert } from "./clip.js";
import { meshBoundingSphere, transformSphere, sphereOutsidePlanes, frustumPlanes } from "./culling.js";

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
  /** 投影类型（缺省 perspective）。ortho 时用 left/right/bottom/top 定义视景体（near/far 沿用）。 */
  projection?: "perspective" | "ortho";
  left?: number;
  right?: number;
  bottom?: number;
  top?: number;
}

export interface Material {
  albedo: [number, number, number]; // 0..1
  specular: number;                 // 高光强度 0..1
  shininess: number;
  /** 着色模型：lambert（无高光）/ blinnPhong（默认，冻结缺省与一期行为一致）。 */
  shadingModel?: "lambert" | "blinnPhong";
  texture?: Texture;
  sampler?: SamplerKind;            // 默认 nearest
  /** mip 一档开关：true 且 texture.mip 存在时启用逐三角形 mip 选层（缺省 false）。 */
  mip?: boolean;
  /** 背面剔除开关：true（默认）剔除背面；false 双面渲染。语义与一期 doubleSided 相反。 */
  cullBackface?: boolean;
  /** 一期遗留别名：doubleSided === !cullBackface。仅作迁移兼容，勿在新代码使用。 */
  doubleSided?: boolean;
}

export interface DrawItem {
  mesh: Mesh;
  model: Mat4;
  material: Material;
}

export type ShadingMode = "textured" | "gouraud" | "phong";

/** 渲染选项。cullFrustum（默认 true）：5 平面视锥剔除 —— 只跳过 0 像素项，像素安全。
 * sortOpaque（默认 false）：按包围球中心到相机距离的**平方**升序稳定排序
 * （不透明 front-to-back，减少 overdraw；平局保持原序）。开深度后像素不变。 */
export interface RenderOptions {
  cullFrustum?: boolean;
  sortOpaque?: boolean;
}

/** 包围球缓存（mesh 对象键；同一 mesh 每帧不重扫顶点）。 */
const sphereCache = new WeakMap<Mesh, ReturnType<typeof meshBoundingSphere>>();

/** 近平面裁剪位置（clip 空间 w = CLIP_EPS），与片段深度丢弃阈值一致。 */
export const CLIP_EPS = 1e-6;

/**
 * 冻结的 mip 选层启发式（导出供测试锚定）：
 * texels-per-pixel = (uv 二倍面积 × texW × texH) / 屏幕二倍面积，
 * ≥ 1（缩小：纹理细节比像素更细）→ 用 mip1；否则 mip0。
 */
export function mipSelect(uvArea2: number, texW: number, texH: number, screenArea2: number): boolean {
  return (uvArea2 * texW * texH) / screenArea2 >= 1;
}

// ---- 顶点阶段 scratch（按需增长） ----
let cap = 0;
let sD = new Float64Array(0);     // 视距
let sClip = new Float64Array(0);  // xyzw
let sNrm = new Float64Array(0);   // xyz（模型变换并归一化）
let sLit = new Float64Array(0);   // gouraud 逐顶点光照色（0..255）
let sPay = new Float64Array(0);   // payload * 8（裁剪期线性插值的 varying）
const LIT_OUT = new Float64Array(3);

function ensureCapacity(n: number): void {
  if (n <= cap) return;
  cap = Math.max(n, 1024);
  sD = new Float64Array(cap);
  sClip = new Float64Array(cap * 4);
  sNrm = new Float64Array(cap * 3);
  sLit = new Float64Array(cap * 3);
  sPay = new Float64Array(cap * 8);
}

// ---- ScreenVert 池（每次最多 4 个裁剪输出顶点；rasterizeShaded 同步消费） ----
const pool: ScreenVert[] = [0, 1, 2, 3, 4].map(() => ({
  x: 0, y: 0, invW: 0, dOverW: 0, aOverW: new Float64Array(MAX_ATTRS),
}));

// ---- ClipVert 池（每三角形 3 个输入） ----
const clipPool: ClipVert[] = [0, 1, 2].map(() => ({
  x: 0, y: 0, z: 0, w: 0, d: 0, a: new Float64Array(MAX_ATTRS),
}));

const TRI_IDS = new Int32Array(3);

export class Renderer {
  constructor(private readonly fb: Framebuffer) {}

  render(camera: Camera, items: DrawItem[], mode: ShadingMode, lights: LightSetup = NO_LIGHTS, opts: RenderOptions = {}): void {
    const { fb } = this;
    const view = mat4LookAt(camera.eye, camera.target, camera.up);
    const proj = camera.projection === "ortho"
      ? mat4Ortho(
          camera.left ?? -1, camera.right ?? 1, camera.bottom ?? -1, camera.top ?? 1,
          camera.near, camera.far,
        )
      : mat4Perspective(camera.fovYRad, camera.aspect, camera.near, camera.far);

    // 视锥剔除（默认开）：5 平面（近平面 + 4 侧面）—— 只跳过产出 0 像素的项，像素安全。
    const planes = opts.cullFrustum === false ? null : frustumPlanes(view, proj);

    // 不透明排序（opt-in）：包围球中心到相机距离平方升序、稳定（平局保原序）。
    let drawList = items;
    if (opts.sortOpaque === true && items.length > 1) {
      const ex = camera.eye.x, ey = camera.eye.y, ez = camera.eye.z;
      const keyed = items.map((item, order) => {
        let s = sphereCache.get(item.mesh);
        if (s === undefined) {
          s = meshBoundingSphere(item.mesh);
          sphereCache.set(item.mesh, s);
        }
        const ws = transformSphere(s, item.model);
        const dx = ws.cx - ex, dy = ws.cy - ey, dz = ws.cz - ez;
        return { item, order, distSq: dx * dx + dy * dy + dz * dz };
      });
      keyed.sort((a, b) => a.distSq - b.distSq || a.order - b.order);
      drawList = keyed.map((k) => k.item);
    }

    // 聚光轴预归一化（每次 render 一次；冻结 sqrt 公式；零轴 → NaN，片元侧恒不命中锥内）
    let spotsNorm: Float64Array | undefined;
    if (lights.spots.length > 0) {
      spotsNorm = new Float64Array(lights.spots.length * 3);
      for (let i = 0; i < lights.spots.length; i++) {
        const d = lights.spots[i].dir;
        const dl = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (dl > 0) {
          spotsNorm[i * 3] = d[0] / dl;
          spotsNorm[i * 3 + 1] = d[1] / dl;
          spotsNorm[i * 3 + 2] = d[2] / dl;
        } else {
          spotsNorm[i * 3] = NaN; spotsNorm[i * 3 + 1] = NaN; spotsNorm[i * 3 + 2] = NaN;
        }
      }
    }

    for (const item of drawList) {
      if (planes !== null) {
        let s = sphereCache.get(item.mesh);
        if (s === undefined) {
          s = meshBoundingSphere(item.mesh);
          sphereCache.set(item.mesh, s);
        }
        if (sphereOutsidePlanes(transformSphere(s, item.model), planes)) continue;
      }
      const { mesh, model, material } = item;
    const vc = mesh.positions.length / 3;
    ensureCapacity(vc);
    const hasTex = material.texture !== undefined;
    // 材质开关冻结缺省：blinnPhong + 背面剔除。cullBackface 显式优先；doubleSided 仅作缺省回退。
    const specOn = (material.shadingModel ?? "blinnPhong") === "blinnPhong";
    const cullBack = material.cullBackface ?? (material.doubleSided !== undefined ? !material.doubleSided : true);
    const useMip = material.mip === true && material.texture?.mip !== undefined;
    // mip 只在 payload 含 uv 的模式生效（textured / phong+tex）；gouraud 无 uv。
    const mipActive = useMip && (mode === "textured" || (mode === "phong" && hasTex));
    const uvOff = mode === "textured" ? 0 : 6; // payload 中 uv 的起始下标（textured:0, phong+tex:6）

      // ---- 顶点阶段(冻结公式的无分配内联:同一操作数、同一运算顺序,逐位一致) ----
      // 矩阵元素提升为局部变量(Float64Array 下标读 → 寄存器;值逐位相同)
      const m00 = model[0], m01 = model[4], m02 = model[8], m03 = model[12];
      const m10 = model[1], m11 = model[5], m12 = model[9], m13 = model[13];
      const m20 = model[2], m21 = model[6], m22 = model[10], m23 = model[14];
      const m30 = model[3], m31 = model[7], m32 = model[11], m33 = model[15];
      const v00 = view[0], v01 = view[4], v02 = view[8], v03 = view[12];
      const v10 = view[1], v11 = view[5], v12 = view[9], v13 = view[13];
      const v20 = view[2], v21 = view[6], v22 = view[10], v23 = view[14];
      const v30 = view[3], v31 = view[7], v32 = view[11], v33 = view[15];
      const p00 = proj[0], p01 = proj[4], p02 = proj[8], p03 = proj[12];
      const p10 = proj[1], p11 = proj[5], p12 = proj[9], p13 = proj[13];
      const p20 = proj[2], p21 = proj[6], p22 = proj[10], p23 = proj[14];
      const p30 = proj[3], p31 = proj[7], p32 = proj[11], p33 = proj[15];
      for (let i = 0; i < vc; i++) {
        const px = mesh.positions[i * 3], py = mesh.positions[i * 3 + 1], pz = mesh.positions[i * 3 + 2];
        // wp = mat4ApplyPoint(model, p) —— applyVec4(w=1) 后除 w
        const wpx = (m00 * px + m01 * py + m02 * pz + m03) / (m30 * px + m31 * py + m32 * pz + m33);
        const wpy = (m10 * px + m11 * py + m12 * pz + m13) / (m30 * px + m31 * py + m32 * pz + m33);
        const wpz = (m20 * px + m21 * py + m22 * pz + m23) / (m30 * px + m31 * py + m32 * pz + m33);
        // vp = mat4ApplyPoint(view, wp)
        const vpx = (v00 * wpx + v01 * wpy + v02 * wpz + v03) / (v30 * wpx + v31 * wpy + v32 * wpz + v33);
        const vpy = (v10 * wpx + v11 * wpy + v12 * wpz + v13) / (v30 * wpx + v31 * wpy + v32 * wpz + v33);
        const vpz = (v20 * wpx + v21 * wpy + v22 * wpz + v23) / (v30 * wpx + v31 * wpy + v32 * wpz + v33);
        // clip = mat4ApplyVec4(proj, (vp, 1))
        const clipX = p00 * vpx + p01 * vpy + p02 * vpz + p03;
        const clipY = p10 * vpx + p11 * vpy + p12 * vpz + p13;
        const clipZ = p20 * vpx + p21 * vpy + p22 * vpz + p23;
        const clipW = p30 * vpx + p31 * vpy + p32 * vpz + p33;
        sD[i] = -vpz;
        sClip[i * 4] = clipX; sClip[i * 4 + 1] = clipY; sClip[i * 4 + 2] = clipZ; sClip[i * 4 + 3] = clipW;
        const nx = mesh.normals[i * 3], ny = mesh.normals[i * 3 + 1], nz = mesh.normals[i * 3 + 2];
        // wn = mat4ApplyDir(model, n)
        const wnx = m00 * nx + m01 * ny + m02 * nz;
        const wny = m10 * nx + m11 * ny + m12 * nz;
        const wnz = m20 * nx + m21 * ny + m22 * nz;
        let nl = Math.hypot(wnx, wny, wnz);
        if (nl === 0) nl = 1;
        sNrm[i * 3] = wnx / nl; sNrm[i * 3 + 1] = wny / nl; sNrm[i * 3 + 2] = wnz / nl;

        // payload：裁剪期线性插值的 varying（投影时统一除 w）
        const o = i * 8;
        if (mode === "textured") {
          sPay[o] = mesh.uvs[i * 2];
          sPay[o + 1] = mesh.uvs[i * 2 + 1];
        } else if (mode === "gouraud") {
          shadePhongPixel(
            sNrm[i * 3], sNrm[i * 3 + 1], sNrm[i * 3 + 2],
            wpx, wpy, wpz,
            camera.eye, lights,
            material.albedo[0], material.albedo[1], material.albedo[2],
            material.specular, material.shininess,
            LIT_OUT, specOn,
          );
          sLit[i * 3] = LIT_OUT[0]; sLit[i * 3 + 1] = LIT_OUT[1]; sLit[i * 3 + 2] = LIT_OUT[2];
          sPay[o] = LIT_OUT[0]; sPay[o + 1] = LIT_OUT[1]; sPay[o + 2] = LIT_OUT[2];
        } else {
          sPay[o] = wpx; sPay[o + 1] = wpy; sPay[o + 2] = wpz;
          sPay[o + 3] = sNrm[i * 3]; sPay[o + 4] = sNrm[i * 3 + 1]; sPay[o + 5] = sNrm[i * 3 + 2];
          if (hasTex) {
            sPay[o + 6] = mesh.uvs[i * 2];
            sPay[o + 7] = mesh.uvs[i * 2 + 1];
          }
        }
      }

      // ---- 着色专用快速路径的着色描述 ----
      let attrCount = 0;
      const shadeSpec: ShaderSpec = {
        kind: mode === "textured" ? "textured" : mode === "gouraud" ? "gouraud" : "phong",
        texture: material.texture,
        mipTex: useMip ? material.texture!.mip : undefined,
        bilinear: (material.sampler ?? "nearest") === "bilinear",
        eye: camera.eye,
        lights,
        spotsNorm,
        ar: material.albedo[0], ag: material.albedo[1], ab: material.albedo[2],
        specular: material.specular,
        shininess: material.shininess,
        specularOn: specOn,
      };
      if (mode === "textured") {
        attrCount = 2;
      } else if (mode === "gouraud") {
        attrCount = 3;
      } else {
        attrCount = hasTex ? 8 : 6;
      }

      // ---- 三角形阶段：裁剪 → 扇形化 → 投影 → 剔除 → 光栅化 ----
      const tc = mesh.indices.length / 3;
      for (let t = 0; t < tc; t++) {
        TRI_IDS[0] = mesh.indices[t * 3];
        TRI_IDS[1] = mesh.indices[t * 3 + 1];
        TRI_IDS[2] = mesh.indices[t * 3 + 2];
        for (let k = 0; k < 3; k++) {
          const i = TRI_IDS[k];
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

        // 扇形化 + 背面剔除 + mip 选层 + 光栅化
        for (let i = 1; i + 1 < n; i++) {
          const p0 = pool[0], p1 = pool[i], p2 = pool[i + 1];
          // 与 rasterizeTriangle 同一 area2 公式。
          // 约定推导：视空间（y 上）CCW 的前向面，经投影 y 翻转后在 y-down 屏幕坐标中
          // area2 < 0，因此前向面 = area2 < 0，剔除 area2 > 0（背面）。
          const area2 = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
          if (!(Math.abs(area2) > 1e-12)) continue;
          if (area2 > 0 && cullBack) continue;

          // mip 一档选层（逐子三角形，冻结启发式）：
          // texels-per-pixel = (uv 面积 × texW × texH) / 屏幕面积（均为"二倍面积"，比值不变），
          // ≥ 1（纹理细节比像素更细，缩小）→ 用半分辨率 mip1，否则 mip0。
          // uv 面积在 clip 空间取 payload（裁剪期线性），屏幕面积取投影后顶点 —— 定义冻结。
          if (mipActive) {
            const c0 = poly[0], c1 = poly[i], c2 = poly[i + 1];
            const u0 = c0.a[uvOff], v0 = c0.a[uvOff + 1];
            const u1 = c1.a[uvOff], v1 = c1.a[uvOff + 1];
            const u2 = c2.a[uvOff], v2 = c2.a[uvOff + 1];
            const uvArea2 = Math.abs((u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0));
            shadeSpec.texture = mipSelect(uvArea2, material.texture!.width, material.texture!.height, Math.abs(area2))
              ? shadeSpec.mipTex!
              : material.texture!;
          }

          rasterizeShaded(fb, p0, p1, p2, { depthTest: true, nearDist: CLIP_EPS }, shadeSpec);
        }
      }
    }
  }
}
