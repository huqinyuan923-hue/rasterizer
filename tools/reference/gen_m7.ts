/**
 * M7 参考实现：近平面 Sutherland–Hodgman 裁剪（多边形列表版，独立编码）。
 * 场景：完全在前 / 单顶点穿近面（输出 2 三角形）/ 双顶点穿近面（输出 1 三角形）/
 * 完全在相机后（输出 0）/ 侧面出界（屏幕 guard）。
 * 裁剪是被测对象；光栅核心（rasterizeTriangle）与数学库是已冻结夹具。
 * 平色光照走 gouraud 通道：ambient=(1,1,1) 且无光源 ⟹ 顶点色 = albedo*255。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import {
  mat4LookAt, mat4Perspective, mat4ApplyPoint, mat4ApplyVec4, mat4Identity,
  type Vec3,
} from "../../src/math.js";
import { rasterizeTriangle, type ScreenVert } from "../../src/raster.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";

export const M7_W = 128;
export const M7_H = 128;

export const M7_CAMERA = {
  eye: { x: 0, y: 0, z: 0 } as Vec3,
  target: { x: 0, y: 0, z: -1 } as Vec3,
  up: { x: 0, y: 1, z: 0 } as Vec3,
  fovYRad: (70 * Math.PI) / 180,
  aspect: 1,
  near: 0.1,
  far: 20,
};

export const M7_LIGHTS = {
  ambient: [1, 1, 1] as [number, number, number],
  dir: null,
  point: null,
};

export interface M7Tri {
  /** 视空间顶点（本场景 model = identity ⟹ 世界 = 视空间）。 */
  p: [Vec3, Vec3, Vec3];
  c: [number, number, number];
}

/** 黄金场景（fov 70°，near 0.1：z=-1.5 处可见 |x|,|y| ≈ 1.05）。 */
export const M7_TRIANGLES: M7Tri[] = [
  // 红：完全在前、四角出屏（屏幕边界 guard）
  { p: [{ x: -1.3, y: -0.9, z: -1.5 }, { x: 1.5, y: -0.9, z: -1.5 }, { x: 0.1, y: 1.2, z: -1.5 }], c: [210, 60, 60] },
  // 绿：单顶点在相机后 ⟹ 裁剪输出 2 个三角形
  { p: [{ x: -0.5, y: 0.2, z: 0.4 }, { x: 0.6, y: -0.45, z: -2 }, { x: -0.6, y: -0.45, z: -2 }], c: [60, 200, 90] },
  // 黄：双顶点在相机后 ⟹ 裁剪输出 1 个三角形
  { p: [{ x: 0.05, y: 0.55, z: 0.6 }, { x: -0.9, y: -0.5, z: 0.6 }, { x: 0.35, y: -0.15, z: -2.5 }], c: [230, 210, 70] },
];

const VIEW = mat4LookAt(M7_CAMERA.eye, M7_CAMERA.target, M7_CAMERA.up);
const PROJ = mat4Perspective(M7_CAMERA.fovYRad, M7_CAMERA.aspect, M7_CAMERA.near, M7_CAMERA.far);
const ID = mat4Identity();

// ---- 参考裁剪（多边形列表版 S–H；lerp 公式与正式实现同源冻结） ----

interface RVert { cx: number; cy: number; cz: number; cw: number; d: number; payload: Float64Array }

function refLerp(a: RVert, b: RVert, eps: number): RVert {
  const t = (a.cw - eps) / (a.cw - b.cw);
  const L = (x: number, y: number): number => x + (y - x) * t;
  const n = a.payload.length;
  const payload = new Float64Array(n);
  for (let k = 0; k < n; k++) payload[k] = L(a.payload[k], b.payload[k]);
  return { cx: L(a.cx, b.cx), cy: L(a.cy, b.cy), cz: L(a.cz, b.cz), cw: L(a.cw, b.cw), d: L(a.d, b.d), payload };
}

export function refClipNear(poly: RVert[], eps: number): RVert[] {
  if (poly.every((v) => v.cw >= eps)) return poly;
  if (poly.every((v) => v.cw < eps)) return [];
  const out: RVert[] = [];
  for (let i = 0; i < poly.length; i++) {
    const cur = poly[i], nxt = poly[(i + 1) % poly.length];
    const cin = cur.cw >= eps, nin = nxt.cw >= eps;
    if (cin) out.push(cur);
    if (cin !== nin) out.push(refLerp(cur, nxt, eps));
  }
  return out;
}

function refProject(v: RVert): ScreenVert {
  const w = v.cw;
  const sv: ScreenVert = { x: 0, y: 0, invW: 0, dOverW: 0, aOverW: new Float64Array(8) };
  sv.x = (v.cx / w + 1) * 0.5 * M7_W;
  sv.y = (1 - v.cy / w) * 0.5 * M7_H;
  sv.invW = 1 / w;
  sv.dOverW = v.d / w;
  for (let k = 0; k < v.payload.length; k++) sv.aOverW[k] = v.payload[k] / w;
  return sv;
}

function buildRVert(p: Vec3, color: [number, number, number]): RVert {
  const wp = mat4ApplyPoint(ID, p);
  const vp = mat4ApplyPoint(VIEW, wp);
  const clip = mat4ApplyVec4(PROJ, { x: vp.x, y: vp.y, z: vp.z, w: 1 });
  return {
    cx: clip.x, cy: clip.y, cz: clip.z, cw: clip.w,
    d: -vp.z,
    payload: Float64Array.from([color[0], color[1], color[2]]),
  };
}

export function renderM7(): RawImage {
  const fb = new Framebuffer(M7_W, M7_H);
  for (const tri of M7_TRIANGLES) {
    const poly = refClipNear(tri.p.map((p) => buildRVert(p, tri.c)), 1e-6);
    if (poly.length < 3) continue;
    const flat = (_l0: number, _l1: number, _l2: number, _den: number, _d: number, attrs: Float64Array, rgb: Float64Array): void => {
      rgb[0] = attrs[0]; rgb[1] = attrs[1]; rgb[2] = attrs[2];
    };
    for (let i = 1; i + 1 < poly.length; i++) {
      const a = refProject(poly[0]);
      const b = refProject(poly[i]);
      const c = refProject(poly[i + 1]);
      rasterizeTriangle(fb, a, b, c, 3, { depthTest: true, nearDist: 1e-6 }, flat);
    }
  }
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("m7_clip.ppm", renderM7());
  console.log("m7_clip.ppm written");
}
