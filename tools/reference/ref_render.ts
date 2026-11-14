/**
 * 参考渲染器共享副本(供二期多个 gen 使用):独立编码的顶点阶段 + 逐像素
 * 重心插值 + Blinn-Phong/Lambert 着色 + 可选背面剔除。与 src/ 的冻结公式同源,
 * 写法独立(黄金对拍纪律)。数学库 / Framebuffer 是夹具。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import {
  mat4LookAt, mat4Perspective, mat4ApplyPoint, mat4ApplyDir, mat4ApplyVec4, type Mat4, type Vec3,
} from "../../src/math.js";
import type { Mesh } from "../../src/pipeline.js";
import type { Texture } from "../../src/texture.js";

export interface RefCamera {
  eye: Vec3;
  target: Vec3;
  up: Vec3;
  fovYRad: number;
  aspect: number;
  near: number;
  far: number;
  /** 正交投影（缺省透视）：参考实现自带 mat4Ortho 公式副本。 */
  ortho?: { left: number; right: number; bottom: number; top: number };
}

/** 参考正交投影矩阵（列主序,冻结公式副本）。 */
function refOrtho(l: number, r: number, b: number, t: number, n: number, f: number): Mat4 {
  const m = new Float64Array(16);
  m[0] = 2 / (r - l);
  m[5] = 2 / (t - b);
  m[10] = -2 / (f - n);
  m[12] = -(r + l) / (r - l);
  m[13] = -(t + b) / (t - b);
  m[14] = -(f + n) / (f - n);
  m[15] = 1;
  return m;
}

export interface RefLights {
  ambient: [number, number, number];
  dirs: Array<{ dir: [number, number, number]; color: [number, number, number] }>;
  points: Array<{ pos: Vec3; color: [number, number, number]; k1: number; k2: number }>;
}

export interface RefMaterial {
  albedo: [number, number, number];
  specular: number;
  shininess: number;
  lambert?: boolean;
  texture?: Texture;
  doubleSided?: boolean;
}

interface RefVert { sx: number; sy: number; invW: number; dOverW: number; payload: Float64Array }

function buildVerts(mesh: Mesh, model: Mat4, cam: RefCamera, W: number, H: number): RefVert[] {
  const view = mat4LookAt(cam.eye, cam.target, cam.up);
  const proj = cam.ortho !== undefined
    ? refOrtho(cam.ortho.left, cam.ortho.right, cam.ortho.bottom, cam.ortho.top, cam.near, cam.far)
    : mat4Perspective(cam.fovYRad, cam.aspect, cam.near, cam.far);
  const vc = mesh.positions.length / 3;
  const out: RefVert[] = [];
  for (let i = 0; i < vc; i++) {
    const p = { x: mesh.positions[i * 3], y: mesh.positions[i * 3 + 1], z: mesh.positions[i * 3 + 2] };
    const wp = mat4ApplyPoint(model, p);
    const vp = mat4ApplyPoint(view, wp);
    const clip = mat4ApplyVec4(proj, { x: vp.x, y: vp.y, z: vp.z, w: 1 });
    const wn = mat4ApplyDir(model, { x: mesh.normals[i * 3], y: mesh.normals[i * 3 + 1], z: mesh.normals[i * 3 + 2] });
    let nl = Math.hypot(wn.x, wn.y, wn.z);
    if (nl === 0) nl = 1;
    const nx = wn.x / nl, ny = wn.y / nl, nz = wn.z / nl;
    const w = clip.w;
    const payload = new Float64Array(8);
    payload[0] = wp.x / w; payload[1] = wp.y / w; payload[2] = wp.z / w;
    payload[3] = nx / w; payload[4] = ny / w; payload[5] = nz / w;
    payload[6] = mesh.uvs[i * 2] / w; payload[7] = mesh.uvs[i * 2 + 1] / w;
    out.push({
      sx: (clip.x / w + 1) * 0.5 * W,
      sy: (1 - clip.y / w) * 0.5 * H,
      invW: 1 / w,
      dOverW: -vp.z / w,
      payload,
    });
  }
  return out;
}

function sampleBilinear(t: Texture, u: number, v: number, out: Float64Array): void {
  const uu = u - Math.floor(u);
  const vv = v - Math.floor(v);
  const fx = uu * t.width - 0.5;
  const fy = vv * t.height - 0.5;
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const tx = fx - x0, ty = fy - y0;
  const Wt = t.width, Ht = t.height;
  const x0w = ((x0 % Wt) + Wt) % Wt;
  const x1w = (((x0 + 1) % Wt) + Wt) % Wt;
  const y0w = ((y0 % Ht) + Ht) % Ht;
  const y1w = (((y0 + 1) % Ht) + Ht) % Ht;
  const T = (x: number, y: number, k: number): number => t.data[(y * Wt + x) * 4 + k];
  for (let k = 0; k < 3; k++) {
    const c00 = T(x0w, y0w, k), c10 = T(x1w, y0w, k), c01 = T(x0w, y1w, k), c11 = T(x1w, y1w, k);
    const row0 = c00 + (c10 - c00) * tx;
    const row1 = c01 + (c11 - c01) * tx;
    out[k] = row0 + (row1 - row0) * ty;
  }
}

function shade(
  nx: number, ny: number, nz: number,
  px: number, py: number, pz: number,
  eye: Vec3,
  lights: RefLights,
  ar: number, ag: number, ab: number,
  specular: number, shininess: number,
  specOn: boolean,
  out: Float64Array,
): void {
  let vx = eye.x - px, vy = eye.y - py, vz = eye.z - pz;
  let vl = Math.sqrt(vx * vx + vy * vy + vz * vz);
  if (vl === 0) vl = 1;
  vx /= vl; vy /= vl; vz /= vl;
  let r = ar * lights.ambient[0];
  let g = ag * lights.ambient[1];
  let b = ab * lights.ambient[2];
  for (const dir of lights.dirs) {
    const lx = dir.dir[0], ly = dir.dir[1], lz = dir.dir[2];
    const ndl = nx * lx + ny * ly + nz * lz;
    if (ndl > 0) {
      r += ar * dir.color[0] * ndl;
      g += ag * dir.color[1] * ndl;
      b += ab * dir.color[2] * ndl;
      if (specOn) {
        const hx = lx + vx, hy = ly + vy, hz = lz + vz;
        const hl = Math.sqrt(hx * hx + hy * hy + hz * hz);
        if (hl > 0) {
          const ndh = (nx * hx + ny * hy + nz * hz) / hl;
          if (ndh > 0) {
            const s = specular * Math.pow(ndh, shininess);
            r += dir.color[0] * s;
            g += dir.color[1] * s;
            b += dir.color[2] * s;
          }
        }
      }
    }
  }
  for (const pt of lights.points) {
    const ldx = pt.pos.x - px, ldy = pt.pos.y - py, ldz = pt.pos.z - pz;
    const dist = Math.sqrt(ldx * ldx + ldy * ldy + ldz * ldz);
    if (dist > 0) {
      const lx = ldx / dist, ly = ldy / dist, lz = ldz / dist;
      const atten = 1 / (1 + pt.k1 * dist + pt.k2 * dist * dist);
      const ndl = nx * lx + ny * ly + nz * lz;
      if (ndl > 0) {
        r += ar * pt.color[0] * ndl * atten;
        g += ag * pt.color[1] * ndl * atten;
        b += ab * pt.color[2] * ndl * atten;
        if (specOn) {
          const hx = lx + vx, hy = ly + vy, hz = lz + vz;
          const hl = Math.sqrt(hx * hx + hy * hy + hz * hz);
          if (hl > 0) {
            const ndh = (nx * hx + ny * hy + nz * hz) / hl;
            if (ndh > 0) {
              const s = specular * Math.pow(ndh, shininess);
              r += pt.color[0] * s * atten;
              g += pt.color[1] * s * atten;
              b += pt.color[2] * s * atten;
            }
          }
        }
      }
    }
  }
  out[0] = r * 255; out[1] = g * 255; out[2] = b * 255;
}

const INV255 = 1 / 255;

/** 参考渲染单个网格(phong,可选纹理/剔除开关)。假定全部顶点在近平面前。 */
export function refRenderMesh(
  fb: Framebuffer,
  mesh: Mesh,
  model: Mat4,
  material: RefMaterial,
  cam: RefCamera,
  lights: RefLights,
  cullBackface: boolean,
): void {
  const W = fb.width, H = fb.height;
  const verts = buildVerts(mesh, model, cam, W, H);
  const specOn = !material.lambert;
  const tc = mesh.indices.length / 3;
  for (let t = 0; t < tc; t++) {
    const a = verts[mesh.indices[t * 3]];
    const b = verts[mesh.indices[t * 3 + 1]];
    const c = verts[mesh.indices[t * 3 + 2]];
    const area = (b.sx - a.sx) * (c.sy - a.sy) - (b.sy - a.sy) * (c.sx - a.sx);
    if (!(Math.abs(area) > 1e-12)) continue;
    if (cullBackface && area > 0) continue; // 背面剔除(与正式管线同约定)
    let va = a, vb = b, vcc = c;
    if (area < 0) { vb = c; vcc = b; }
    const A = area < 0 ? -area : area;
    const e0x = vb.sx - va.sx, e0y = vb.sy - va.sy;
    const e1x = vcc.sx - vb.sx, e1y = vcc.sy - vb.sy;
    const e2x = va.sx - vcc.sx, e2y = va.sy - vcc.sy;
    const tl = (dy: number, dx: number): boolean => dy < 0 || (dy === 0 && dx > 0);
    const tl0 = tl(e0y, e0x), tl1 = tl(e1y, e1x), tl2 = tl(e2y, e2x);
    const xs = Math.max(0, Math.ceil(Math.min(va.sx, vb.sx, vcc.sx) - 0.5));
    const xe = Math.min(fb.width - 1, Math.floor(Math.max(va.sx, vb.sx, vcc.sx) - 0.5));
    const ys = Math.max(0, Math.ceil(Math.min(va.sy, vb.sy, vcc.sy) - 0.5));
    const ye = Math.min(fb.height - 1, Math.floor(Math.max(va.sy, vb.sy, vcc.sy) - 0.5));
    const out = new Float64Array(3);
    const texel = new Float64Array(3);
    const tex = material.texture;
    for (let y = ys; y <= ye; y++) {
      for (let x = xs; x <= xe; x++) {
        const px = x + 0.5, py = y + 0.5;
        const E0 = e0x * (py - va.sy) - e0y * (px - va.sx);
        const E1 = e1x * (py - vb.sy) - e1y * (px - vb.sx);
        const E2 = e2x * (py - vcc.sy) - e2y * (px - vcc.sx);
        let inside: boolean;
        if (E0 > 0 && E1 > 0 && E2 > 0) inside = true;
        else if (E0 >= 0 && E1 >= 0 && E2 >= 0) inside = (E0 > 0 || tl0) && (E1 > 0 || tl1) && (E2 > 0 || tl2);
        else inside = false;
        if (!inside) continue;
        const l0 = E1 / A, l1 = E2 / A, l2 = E0 / A;
        const denom = l0 * va.invW + l1 * vb.invW + l2 * vcc.invW;
        const d = (l0 * va.dOverW + l1 * vb.dOverW + l2 * vcc.dOverW) / denom;
        const idx = y * fb.width + x;
        if (!(d < fb.depth[idx])) continue;
        const p0x = (l0 * va.payload[0] + l1 * vb.payload[0] + l2 * vcc.payload[0]) / denom;
        const p0y = (l0 * va.payload[1] + l1 * vb.payload[1] + l2 * vcc.payload[1]) / denom;
        const p0z = (l0 * va.payload[2] + l1 * vb.payload[2] + l2 * vcc.payload[2]) / denom;
        let nx = (l0 * va.payload[3] + l1 * vb.payload[3] + l2 * vcc.payload[3]) / denom;
        let ny = (l0 * va.payload[4] + l1 * vb.payload[4] + l2 * vcc.payload[4]) / denom;
        let nz = (l0 * va.payload[5] + l1 * vb.payload[5] + l2 * vcc.payload[5]) / denom;
        let nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (nl === 0) nl = 1;
        nx /= nl; ny /= nl; nz /= nl;
        let ar = material.albedo[0], ag = material.albedo[1], ab = material.albedo[2];
        if (tex !== undefined) {
          const u = (l0 * va.payload[6] + l1 * vb.payload[6] + l2 * vcc.payload[6]) / denom;
          const v = (l0 * va.payload[7] + l1 * vb.payload[7] + l2 * vcc.payload[7]) / denom;
          sampleBilinear(tex, u, v, texel);
          ar = texel[0] * INV255 * ar;
          ag = texel[1] * INV255 * ag;
          ab = texel[2] * INV255 * ab;
        }
        shade(nx, ny, nz, p0x, p0y, p0z, cam.eye, lights, ar, ag, ab, material.specular, material.shininess, specOn, out);
        fb.depth[idx] = d;
        const o = idx * 4;
        fb.color[o] = out[0]; fb.color[o + 1] = out[1]; fb.color[o + 2] = out[2];
      }
    }
  }
}
