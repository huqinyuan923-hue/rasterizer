/**
 * M6 参考实现：Gouraud 与 Phong 两档光照（独立编码）。
 * 场景：球（平滑法线）+ 立方体（面法线），方向光 + 点光 + 环境项。
 * 参考实现自带：顶点阶段（冻结公式）、逐像素重心插值、独立着色公式副本。
 * 几何（makeSphere/makeCube）与数学库是夹具，着色是本里程碑被测对象。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import {
  mat4LookAt, mat4Perspective, mat4ApplyPoint, mat4ApplyDir, mat4ApplyVec4,
  type Mat4, type Vec3,
} from "../../src/math.js";
import { makeSphere, makeCube } from "../../src/scene.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";

export const M6_W = 128;
export const M6_H = 128;

export const M6_CAMERA = {
  eye: { x: 0, y: 0.25, z: 4.5 } as Vec3,
  target: { x: 0.2, y: -0.05, z: 0 } as Vec3,
  up: { x: 0, y: 1, z: 0 } as Vec3,
  fovYRad: (50 * Math.PI) / 180,
  aspect: 1,
  near: 0.1,
  far: 20,
};

export const M6_LIGHTS = {
  ambient: [0.1, 0.1, 0.12] as [number, number, number],
  dir: { dir: [0.45, 0.8, 0.5] as [number, number, number], color: [0.85, 0.82, 0.75] as [number, number, number] },
  point: { pos: { x: 2.2, y: 2.2, z: 1.6 } as Vec3, color: [1.0, 0.75, 0.5] as [number, number, number], k1: 0.1, k2: 0.03 },
};

// 方向光方向归一化（生成时做一次，冻结结果）
{
  const d = M6_LIGHTS.dir.dir;
  const l = Math.hypot(d[0], d[1], d[2]);
  d[0] /= l; d[1] /= l; d[2] /= l;
}

export interface M6Item {
  mesh: { positions: number[]; normals: number[]; uvs: number[]; indices: number[] };
  model: Mat4;
  albedo: [number, number, number];
  specular: number;
  shininess: number;
}

export const M6_ITEMS: M6Item[] = [
  {
    mesh: makeSphere(1, 32, 16),
    model: mat4Identity(),
    albedo: [0.85, 0.3, 0.22],
    specular: 0.55,
    shininess: 32,
  },
  {
    mesh: makeCube(0.9),
    model: translation(1.5, -0.2, 0.1),
    albedo: [0.25, 0.55, 0.85],
    specular: 0.25,
    shininess: 12,
  },
];

function mat4Identity(): Mat4 {
  const m = new Float64Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
}
function translation(x: number, y: number, z: number): Mat4 {
  const m = mat4Identity();
  m[12] = x; m[13] = y; m[14] = z;
  return m;
}

// ---- 参考着色公式副本（与 src/shading.ts 冻结公式一致） ----

function refShade(
  nx: number, ny: number, nz: number,
  px: number, py: number, pz: number,
  lights: typeof M6_LIGHTS,
  ar: number, ag: number, ab: number,
  specular: number, shininess: number,
  out: Float64Array,
): void {
  const eye = M6_CAMERA.eye;
  let vx = eye.x - px, vy = eye.y - py, vz = eye.z - pz;
  let vl = Math.sqrt(vx * vx + vy * vy + vz * vz);
  if (vl === 0) vl = 1;
  vx /= vl; vy /= vl; vz /= vl;

  let r = ar * lights.ambient[0];
  let g = ag * lights.ambient[1];
  let b = ab * lights.ambient[2];

  {
    const lx = lights.dir.dir[0], ly = lights.dir.dir[1], lz = lights.dir.dir[2];
    const ndl = nx * lx + ny * ly + nz * lz;
    if (ndl > 0) {
      r += ar * lights.dir.color[0] * ndl;
      g += ag * lights.dir.color[1] * ndl;
      b += ab * lights.dir.color[2] * ndl;
      const hx = lx + vx, hy = ly + vy, hz = lz + vz;
      const hl = Math.sqrt(hx * hx + hy * hy + hz * hz);
      if (hl > 0) {
        const ndh = (nx * hx + ny * hy + nz * hz) / hl;
        if (ndh > 0) {
          const s = specular * Math.pow(ndh, shininess);
          r += lights.dir.color[0] * s;
          g += lights.dir.color[1] * s;
          b += lights.dir.color[2] * s;
        }
      }
    }
  }
  {
    const p = lights.point;
    const ldx = p.pos.x - px, ldy = p.pos.y - py, ldz = p.pos.z - pz;
    const dist = Math.sqrt(ldx * ldx + ldy * ldy + ldz * ldz);
    if (dist > 0) {
      const lx = ldx / dist, ly = ldy / dist, lz = ldz / dist;
      const atten = 1 / (1 + p.k1 * dist + p.k2 * dist * dist);
      const ndl = nx * lx + ny * ly + nz * lz;
      if (ndl > 0) {
        r += ar * p.color[0] * ndl * atten;
        g += ag * p.color[1] * ndl * atten;
        b += ab * p.color[2] * ndl * atten;
        const hx = lx + vx, hy = ly + vy, hz = lz + vz;
        const hl = Math.sqrt(hx * hx + hy * hy + hz * hz);
        if (hl > 0) {
          const ndh = (nx * hx + ny * hy + nz * hz) / hl;
          if (ndh > 0) {
            const s = specular * Math.pow(ndh, shininess);
            r += p.color[0] * s * atten;
            g += p.color[1] * s * atten;
            b += p.color[2] * s * atten;
          }
        }
      }
    }
  }
  out[0] = r * 255; out[1] = g * 255; out[2] = b * 255;
}

// ---- 参考顶点阶段（冻结公式，含 Math.hypot 法线归一化） ----

interface RefVert { sx: number; sy: number; invW: number; dOverW: number; payload: Float64Array }

function buildVerts(item: M6Item, gouraud: boolean): RefVert[] {
  const view = mat4LookAt(M6_CAMERA.eye, M6_CAMERA.target, M6_CAMERA.up);
  const proj = mat4Perspective(M6_CAMERA.fovYRad, M6_CAMERA.aspect, M6_CAMERA.near, M6_CAMERA.far);
  const mesh = item.mesh;
  const vc = mesh.positions.length / 3;
  const out: RefVert[] = [];
  const lit = new Float64Array(3);
  for (let i = 0; i < vc; i++) {
    const p = { x: mesh.positions[i * 3], y: mesh.positions[i * 3 + 1], z: mesh.positions[i * 3 + 2] };
    const wp = mat4ApplyPoint(item.model, p);
    const vp = mat4ApplyPoint(view, wp);
    const clip = mat4ApplyVec4(proj, { x: vp.x, y: vp.y, z: vp.z, w: 1 });
    const wn = mat4ApplyDir(item.model, { x: mesh.normals[i * 3], y: mesh.normals[i * 3 + 1], z: mesh.normals[i * 3 + 2] });
    let nl = Math.hypot(wn.x, wn.y, wn.z);
    if (nl === 0) nl = 1;
    const nx = wn.x / nl, ny = wn.y / nl, nz = wn.z / nl;
    const w = clip.w;
    const payload = new Float64Array(gouraud ? 3 : 6);
    if (gouraud) {
      refShade(nx, ny, nz, wp.x, wp.y, wp.z, M6_LIGHTS, item.albedo[0], item.albedo[1], item.albedo[2], item.specular, item.shininess, lit);
      payload[0] = lit[0] / w; payload[1] = lit[1] / w; payload[2] = lit[2] / w;
    } else {
      payload[0] = wp.x / w; payload[1] = wp.y / w; payload[2] = wp.z / w;
      payload[3] = nx / w; payload[4] = ny / w; payload[5] = nz / w;
    }
    out.push({
      sx: (clip.x / w + 1) * 0.5 * M6_W,
      sy: (1 - clip.y / w) * 0.5 * M6_H,
      invW: 1 / w,
      dOverW: -vp.z / w,
      payload,
    });
  }
  return out;
}

function refRenderItem(fb: Framebuffer, item: M6Item, gouraud: boolean): void {
  const verts = buildVerts(item, gouraud);
  const mesh = item.mesh;
  const tc = mesh.indices.length / 3;
  for (let t = 0; t < tc; t++) {
    const a = verts[mesh.indices[t * 3]];
    const b = verts[mesh.indices[t * 3 + 1]];
    const c = verts[mesh.indices[t * 3 + 2]];
    const area = (b.sx - a.sx) * (c.sy - a.sy) - (b.sy - a.sy) * (c.sx - a.sx);
    if (!(Math.abs(area) > 1e-12)) continue;
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
        if (gouraud) {
          out[0] = (l0 * va.payload[0] + l1 * vb.payload[0] + l2 * vcc.payload[0]) / denom;
          out[1] = (l0 * va.payload[1] + l1 * vb.payload[1] + l2 * vcc.payload[1]) / denom;
          out[2] = (l0 * va.payload[2] + l1 * vb.payload[2] + l2 * vcc.payload[2]) / denom;
        } else {
          const p0x = (l0 * va.payload[0] + l1 * vb.payload[0] + l2 * vcc.payload[0]) / denom;
          const p0y = (l0 * va.payload[1] + l1 * vb.payload[1] + l2 * vcc.payload[1]) / denom;
          const p0z = (l0 * va.payload[2] + l1 * vb.payload[2] + l2 * vcc.payload[2]) / denom;
          let nx = (l0 * va.payload[3] + l1 * vb.payload[3] + l2 * vcc.payload[3]) / denom;
          let ny = (l0 * va.payload[4] + l1 * vb.payload[4] + l2 * vcc.payload[4]) / denom;
          let nz = (l0 * va.payload[5] + l1 * vb.payload[5] + l2 * vcc.payload[5]) / denom;
          let nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
          if (nl === 0) nl = 1;
          nx /= nl; ny /= nl; nz /= nl;
          refShade(nx, ny, nz, p0x, p0y, p0z, M6_LIGHTS, item.albedo[0], item.albedo[1], item.albedo[2], item.specular, item.shininess, out);
        }
        fb.depth[idx] = d;
        const o = idx * 4;
        fb.color[o] = out[0]; fb.color[o + 1] = out[1]; fb.color[o + 2] = out[2];
      }
    }
  }
}

export function renderM6(gouraud: boolean): RawImage {
  const fb = new Framebuffer(M6_W, M6_H);
  for (const item of M6_ITEMS) refRenderItem(fb, item, gouraud);
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("m6_gouraud.ppm", renderM6(true));
  writeGolden("m6_phong.ppm", renderM6(false));
  console.log("m6_gouraud.ppm / m6_phong.ppm written");
}
