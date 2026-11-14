/**
 * M2(二期)参考实现:多光源材质系统 —— 方向光/点光数组 + 聚光(软锥)+ Lambert/Blinn 开关。
 * 独立编码:顶点阶段、逐像素重心插值、多光源着色公式副本。数学库与网格生成器是夹具。
 *
 * 黄金图:
 * - p2_lights_multi.ppm:2 方向光 + 2 点光,场景含 blinnPhong 与 lambert 两种材质
 * - p2_lights_spot.ppm:1 弱方向光 + 2 聚光(软锥,地面可见锥形光斑)
 */

import { Framebuffer } from "../../src/framebuffer.js";
import type { Material, Mesh } from "../../src/pipeline.js";
import {
  mat4LookAt, mat4Perspective, mat4ApplyPoint, mat4ApplyDir, mat4ApplyVec4,
  type Mat4, type Vec3,
} from "../../src/math.js";
import { makeSphere, makeCube } from "../../src/scene.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";

const W = 128;
const H = 128;

export const P2L_CAMERA = {
  eye: { x: 0, y: 1.1, z: 5.2 } as Vec3,
  target: { x: 0, y: -0.35, z: 0 } as Vec3,
  up: { x: 0, y: 1, z: 0 } as Vec3,
  fovYRad: (50 * Math.PI) / 180,
  aspect: 1,
  near: 0.1,
  far: 30,
};

export interface P2LSetup {
  ambient: [number, number, number];
  dirs: Array<{ dir: [number, number, number]; color: [number, number, number] }>;
  points: Array<{ pos: Vec3; color: [number, number, number]; k1: number; k2: number }>;
  spots: Array<{ pos: Vec3; dir: [number, number, number]; color: [number, number, number]; cosInner: number; cosOuter: number; k1: number; k2: number }>;
}

export const P2L_LIGHTS_MULTI: P2LSetup = {
  ambient: [0.1, 0.1, 0.12],
  dirs: [
    { dir: [0.45, 0.8, 0.5], color: [0.7, 0.72, 0.75] },
    { dir: [-0.6, 0.3, -0.2], color: [0.35, 0.28, 0.22] },
  ],
  points: [
    { pos: { x: 2.2, y: 2.2, z: 1.6 }, color: [1.0, 0.75, 0.5], k1: 0.1, k2: 0.03 },
    { pos: { x: -2.5, y: 1.2, z: -1.0 }, color: [0.3, 0.5, 1.0], k1: 0.15, k2: 0.05 },
  ],
  spots: [],
};

export const P2L_LIGHTS_SPOT: P2LSetup = {
  ambient: [0.06, 0.06, 0.08],
  dirs: [
    { dir: [0.2, 0.9, 0.1], color: [0.25, 0.25, 0.28] },
  ],
  points: [],
  spots: [
    { pos: { x: 0, y: 3.2, z: 0 }, dir: [0, -1, 0.15], color: [1.0, 0.9, 0.7], cosInner: Math.cos(20 * Math.PI / 180), cosOuter: Math.cos(45 * Math.PI / 180), k1: 0.05, k2: 0.01 },
    { pos: { x: -2.2, y: 2.4, z: 1.5 }, dir: [0.3, -1, -0.2], color: [0.4, 0.7, 1.0], cosInner: Math.cos(12 * Math.PI / 180), cosOuter: Math.cos(35 * Math.PI / 180), k1: 0.1, k2: 0.04 },
  ],
};

// 方向光归一化(生成时一次,冻结结果)——两个 setup 都做
for (const setup of [P2L_LIGHTS_MULTI, P2L_LIGHTS_SPOT]) {
  for (const d of setup.dirs) {
    const l = Math.hypot(d.dir[0], d.dir[1], d.dir[2]);
    d.dir[0] /= l; d.dir[1] /= l; d.dir[2] /= l;
  }
}

export interface P2LItem {
  mesh: Mesh;
  model: Mat4;
  albedo: [number, number, number];
  specular: number;
  shininess: number;
  lambert: boolean;
  doubleSided: boolean;
}

/** 场景:球(blinn 红)+ 立方体(lambert 蓝)+ 地面(blinn 灰)。材质开关是 M2 被测对象的一部分。 */
export const P2L_ITEMS: P2LItem[] = [
  {
    mesh: makeSphere(0.8, 32, 16),
    model: translation(0, 0.1, 0),
    albedo: [0.85, 0.3, 0.22],
    specular: 0.55,
    shininess: 32,
    lambert: false,
    doubleSided: false,
  },
  {
    mesh: makeCube(0.9),
    model: translation(1.6, -0.25, 0.3),
    albedo: [0.25, 0.55, 0.85],
    specular: 0.4,
    shininess: 12,
    lambert: true,
    doubleSided: false,
  },
  {
    mesh: planeMesh(9),
    model: identity(),
    albedo: [0.55, 0.58, 0.62],
    specular: 0.2,
    shininess: 8,
    lambert: false,
    doubleSided: true,
  },
];

/** 正式实现用的材质视图(lambert → shadingModel)。 */
export function p2lMaterial(it: P2LItem): Material {
  return {
    albedo: it.albedo,
    specular: it.specular,
    shininess: it.shininess,
    ...(it.lambert ? { shadingModel: "lambert" as const } : {}),
    doubleSided: it.doubleSided,
  };
}

function planeMesh(size: number): Mesh {
  const h = size / 2;
  return {
    positions: [-h, -0.9, -h, h, -0.9, -h, h, -0.9, h, -h, -0.9, h],
    normals: [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    uvs: [0, 0, 1, 0, 1, 1, 0, 1],
    indices: [0, 2, 1, 0, 3, 2],
  };
}

function identity(): Mat4 {
  const m = new Float64Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
}
function translation(x: number, y: number, z: number): Mat4 {
  const m = identity();
  m[12] = x; m[13] = y; m[14] = z;
  return m;
}

// ---- 参考着色公式副本(多光源数组 + 聚光,与 src/shading.ts 冻结公式一致) ----

function refShade(
  nx: number, ny: number, nz: number,
  px: number, py: number, pz: number,
  eye: Vec3,
  lights: P2LSetup,
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
  for (const sp of lights.spots) {
    const d0 = sp.dir[0], d1 = sp.dir[1], d2 = sp.dir[2];
    const dl = Math.sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    if (dl === 0) continue;
    const ax = d0 / dl, ay = d1 / dl, az = d2 / dl;
    const ldx = sp.pos.x - px, ldy = sp.pos.y - py, ldz = sp.pos.z - pz;
    const dist = Math.sqrt(ldx * ldx + ldy * ldy + ldz * ldz);
    if (dist > 0) {
      const lx = ldx / dist, ly = ldy / dist, lz = ldz / dist;
      const cosTheta = -(lx * ax + ly * ay + lz * az);
      if (cosTheta > sp.cosOuter) {
        const denom = sp.cosInner - sp.cosOuter;
        let t = 1;
        if (denom > 0) {
          t = (cosTheta - sp.cosOuter) / denom;
          if (t > 1) t = 1;
        }
        const atten = 1 / (1 + sp.k1 * dist + sp.k2 * dist * dist);
        const ndl = nx * lx + ny * ly + nz * lz;
        if (ndl > 0) {
          r += ar * sp.color[0] * ndl * atten * t;
          g += ag * sp.color[1] * ndl * atten * t;
          b += ab * sp.color[2] * ndl * atten * t;
          if (specOn) {
            const hx = lx + vx, hy = ly + vy, hz = lz + vz;
            const hl = Math.sqrt(hx * hx + hy * hy + hz * hz);
            if (hl > 0) {
              const ndh = (nx * hx + ny * hy + nz * hz) / hl;
              if (ndh > 0) {
                const s = specular * Math.pow(ndh, shininess);
                r += sp.color[0] * s * atten * t;
                g += sp.color[1] * s * atten * t;
                b += sp.color[2] * s * atten * t;
              }
            }
          }
        }
      }
    }
  }
  out[0] = r * 255; out[1] = g * 255; out[2] = b * 255;
}

// ---- 参考顶点阶段(冻结公式,含 Math.hypot 法线归一化) ----

interface RefVert { sx: number; sy: number; invW: number; dOverW: number; payload: Float64Array }

function buildVerts(item: P2LItem): RefVert[] {
  const view = mat4LookAt(P2L_CAMERA.eye, P2L_CAMERA.target, P2L_CAMERA.up);
  const proj = mat4Perspective(P2L_CAMERA.fovYRad, P2L_CAMERA.aspect, P2L_CAMERA.near, P2L_CAMERA.far);
  const mesh = item.mesh;
  const vc = mesh.positions.length / 3;
  const out: RefVert[] = [];
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
    const payload = new Float64Array(6);
    payload[0] = wp.x / w; payload[1] = wp.y / w; payload[2] = wp.z / w;
    payload[3] = nx / w; payload[4] = ny / w; payload[5] = nz / w;
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

function refRenderItem(fb: Framebuffer, item: P2LItem, lights: P2LSetup): void {
  const verts = buildVerts(item);
  const mesh = item.mesh;
  const specOn = !item.lambert;
  const tc = mesh.indices.length / 3;
  for (let t = 0; t < tc; t++) {
    const a = verts[mesh.indices[t * 3]];
    const b = verts[mesh.indices[t * 3 + 1]];
    const c = verts[mesh.indices[t * 3 + 2]];
    const area = (b.sx - a.sx) * (c.sy - a.sy) - (b.sy - a.sy) * (c.sx - a.sx);
    if (!(Math.abs(area) > 1e-12)) continue;
    if (area > 0 && !item.doubleSided) continue; // 背面剔除(约定与正式管线一致)
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
        const p0x = (l0 * va.payload[0] + l1 * vb.payload[0] + l2 * vcc.payload[0]) / denom;
        const p0y = (l0 * va.payload[1] + l1 * vb.payload[1] + l2 * vcc.payload[1]) / denom;
        const p0z = (l0 * va.payload[2] + l1 * vb.payload[2] + l2 * vcc.payload[2]) / denom;
        let nx = (l0 * va.payload[3] + l1 * vb.payload[3] + l2 * vcc.payload[3]) / denom;
        let ny = (l0 * va.payload[4] + l1 * vb.payload[4] + l2 * vcc.payload[4]) / denom;
        let nz = (l0 * va.payload[5] + l1 * vb.payload[5] + l2 * vcc.payload[5]) / denom;
        let nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (nl === 0) nl = 1;
        nx /= nl; ny /= nl; nz /= nl;
        refShade(nx, ny, nz, p0x, p0y, p0z, P2L_CAMERA.eye, lights, item.albedo[0], item.albedo[1], item.albedo[2], item.specular, item.shininess, specOn, out);
        fb.depth[idx] = d;
        const o = idx * 4;
        fb.color[o] = out[0]; fb.color[o + 1] = out[1]; fb.color[o + 2] = out[2];
      }
    }
  }
}

export function renderP2L(lights: P2LSetup): RawImage {
  const fb = new Framebuffer(W, H);
  for (const item of P2L_ITEMS) refRenderItem(fb, item, lights);
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("p2_lights_multi.ppm", renderP2L(P2L_LIGHTS_MULTI));
  writeGolden("p2_lights_spot.ppm", renderP2L(P2L_LIGHTS_SPOT));
  console.log("p2_lights_multi.ppm / p2_lights_spot.ppm written");
}

if (process.argv[1] && process.argv[1].endsWith("gen_p2_lights.js")) {
  run();
}
