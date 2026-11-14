/**
 * M3(二期)参考实现:OBJ 加载 —— 独立解析器副本 + 参考 phong+纹理渲染。
 * 场景夹具:tests/fixtures/model.obj(覆盖 v / v/vt / v//vn / v/vt/vn 四种索引形态、
 * 负索引、四边形扇形三角化、注释与未知指令)。
 * 黄金图:p2_obj.ppm —— 参考解析出的网格经参考渲染管线落盘;正式实现
 * (src/obj.ts parseOBJ + Renderer)必须逐像素一致。
 */

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Framebuffer } from "../../src/framebuffer.js";
import type { Mesh } from "../../src/pipeline.js";
import {
  mat4LookAt, mat4Perspective, mat4ApplyPoint, mat4ApplyDir, mat4ApplyVec4, mat4Identity,
  type Mat4, type Vec3,
} from "../../src/math.js";
import { checkerTexture, type Texture } from "../../src/texture.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";

const HERE = dirname(fileURLToPath(import.meta.url));
// 编译后位于 build/tools/reference/,仓库根为其上三级
export const OBJ_FIXTURE_PATH = join(HERE, "..", "..", "..", "tests", "fixtures", "model.obj");

const W = 128;
const H = 128;

export const P2O_CAMERA = {
  eye: { x: 0, y: 1.0, z: 4.6 } as Vec3,
  target: { x: 0, y: 0.15, z: 0.3 } as Vec3,
  up: { x: 0, y: 1, z: 0 } as Vec3,
  fovYRad: (50 * Math.PI) / 180,
  aspect: 1,
  near: 0.1,
  far: 30,
};

export const P2O_LIGHTS = {
  ambient: [0.1, 0.1, 0.12] as [number, number, number],
  dirs: [{ dir: [0.45, 0.8, 0.5] as [number, number, number], color: [0.85, 0.82, 0.75] as [number, number, number] }],
  points: [{ pos: { x: 2.2, y: 2.2, z: 1.6 } as Vec3, color: [1.0, 0.75, 0.5] as [number, number, number], k1: 0.1, k2: 0.03 }],
  spots: [] as never[],
};
{
  const d = P2O_LIGHTS.dirs[0].dir;
  const l = Math.hypot(d[0], d[1], d[2]);
  d[0] /= l; d[1] /= l; d[2] /= l;
}

export const P2O_TEX = { size: 32, cells: 4, a: [205, 150, 80] as [number, number, number], b: [90, 60, 40] as [number, number, number] };
export const P2O_MATERIAL = {
  albedo: [1, 1, 1] as [number, number, number],
  specular: 0.35,
  shininess: 24,
  doubleSided: true,
};

// ---- 独立解析器副本(冻结规则,写法独立于 src/obj.ts) ----

export function refParseOBJ(text: string): Mesh {
  const vRaw: number[] = [];
  const uvRaw: number[] = [];
  const nRaw: number[] = [];
  const expPos: number[] = [];
  const expUV: number[] = [];
  const expNrm: number[] = [];
  const expIdx: number[] = [];
  for (const rawLine of text.split("\n")) {
    const line = rawLine.replace(/\r$/, "");
    if (line === "" || line.startsWith("#")) continue;
    const t = line.trim().split(/\s+/);
    if (t[0] === "v") {
      vRaw.push(Number(t[1]), Number(t[2]), Number(t[3]));
    } else if (t[0] === "vt") {
      uvRaw.push(Number(t[1]), Number(t[2]));
    } else if (t[0] === "vn") {
      nRaw.push(Number(t[1]), Number(t[2]), Number(t[3]));
    } else if (t[0] === "f" && t.length >= 4) {
      const nv = vRaw.length / 3, nu = uvRaw.length / 2, nn = nRaw.length / 3;
      const corners = t.slice(1).map((tok) => {
        const seg = tok.split("/");
        const rawV = Number(seg[0]);
        const vi = rawV > 0 ? rawV - 1 : nv + rawV;
        let ti = -1;
        if (seg.length >= 2 && seg[1] !== "") {
          const rawT = Number(seg[1]);
          ti = rawT > 0 ? rawT - 1 : nu + rawT;
        }
        let ni = -1;
        if (seg.length >= 3 && seg[2] !== "") {
          const rawN = Number(seg[2]);
          ni = rawN > 0 ? rawN - 1 : nn + rawN;
        }
        return { vi, ti, ni };
      });
      for (let k = 1; k + 1 < corners.length; k++) {
        const tri = [corners[0], corners[k], corners[k + 1]];
        const pts = tri.map((c) => ({ x: vRaw[c.vi * 3], y: vRaw[c.vi * 3 + 1], z: vRaw[c.vi * 3 + 2] }));
        let gx = 0, gy = 1, gz = 0;
        if (tri.some((c) => c.ni < 0)) {
          const ax = pts[1].x - pts[0].x, ay = pts[1].y - pts[0].y, az = pts[1].z - pts[0].z;
          const bx = pts[2].x - pts[0].x, by = pts[2].y - pts[0].y, bz = pts[2].z - pts[0].z;
          const cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
          const l = Math.sqrt(cx * cx + cy * cy + cz * cz);
          if (l > 0) { gx = cx / l; gy = cy / l; gz = cz / l; }
        }
        const base = expPos.length / 3;
        for (let m = 0; m < 3; m++) {
          const c = tri[m];
          expPos.push(pts[m].x, pts[m].y, pts[m].z);
          if (c.ti >= 0) expUV.push(uvRaw[c.ti * 2], uvRaw[c.ti * 2 + 1]); else expUV.push(0, 0);
          if (c.ni >= 0) expNrm.push(nRaw[c.ni * 3], nRaw[c.ni * 3 + 1], nRaw[c.ni * 3 + 2]);
          else expNrm.push(gx, gy, gz);
        }
        expIdx.push(base, base + 1, base + 2);
      }
    }
  }
  // 输出 = 展开块（与正式实现同布局冻结）
  return { positions: expPos, normals: expNrm, uvs: expUV, indices: expIdx };
}

// ---- 参考 phong+纹理渲染(payload: pos/normal/uv 共 8 varying) ----

interface RefVert { sx: number; sy: number; invW: number; dOverW: number; payload: Float64Array }

function buildVerts(mesh: Mesh, model: Mat4): RefVert[] {
  const view = mat4LookAt(P2O_CAMERA.eye, P2O_CAMERA.target, P2O_CAMERA.up);
  const proj = mat4Perspective(P2O_CAMERA.fovYRad, P2O_CAMERA.aspect, P2O_CAMERA.near, P2O_CAMERA.far);
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

function refSampleBilinear(t: Texture, u: number, v: number, out: Float64Array): void {
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

function refShade(
  nx: number, ny: number, nz: number,
  px: number, py: number, pz: number,
  ar: number, ag: number, ab: number,
  specular: number, shininess: number,
  out: Float64Array,
): void {
  let vx = P2O_CAMERA.eye.x - px, vy = P2O_CAMERA.eye.y - py, vz = P2O_CAMERA.eye.z - pz;
  let vl = Math.sqrt(vx * vx + vy * vy + vz * vz);
  if (vl === 0) vl = 1;
  vx /= vl; vy /= vl; vz /= vl;
  let r = ar * P2O_LIGHTS.ambient[0];
  let g = ag * P2O_LIGHTS.ambient[1];
  let b = ab * P2O_LIGHTS.ambient[2];
  for (const dir of P2O_LIGHTS.dirs) {
    const lx = dir.dir[0], ly = dir.dir[1], lz = dir.dir[2];
    const ndl = nx * lx + ny * ly + nz * lz;
    if (ndl > 0) {
      r += ar * dir.color[0] * ndl;
      g += ag * dir.color[1] * ndl;
      b += ab * dir.color[2] * ndl;
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
  for (const pt of P2O_LIGHTS.points) {
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
  out[0] = r * 255; out[1] = g * 255; out[2] = b * 255;
}

const INV255 = 1 / 255;

function refRenderMesh(fb: Framebuffer, mesh: Mesh, model: Mat4, tex: Texture): void {
  const verts = buildVerts(mesh, model);
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
    const texel = new Float64Array(3);
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
        const u = (l0 * va.payload[6] + l1 * vb.payload[6] + l2 * vcc.payload[6]) / denom;
        const v = (l0 * va.payload[7] + l1 * vb.payload[7] + l2 * vcc.payload[7]) / denom;
        refSampleBilinear(tex, u, v, texel);
        const ar = texel[0] * INV255 * P2O_MATERIAL.albedo[0];
        const ag = texel[1] * INV255 * P2O_MATERIAL.albedo[1];
        const ab = texel[2] * INV255 * P2O_MATERIAL.albedo[2];
        refShade(nx, ny, nz, p0x, p0y, p0z, ar, ag, ab, P2O_MATERIAL.specular, P2O_MATERIAL.shininess, out);
        fb.depth[idx] = d;
        const o = idx * 4;
        fb.color[o] = out[0]; fb.color[o + 1] = out[1]; fb.color[o + 2] = out[2];
      }
    }
  }
}

export function loadFixtureText(): string {
  return readFileSync(OBJ_FIXTURE_PATH, "utf8");
}

export function renderP2O(): RawImage {
  const fb = new Framebuffer(W, H);
  const mesh = refParseOBJ(loadFixtureText());
  const tex = checkerTexture(P2O_TEX.size, P2O_TEX.cells, P2O_TEX.a, P2O_TEX.b);
  refRenderMesh(fb, mesh, mat4Identity(), tex);
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("p2_obj.ppm", renderP2O());
  console.log("p2_obj.ppm written");
}

if (process.argv[1] && process.argv[1].endsWith("gen_p2_obj.js")) {
  run();
}
