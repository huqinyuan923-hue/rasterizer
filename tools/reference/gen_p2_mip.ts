/**
 * M2(二期)参考实现:纹理 mip 一档 —— 半分辨率 box filter、逐子三角形 texel 密度选层、
 * 选层后采样。独立编码:box filter 副本、选层启发式副本、nearest/bilinear 采样器副本、
 * 顶点阶段与逐像素插值副本。checkerTexture 等既有生成器是夹具。
 *
 * 冻结口径(与正式管线一致):
 * - mip1 = 2×2 box filter(四通道均值 → Uint8ClampedArray 量化),奇数尺寸丢末行/列
 * - density = (uvArea2 × texW × texH) / screenArea2(clip 空间 uv 二倍面积 / 投影屏幕二倍面积)
 *   density ≥ 1(缩小)→ mip1,否则 mip0
 *
 * 黄金图:textured 模式(无光照),近处立方体(放大,mip0)与远处地面(缩小,mip1)同框:
 * - p2_mip_on.ppm:material.mip = true(地面平滑)
 * - p2_mip_off.ppm:material.mip = false(地面走样)
 * 两图必须不同(mip 选层确实生效)。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import type { DrawItem, Mesh, Material } from "../../src/pipeline.js";
import type { Texture } from "../../src/texture.js";
import {
  mat4LookAt, mat4Perspective, mat4ApplyPoint, mat4ApplyVec4, mat4Identity, mat4Translation,
  type Mat4, type Vec3,
} from "../../src/math.js";
import { checkerTexture } from "../../src/texture.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";

const W = 128;
const H = 128;

export const P2M_CAMERA = {
  eye: { x: 0, y: 1.6, z: 3.2 } as Vec3,
  target: { x: 0, y: -0.9, z: -2.0 } as Vec3,
  up: { x: 0, y: 1, z: 0 } as Vec3,
  fovYRad: (55 * Math.PI) / 180,
  aspect: 1,
  near: 0.1,
  far: 60,
};

/** 冻结场景参数(参考与正式共用同一份数据,实现各自独立)。 */
export const P2M_GROUND_TEX = { size: 128, cells: 16, a: [190, 184, 172] as [number, number, number], b: [70, 76, 82] as [number, number, number] };
export const P2M_CUBE_TEX = { size: 32, cells: 4, a: [205, 150, 80] as [number, number, number], b: [90, 60, 40] as [number, number, number] };

function groundMesh(): Mesh {
  // 前边 z=2、后边 z=-20 的地面(相机 z=3.2:四角全部在近平面前 —— 参考实现无近裁剪,
  // 场景夹具必须避开该已知限制;近平面裁剪语义已由 m7 黄金图单独仲裁),uv 0..8(8 次重复)。
  return {
    positions: [-20, -0.8, -20, 20, -0.8, -20, 20, -0.8, 2, -20, -0.8, 2],
    normals: [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    uvs: [0, 0, 8, 0, 8, 8, 0, 8],
    indices: [0, 2, 1, 0, 3, 2],
  };
}

/** 正式实现用的场景装配(mip 开关参数化)。 */
export function makeP2MItems(mip: boolean, groundTex: Texture, cubeTex: Texture): DrawItem[] {
  const groundMat: Material = {
    albedo: [1, 1, 1], specular: 0, shininess: 1,
    texture: groundTex, sampler: "bilinear", doubleSided: true, mip,
  };
  const cubeMat: Material = {
    albedo: [1, 1, 1], specular: 0, shininess: 1,
    texture: cubeTex, sampler: "bilinear", mip,
  };
  return [
    { mesh: groundMesh(), model: mat4Identity(), material: groundMat },
    { mesh: cubeMesh(), model: mat4Translation({ x: 0.85, y: -0.3, z: -0.4 }), material: cubeMat },
  ];
}

function cubeMesh(): Mesh {
  const h = 0.5;
  const faces: Array<{ n: [number, number, number]; verts: [number, number, number][] }> = [
    { n: [0, 0, 1], verts: [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]] },
    { n: [0, 0, -1], verts: [[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]] },
    { n: [1, 0, 0], verts: [[h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h]] },
    { n: [-1, 0, 0], verts: [[-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h]] },
    { n: [0, 1, 0], verts: [[-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h]] },
    { n: [0, -1, 0], verts: [[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]] },
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

// ---- 参考 box filter(mip1)副本 ----

function refMip1(t: Texture): Texture {
  const mw = Math.max(1, Math.floor(t.width / 2));
  const mh = Math.max(1, Math.floor(t.height / 2));
  const data = new Uint8ClampedArray(mw * mh * 4);
  for (let y = 0; y < mh; y++) {
    for (let x = 0; x < mw; x++) {
      const o = (y * mw + x) * 4;
      for (let c = 0; c < 4; c++) {
        const s00 = t.data[((y * 2) * t.width + (x * 2)) * 4 + c];
        const s10 = t.data[((y * 2) * t.width + (x * 2 + 1)) * 4 + c];
        const s01 = t.data[((y * 2 + 1) * t.width + (x * 2)) * 4 + c];
        const s11 = t.data[((y * 2 + 1) * t.width + (x * 2 + 1)) * 4 + c];
        data[o + c] = (s00 + s10 + s01 + s11) / 4;
      }
    }
  }
  return { width: mw, height: mh, data };
}

// ---- 参考采样器副本(repeat 环绕,冻结公式;本场景固定 bilinear) ----

function refBilinear(t: Texture, u: number, v: number, out: Float64Array): void {
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

// ---- 参考顶点阶段(textured 模式:payload = [u/w, v/w]) ----

interface RefVert { sx: number; sy: number; invW: number; dOverW: number; payload: Float64Array; uRaw: number; vRaw: number }

function refRenderTextured(fb: Framebuffer, mesh: Mesh, model: Mat4, tex: Texture, mip1: Texture | undefined, mipOn: boolean): void {
  const verts = buildVertsWithModel(mesh, model);
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

    // 逐三角形 mip 选层(冻结启发式;doubleSided 场景两面色序统一,选层在剔除后)
    let chosen = tex;
    if (mipOn && mip1 !== undefined) {
      const uvArea2 = Math.abs((vb.uRaw - va.uRaw) * (vcc.vRaw - va.vRaw) - (vb.vRaw - va.vRaw) * (vcc.uRaw - va.uRaw));
      const density = (uvArea2 * tex.width * tex.height) / A;
      chosen = density >= 1 ? mip1 : tex;
    }

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
        const u = (l0 * va.payload[0] + l1 * vb.payload[0] + l2 * vcc.payload[0]) / denom;
        const v = (l0 * va.payload[1] + l1 * vb.payload[1] + l2 * vcc.payload[1]) / denom;
        refBilinear(chosen, u, v, out);
        fb.depth[idx] = d;
        const o = idx * 4;
        fb.color[o] = out[0]; fb.color[o + 1] = out[1]; fb.color[o + 2] = out[2];
      }
    }
  }
}

function buildVertsWithModel(mesh: Mesh, model: Mat4): RefVert[] {
  const view = mat4LookAt(P2M_CAMERA.eye, P2M_CAMERA.target, P2M_CAMERA.up);
  const proj = mat4Perspective(P2M_CAMERA.fovYRad, P2M_CAMERA.aspect, P2M_CAMERA.near, P2M_CAMERA.far);
  const vc = mesh.positions.length / 3;
  const out: RefVert[] = [];
  for (let i = 0; i < vc; i++) {
    const p = { x: mesh.positions[i * 3], y: mesh.positions[i * 3 + 1], z: mesh.positions[i * 3 + 2] };
    const wp = mat4ApplyPoint(model, p);
    const vp = mat4ApplyPoint(view, wp);
    const clip = mat4ApplyVec4(proj, { x: vp.x, y: vp.y, z: vp.z, w: 1 });
    const w = clip.w;
    const u = mesh.uvs[i * 2], v = mesh.uvs[i * 2 + 1];
    const payload = new Float64Array(2);
    payload[0] = u / w; payload[1] = v / w;
    out.push({
      sx: (clip.x / w + 1) * 0.5 * W,
      sy: (1 - clip.y / w) * 0.5 * H,
      invW: 1 / w,
      dOverW: -vp.z / w,
      payload,
      uRaw: u,
      vRaw: v,
    });
  }
  return out;
}

export function renderP2M(mipOn: boolean): RawImage {
  const fb = new Framebuffer(W, H);
  const groundTex = checkerTexture(P2M_GROUND_TEX.size, P2M_GROUND_TEX.cells, P2M_GROUND_TEX.a, P2M_GROUND_TEX.b);
  const cubeTex = checkerTexture(P2M_CUBE_TEX.size, P2M_CUBE_TEX.cells, P2M_CUBE_TEX.a, P2M_CUBE_TEX.b);
  const groundMip1 = refMip1(groundTex);
  const cubeMip1 = refMip1(cubeTex);
  refRenderTextured(fb, groundMesh(), mat4Identity(), groundTex, groundMip1, mipOn);
  refRenderTextured(fb, cubeMesh(), mat4Translation({ x: 0.85, y: -0.3, z: -0.4 }), cubeTex, cubeMip1, mipOn);
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("p2_mip_on.ppm", renderP2M(true));
  writeGolden("p2_mip_off.ppm", renderP2M(false));
  console.log("p2_mip_on.ppm / p2_mip_off.ppm written");
}

if (process.argv[1] && process.argv[1].endsWith("gen_p2_mip.js")) {
  run();
}
