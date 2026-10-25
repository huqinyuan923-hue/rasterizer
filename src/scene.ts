/**
 * 程序化几何生成器（确定性，无随机）。网格约定：
 * - 三角形顶点序从外部看为逆时针（cross(v1-v0, v2-v0) 指向外侧）—— 与背面剔除约定配套。
 * - 法线为平滑（球）或面法线（立方体），单位长度。
 * - uv ∈ [0,1]。
 */

import type { Mesh } from "./pipeline.js";

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
