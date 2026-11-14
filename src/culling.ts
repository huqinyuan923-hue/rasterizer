/**
 * 剔除（二期 M3/M4）：网格包围球 + 视锥剔除（5 平面：近平面 + 4 侧面）。
 *
 * 冻结规则（黄金图对拍口径）：
 * - 包围球：中心 = 顶点 AABB 中心；半径 = 中心到最远顶点的距离。
 * - 平面：Gribb–Hartmann 从 view·proj 矩阵行提取（列主序 m[c*4+r]，第 i 行第 j 列
 *   元素 = m[j*4+i]）。只用 5 个平面（near/left/right/bottom/top）—— 本管线不裁剪
 *   远平面（远于 far 的几何仍会被屏幕空间 guard 画出），按远平面剔除会改变像素；
 *   侧面/近平面之外的项目在既有光栅器下产出 0 像素，剔除是纯性能优化、像素安全。
 * - 平面归一化：除以法线模长（sqrt 公式）；模长 0 跳过该平面（防御，不发生）。
 * - 世界空间球：center' = model·center（仿射直接展开，不除 w）；半径
 *   r' = r × model 3×3 部分的 Frobenius 范数（σmax 的保守上界，绝不误剔）。
 * - 剔除判定：dist = a·cx+b·cy+c·cz+d < −r' → 整项跳过。边界（dist == −r'）不剔。
 */

import { mat4Mul, type Mat4 } from "./math.js";
import type { Mesh } from "./pipeline.js";

export interface BoundingSphere {
  cx: number;
  cy: number;
  cz: number;
  r: number;
}

/** 网格包围球（AABB 中心 + 最远顶点距离）。空网格返回半径 0 的原点球（调用方自防）。 */
export function meshBoundingSphere(mesh: Mesh): BoundingSphere {
  const n = mesh.positions.length / 3;
  if (n === 0) return { cx: 0, cy: 0, cz: 0, r: 0 };
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = mesh.positions[i * 3], y = mesh.positions[i * 3 + 1], z = mesh.positions[i * 3 + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
  let r = 0;
  for (let i = 0; i < n; i++) {
    const dx = mesh.positions[i * 3] - cx;
    const dy = mesh.positions[i * 3 + 1] - cy;
    const dz = mesh.positions[i * 3 + 2] - cz;
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (d > r) r = d;
  }
  return { cx, cy, cz, r };
}

/** 5 个视锥平面（世界空间，已归一化）。布局：每平面 4 个 float (a,b,c,d)。 */
export function frustumPlanes(view: Mat4, proj: Mat4): Float64Array {
  const vp = mat4Mul(proj, view);
  // 列主序的第 i 行:row(i) = [m[0*4+i], m[1*4+i], m[2*4+i], m[3*4+i]]
  const r0 = [vp[0], vp[4], vp[8], vp[12]];
  const r1 = [vp[1], vp[5], vp[9], vp[13]];
  const r2 = [vp[2], vp[6], vp[10], vp[14]];
  const r3 = [vp[3], vp[7], vp[11], vp[15]];
  const raw: number[][] = [
    [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]], // left
    [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]], // right
    [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]], // bottom
    [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]], // top
    [r3[0] + r2[0], r3[1] + r2[1], r3[2] + r2[2], r3[3] + r2[3]], // near
  ];
  const planes = new Float64Array(20);
  for (let p = 0; p < 5; p++) {
    const a = raw[p][0], b = raw[p][1], c = raw[p][2], d = raw[p][3];
    const l = Math.sqrt(a * a + b * b + c * c);
    if (l > 0) {
      planes[p * 4] = a / l;
      planes[p * 4 + 1] = b / l;
      planes[p * 4 + 2] = c / l;
      planes[p * 4 + 3] = d / l;
    } else {
      planes[p * 4] = 0; planes[p * 4 + 1] = 0; planes[p * 4 + 2] = 0; planes[p * 4 + 3] = 0;
    }
  }
  return planes;
}

/** 包围球经仿射 model 变换后的世界空间球（冻结公式）。 */
export function transformSphere(s: BoundingSphere, model: Mat4): BoundingSphere {
  const cx = model[0] * s.cx + model[4] * s.cy + model[8] * s.cz + model[12];
  const cy = model[1] * s.cx + model[5] * s.cy + model[9] * s.cz + model[13];
  const cz = model[2] * s.cx + model[6] * s.cy + model[10] * s.cz + model[14];
  // Frobenius 范数（3×3 部分）—— 线性部分最大奇异值的保守上界
  let f = 0;
  for (let i = 0; i < 12; i++) {
    const e = model[i];
    f += e * e;
  }
  return { cx, cy, cz, r: s.r * Math.sqrt(f) };
}

/** 球是否在任一平面外（dist < −r）→ 应剔除。 */
export function sphereOutsidePlanes(s: BoundingSphere, planes: Float64Array): boolean {
  for (let p = 0; p < planes.length / 4; p++) {
    const dist = planes[p * 4] * s.cx + planes[p * 4 + 1] * s.cy + planes[p * 4 + 2] * s.cz + planes[p * 4 + 3];
    if (dist < -s.r) return true;
  }
  return false;
}

/** 便捷入口:mesh + model 是否应被视锥剔除。 */
export function shouldCull(mesh: Mesh, model: Mat4, view: Mat4, proj: Mat4): boolean {
  const planes = frustumPlanes(view, proj);
  return sphereOutsidePlanes(transformSphere(meshBoundingSphere(mesh), model), planes);
}
