/**
 * 编辑器拾取（二期 M5）：屏幕像素 → 世界射线 → 网格三角形求交。
 *
 * 冻结公式：
 * - 屏幕射线（透视）：由相机基向量直接构造（避免矩阵求逆）。
 *   f = normalize(target - eye)；s = normalize(f × up)；u = s × f。
 *   ndcX = (px + 0.5)/W * 2 - 1；ndcY = 1 - (py + 0.5)/H * 2（屏幕 y 向下翻转）。
 *   tanH = tan(fovY/2)；dir = normalize(f + s·ndcX·tanH·aspect + u·ndcY·tanH)。
 * - 屏幕射线（正交）：origin = eye + s·ndcX·(right-left)/2 + u·ndcY·(top-bottom)/2，
 *   dir = f。
 * - 三角形求交：Möller–Trumbore（双面，返回 t > eps 的最近命中）。
 * - 球求交：解二次方程，返回最近正根（无交或负根 → null）。
 * - 场景遍历：与绘制同序（DFS 先序、visible 子树跳过）；同一 t 平局取遍历序靠前者。
 */

import {
  v3cross, v3normalize, v3sub, mat4ApplyPoint,
  type Mat4, type Vec3,
} from "../math.js";
import type { Camera, Mesh } from "../pipeline.js";
import { SceneNode, updateWorldMatrices } from "../scenegraph.js";

export interface Ray {
  origin: Vec3;
  /** 单位方向。 */
  dir: Vec3;
}

/** 相机基向量（与 mat4LookAt 同源公式）。 */
export function cameraBasis(camera: Camera): { f: Vec3; s: Vec3; u: Vec3 } {
  const f = v3normalize(v3sub(camera.target, camera.eye));
  const s = v3normalize(v3cross(f, camera.up));
  const u = v3cross(s, f);
  return { f, s, u };
}

/** 屏幕像素 → 世界射线（像素中心采样，与光栅化约定一致）。 */
export function screenRay(camera: Camera, fbW: number, fbH: number, px: number, py: number): Ray {
  const { f, s, u } = cameraBasis(camera);
  const ndcX = ((px + 0.5) / fbW) * 2 - 1;
  const ndcY = 1 - ((py + 0.5) / fbH) * 2;
  if (camera.projection === "ortho") {
    const halfW = ((camera.right ?? 1) - (camera.left ?? -1)) / 2;
    const halfH = ((camera.top ?? 1) - (camera.bottom ?? -1)) / 2;
    return {
      origin: {
        x: camera.eye.x + s.x * ndcX * halfW + u.x * ndcY * halfH,
        y: camera.eye.y + s.y * ndcX * halfW + u.y * ndcY * halfH,
        z: camera.eye.z + s.z * ndcX * halfW + u.z * ndcY * halfH,
      },
      dir: f,
    };
  }
  const tanH = Math.tan(camera.fovYRad / 2);
  const dx = f.x + s.x * ndcX * tanH * camera.aspect + u.x * ndcY * tanH;
  const dy = f.y + s.y * ndcX * tanH * camera.aspect + u.y * ndcY * tanH;
  const dz = f.z + s.z * ndcX * tanH * camera.aspect + u.z * ndcY * tanH;
  return { origin: { x: camera.eye.x, y: camera.eye.y, z: camera.eye.z }, dir: v3normalize({ x: dx, y: dy, z: dz }) };
}

/**
 * Möller–Trumbore 射线-三角形求交（双面）。返回 t（沿 dir 的距离）或 null。
 */
export function rayTriangle(o: Vec3, d: Vec3, p0: Vec3, p1: Vec3, p2: Vec3, eps = 1e-9): number | null {
  const e1x = p1.x - p0.x, e1y = p1.y - p0.y, e1z = p1.z - p0.z;
  const e2x = p2.x - p0.x, e2y = p2.y - p0.y, e2z = p2.z - p0.z;
  const px = d.y * e2z - d.z * e2y;
  const py = d.z * e2x - d.x * e2z;
  const pz = d.x * e2y - d.y * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  if (Math.abs(det) < 1e-12) return null;
  const inv = 1 / det;
  const tx = o.x - p0.x, ty = o.y - p0.y, tz = o.z - p0.z;
  const uu = (tx * px + ty * py + tz * pz) * inv;
  if (uu < -eps || uu > 1 + eps) return null;
  const qx = ty * e1z - tz * e1y;
  const qy = tz * e1x - tx * e1z;
  const qz = tx * e1y - ty * e1x;
  const vv = (d.x * qx + d.y * qy + d.z * qz) * inv;
  if (vv < -eps || uu + vv > 1 + eps) return null;
  const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
  return t > eps ? t : null;
}

/** 射线-球求交（最近正根，无交 → null）。 */
export function raySphere(o: Vec3, d: Vec3, c: Vec3, r: number): number | null {
  const ox = o.x - c.x, oy = o.y - c.y, oz = o.z - c.z;
  const b = ox * d.x + oy * d.y + oz * d.z;
  const cc = ox * ox + oy * oy + oz * oz - r * r;
  const disc = b * b - cc;
  if (disc < 0) return null;
  const sq = Math.sqrt(disc);
  const t0 = -b - sq;
  const t1 = -b + sq;
  if (t0 > 1e-9) return t0;
  if (t1 > 1e-9) return t1;
  return null;
}

/** 可绘制节点对（与绘制同序:DFS 先序、visible 子树跳过）。 */
export interface DrawablePair {
  node: SceneNode;
  model: Mat4;
}

function collectPairs(root: SceneNode, out: DrawablePair[]): void {
  if (!root.visible) return;
  if (root.mesh !== null && root.material !== null) {
    out.push({ node: root, model: root.world });
  }
  for (const child of root.children) {
    collectPairs(child, out);
  }
}

export interface PickHit {
  node: SceneNode;
  /** 沿射线的距离 t（世界单位）。 */
  t: number;
  /** 命中点的世界坐标。 */
  point: Vec3;
}

/**
 * 场景拾取：先序遍历可见网格项,逐三角形 Möller–Trumbore,返回最近命中。
 * 同一 t 平局取遍历序靠前者（冻结）。
 */
export function pickScene(root: SceneNode, camera: Camera, fbW: number, fbH: number, px: number, py: number): PickHit | null {
  updateWorldMatrices(root);
  const ray = screenRay(camera, fbW, fbH, px, py);
  const pairs: DrawablePair[] = [];
  collectPairs(root, pairs);
  let best: PickHit | null = null;
  for (const pair of pairs) {
    const mesh: Mesh = pair.node.mesh!;
    const tc = mesh.indices.length / 3;
    for (let t = 0; t < tc; t++) {
      const i0 = mesh.indices[t * 3], i1 = mesh.indices[t * 3 + 1], i2 = mesh.indices[t * 3 + 2];
      const p0 = mat4ApplyPoint(pair.model, { x: mesh.positions[i0 * 3], y: mesh.positions[i0 * 3 + 1], z: mesh.positions[i0 * 3 + 2] });
      const p1 = mat4ApplyPoint(pair.model, { x: mesh.positions[i1 * 3], y: mesh.positions[i1 * 3 + 1], z: mesh.positions[i1 * 3 + 2] });
      const p2 = mat4ApplyPoint(pair.model, { x: mesh.positions[i2 * 3], y: mesh.positions[i2 * 3 + 1], z: mesh.positions[i2 * 3 + 2] });
      const hit = rayTriangle(ray.origin, ray.dir, p0, p1, p2);
      if (hit !== null && (best === null || hit < best.t)) {
        best = {
          node: pair.node,
          t: hit,
          point: {
            x: ray.origin.x + ray.dir.x * hit,
            y: ray.origin.y + ray.dir.y * hit,
            z: ray.origin.z + ray.dir.z * hit,
          },
        };
      }
    }
  }
  return best;
}
