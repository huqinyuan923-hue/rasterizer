/**
 * Gizmo 数学（二期 M5）：移动箭头与旋转环的投影、拾取、拖拽增量。
 *
 * 冻结约定：
 * - 轴序 0=X(红)、1=Y(绿)、2=Z(蓝)；旋转环 i 躺在垂直于轴 i 的平面（X 环 = YZ 平面）。
 * - 轴长 GIZMO_LEN（世界单位）；屏幕拾取阈值 GIZMO_PICK_PX（像素）。
 * - 点投影：view·proj 后按光栅器冻结公式映射屏幕
 *   （sx = (x/w+1)*0.5*W；sy = (1-y/w)*0.5*H）；w <= 1e-6 标记 behind（不参与拾取）。
 * - 移动拾取：鼠标点到投影轴段（origin → origin+axis·LEN）的屏幕距离 < 阈值，
 *   取最近者；同时命中的轴按"距鼠标最近"胜出。
 * - 移动拖拽：起止两帧的鼠标射线与"过 origin、法线 = 相机前向 f"的平面求交，
 *   增量 = dot(pCur - pStart, axisUnit)。
 * - 旋转拖拽：起止射线与环平面（过 center、法线 = 旋转轴）求交，
 *   增量角 = signedAngle(pStart - center → pCur - center, 轴向)（弧度）。
 */

import { v3cross, v3dot, v3sub, mat4ApplyVec4, type Vec3 } from "../math.js";
import { mat4LookAt, mat4Perspective, mat4Ortho, type Mat4 } from "../math.js";
import type { Camera } from "../pipeline.js";
import { cameraBasis, screenRay, type Ray } from "./picking.js";

export const GIZMO_LEN = 1.2;
export const GIZMO_PICK_PX = 14;

export const AXIS_VECTORS: [Vec3, Vec3, Vec3] = [
  { x: 1, y: 0, z: 0 },
  { x: 0, y: 1, z: 0 },
  { x: 0, y: 0, z: 1 },
];

/** view·proj 组合矩阵（与渲染管线同一构造）。 */
export function viewProj(camera: Camera): Mat4 {
  const view = mat4LookAt(camera.eye, camera.target, camera.up);
  const proj = camera.projection === "ortho"
    ? mat4Ortho(camera.left ?? -1, camera.right ?? 1, camera.bottom ?? -1, camera.top ?? 1, camera.near, camera.far)
    : mat4Perspective(camera.fovYRad, camera.aspect, camera.near, camera.far);
  // mat4Mul(proj, view) —— 与 culling.ts 同式
  return mulMat(proj, view);
}

function mulMat(a: Mat4, b: Mat4): Mat4 {
  const o = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[r] * b[c * 4] +
        a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] +
        a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

export interface ScreenPoint {
  x: number;
  y: number;
  behind: boolean;
}

/** 世界点 → 屏幕像素（冻结公式；behind = w <= 1e-6）。 */
export function projectPoint(camera: Camera, fbW: number, fbH: number, p: Vec3): ScreenPoint {
  const vp = viewProj(camera);
  const clip = mat4ApplyVec4(vp, { x: p.x, y: p.y, z: p.z, w: 1 });
  if (clip.w <= 1e-6) {
    return { x: NaN, y: NaN, behind: true };
  }
  return {
    x: (clip.x / clip.w + 1) * 0.5 * fbW,
    y: (1 - clip.y / clip.w) * 0.5 * fbH,
    behind: false,
  };
}

/** 点到线段（屏幕空间）的距离平方。 */
function distToSegmentSq(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  let t = 0;
  if (lenSq > 0) {
    t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
    if (t < 0) t = 0;
    if (t > 1) t = 1;
  }
  const cx = ax + dx * t, cy = ay + dy * t;
  return (px - cx) * (px - cx) + (py - cy) * (py - cy);
}

/**
 * 拾取移动轴：返回被选轴（0/1/2）或 null。
 * 规则：轴段在前（两端都不 behind）且屏幕距离 < GIZMO_PICK_PX，取最近者。
 */
export function pickGizmoMove(origin: Vec3, camera: Camera, fbW: number, fbH: number, px: number, py: number): 0 | 1 | 2 | null {
  let best: 0 | 1 | 2 | null = null;
  let bestDist = GIZMO_PICK_PX * GIZMO_PICK_PX;
  for (let axis = 0 as 0 | 1 | 2; axis < 3; axis = (axis + 1) as 0 | 1 | 2) {
    const dir = AXIS_VECTORS[axis];
    const a = projectPoint(camera, fbW, fbH, origin);
    const b = projectPoint(camera, fbW, fbH, { x: origin.x + dir.x * GIZMO_LEN, y: origin.y + dir.y * GIZMO_LEN, z: origin.z + dir.z * GIZMO_LEN });
    if (a.behind || b.behind) continue;
    const d = distToSegmentSq(px, py, a.x, a.y, b.x, b.y);
    if (d < bestDist) {
      bestDist = d;
      best = axis;
    }
  }
  return best;
}

/** 射线与平面（过 point、法线 unit n）求交。返回 t 或 null（射线平行/背向）。 */
export function rayPlane(o: Vec3, d: Vec3, point: Vec3, n: Vec3, eps = 1e-9): number | null {
  const denom = d.x * n.x + d.y * n.y + d.z * n.z;
  if (Math.abs(denom) < eps) return null;
  const t = ((point.x - o.x) * n.x + (point.y - o.y) * n.y + (point.z - o.z) * n.z) / denom;
  return t > eps ? t : null;
}

function rayAt(ray: Ray, t: number): Vec3 {
  return { x: ray.origin.x + ray.dir.x * t, y: ray.origin.y + ray.dir.y * t, z: ray.origin.z + ray.dir.z * t };
}

/**
 * 移动拖拽：起止像素 → 射线 → 相机面向平面求交 → 沿轴投影增量（世界单位）。
 * 任一端无交 → null（拖拽无效）。
 */
export function dragMoveDelta(
  axis: 0 | 1 | 2,
  origin: Vec3,
  camera: Camera,
  fbW: number, fbH: number,
  px0: number, py0: number, px1: number, py1: number,
): Vec3 | null {
  const { f } = cameraBasis(camera);
  const r0 = screenRay(camera, fbW, fbH, px0, py0);
  const r1 = screenRay(camera, fbW, fbH, px1, py1);
  const t0 = rayPlane(r0.origin, r0.dir, origin, f);
  const t1 = rayPlane(r1.origin, r1.dir, origin, f);
  if (t0 === null || t1 === null) return null;
  const p0 = rayAt(r0, t0);
  const p1 = rayAt(r1, t1);
  const axisUnit = AXIS_VECTORS[axis];
  const delta = (p1.x - p0.x) * axisUnit.x + (p1.y - p0.y) * axisUnit.y + (p1.z - p0.z) * axisUnit.z;
  return { x: axisUnit.x * delta, y: axisUnit.y * delta, z: axisUnit.z * delta };
}

/** 旋转环采样点数（屏幕距离判定的折线近似，冻结）。 */
const RING_SAMPLES = 48;

/** 环 i（轴 i）上参数 θ ∈ [0,1) 的点：axis 0 → YZ 平面,1 → XZ,2 → XY。 */
function ringPoint(axis: 0 | 1 | 2, center: Vec3, radius: number, theta: number): Vec3 {
  const a = Math.cos(theta * 2 * Math.PI) * radius;
  const b = Math.sin(theta * 2 * Math.PI) * radius;
  if (axis === 0) return { x: center.x, y: center.y + a, z: center.z + b };       // YZ
  if (axis === 1) return { x: center.x + a, y: center.y, z: center.z + b };       // XZ
  return { x: center.x + a, y: center.y + b, z: center.z };                        // XY
}

export const RING_RADIUS = 1.0;

/**
 * 拾取旋转环：环折线各采样段的屏幕距离最小者且 < 阈值。返回轴（0/1/2）或 null。
 */
export function pickGizmoRotate(center: Vec3, camera: Camera, fbW: number, fbH: number, px: number, py: number): 0 | 1 | 2 | null {
  let best: 0 | 1 | 2 | null = null;
  let bestDist = GIZMO_PICK_PX * GIZMO_PICK_PX;
  for (let axis = 0 as 0 | 1 | 2; axis < 3; axis = (axis + 1) as 0 | 1 | 2) {
    let prev = projectPoint(camera, fbW, fbH, ringPoint(axis, center, RING_RADIUS, 0));
    for (let i = 1; i <= RING_SAMPLES; i++) {
      const cur = projectPoint(camera, fbW, fbH, ringPoint(axis, center, RING_RADIUS, i / RING_SAMPLES));
      if (!prev.behind && !cur.behind) {
        const d = distToSegmentSq(px, py, prev.x, prev.y, cur.x, cur.y);
        if (d < bestDist) {
          bestDist = d;
          best = axis;
        }
      }
      prev = cur;
    }
  }
  return best;
}

/** 有符号角：a 绕 n 转到 b 的角度（弧度,右手系,冻结公式）。 */
export function signedAngle(a: Vec3, b: Vec3, n: Vec3): number {
  const axb = v3cross(a, b);
  return Math.atan2(v3dot(axb, n), v3dot(a, b));
}

/**
 * 旋转拖拽：起止射线与环平面（过 center、法线 = 轴）求交 → 有符号角增量（弧度）。
 * 任一端无交或交点距中心过近 → null。
 */
export function dragRotateDelta(
  axis: 0 | 1 | 2,
  center: Vec3,
  camera: Camera,
  fbW: number, fbH: number,
  px0: number, py0: number, px1: number, py1: number,
): number | null {
  const n = AXIS_VECTORS[axis];
  const r0 = screenRay(camera, fbW, fbH, px0, py0);
  const r1 = screenRay(camera, fbW, fbH, px1, py1);
  const t0 = rayPlane(r0.origin, r0.dir, center, n);
  const t1 = rayPlane(r1.origin, r1.dir, center, n);
  if (t0 === null || t1 === null) return null;
  const p0 = rayAt(r0, t0);
  const p1 = rayAt(r1, t1);
  const a = v3sub(p0, center);
  const b = v3sub(p1, center);
  const la = Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
  const lb = Math.sqrt(b.x * b.x + b.y * b.y + b.z * b.z);
  if (la < 1e-6 || lb < 1e-6) return null;
  return signedAngle(a, b, n);
}
