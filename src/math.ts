/**
 * 手写数学库：Vec2/Vec3/Vec4 + Mat4。
 * 约定：Mat4 为列主序 Float64Array(16)，元素 m[c*4 + r]；向量按列向量右乘：v' = M * v。
 * 坐标系：右手系，+Y 向上，相机看向 -Z（GL 惯例）。
 */

export interface Vec2 { x: number; y: number }
export interface Vec3 { x: number; y: number; z: number }
export interface Vec4 { x: number; y: number; z: number; w: number }

export const vec3 = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

export function v3add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}
export function v3sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}
export function v3scale(a: Vec3, s: number): Vec3 {
  return { x: a.x * s, y: a.y * s, z: a.z * s };
}
export function v3dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
export function v3cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}
export function v3length(a: Vec3): number {
  return Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}
export function v3normalize(a: Vec3): Vec3 {
  const l = v3length(a);
  return { x: a.x / l, y: a.y / l, z: a.z / l };
}
export function v3lerp(a: Vec3, b: Vec3, t: number): Vec3 {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    z: a.z + (b.z - a.z) * t,
  };
}

/** 列主序 4x4 矩阵，Float64Array(16)。 */
export type Mat4 = Float64Array;

export function mat4Identity(): Mat4 {
  const m = new Float64Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
}

/** 返回 a * b（先应用 b 再应用 a）。 */
export function mat4Mul(a: Mat4, b: Mat4): Mat4 {
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

/** 透视投影（gluPerspective 形式）：视锥 [near, far]，w_clip = -z_view，前方为正。 */
export function mat4Perspective(fovYRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovYRad / 2);
  const m = new Float64Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

/** 正交投影：w 恒为 1（屏幕空间线性 z 在此投影下是精确的）。 */
export function mat4Ortho(left: number, right: number, bottom: number, top: number, near: number, far: number): Mat4 {
  const m = new Float64Array(16);
  m[0] = 2 / (right - left);
  m[5] = 2 / (top - bottom);
  m[10] = -2 / (far - near);
  m[12] = -(right + left) / (right - left);
  m[13] = -(top + bottom) / (top - bottom);
  m[14] = -(far + near) / (far - near);
  m[15] = 1;
  return m;
}

/** 视图矩阵（右手系 lookAt）。 */
export function mat4LookAt(eye: Vec3, center: Vec3, up: Vec3): Mat4 {
  const f = v3normalize(v3sub(center, eye)); // 前向（指向目标）
  const s = v3normalize(v3cross(f, up));     // 右
  const u = v3cross(s, f);                   // 上
  const m = new Float64Array(16);
  m[0] = s.x; m[4] = s.y; m[8] = s.z;  m[12] = -v3dot(s, eye);
  m[1] = u.x; m[5] = u.y; m[9] = u.z;  m[13] = -v3dot(u, eye);
  m[2] = -f.x; m[6] = -f.y; m[10] = -f.z; m[14] = v3dot(f, eye);
  m[15] = 1;
  return m;
}

export function mat4RotX(rad: number): Mat4 {
  const c = Math.cos(rad), s = Math.sin(rad);
  const m = mat4Identity();
  m[5] = c; m[6] = s; m[9] = -s; m[10] = c;
  return m;
}
export function mat4RotY(rad: number): Mat4 {
  const c = Math.cos(rad), s = Math.sin(rad);
  const m = mat4Identity();
  m[0] = c; m[2] = -s; m[8] = s; m[10] = c;
  return m;
}
export function mat4RotZ(rad: number): Mat4 {
  const c = Math.cos(rad), s = Math.sin(rad);
  const m = mat4Identity();
  m[0] = c; m[1] = s; m[4] = -s; m[5] = c;
  return m;
}
export function mat4Translation(t: Vec3): Mat4 {
  const m = mat4Identity();
  m[12] = t.x; m[13] = t.y; m[14] = t.z;
  return m;
}
export function mat4Scaling(s: Vec3): Mat4 {
  const m = new Float64Array(16);
  m[0] = s.x; m[5] = s.y; m[10] = s.z; m[15] = 1;
  return m;
}

/** 变换四维向量。 */
export function mat4ApplyVec4(m: Mat4, v: Vec4): Vec4 {
  return {
    x: m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12] * v.w,
    y: m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13] * v.w,
    z: m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
    w: m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w,
  };
}

/** 变换点（w=1，含透视除法）。 */
export function mat4ApplyPoint(m: Mat4, v: Vec3): Vec3 {
  const p = mat4ApplyVec4(m, { x: v.x, y: v.y, z: v.z, w: 1 });
  return { x: p.x / p.w, y: p.y / p.w, z: p.z / p.w };
}

/** 变换方向（w=0，不平移、不除 w）。 */
export function mat4ApplyDir(m: Mat4, v: Vec3): Vec3 {
  return {
    x: m[0] * v.x + m[4] * v.y + m[8] * v.z,
    y: m[1] * v.x + m[5] * v.y + m[9] * v.z,
    z: m[2] * v.x + m[6] * v.y + m[10] * v.z,
  };
}
