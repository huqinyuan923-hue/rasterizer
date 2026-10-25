/**
 * 光照与着色（Lambert 漫反射 + Blinn-Phong 高光）。
 * 公式冻结（参考实现逐位复刻）：
 *   c = albedo⊙ambient + Σ_light [ albedo⊙lightColor·max(N·L,0)·atten + lightColor·specular·max(N·H,0)^shininess·atten ]
 *   V = normalize(eye - P)；H = normalize(L + V)，ndh = (N·(L+V))/|L+V|（先点积后除模长）
 *   方向光：L = dir（已归一化、指向光源），atten = 1
 *   点光：L = normalize(pos - P)，atten = 1/(1 + k1·dist + k2·dist²)
 * 输出为 0..255 的 float（量化交由 Uint8ClampedArray 写入完成）。
 */

import type { Vec3 } from "./math.js";

export interface DirectionalLight {
  /** 指向光源的单位方向。 */
  dir: [number, number, number];
  color: [number, number, number];
}

export interface PointLight {
  pos: Vec3;
  color: [number, number, number];
  k1: number;
  k2: number;
}

export interface LightSetup {
  ambient: [number, number, number];
  dir: DirectionalLight | null;
  point: PointLight | null;
}

export const NO_LIGHTS: LightSetup = { ambient: [0, 0, 0], dir: null, point: null };

/**
 * 整数指数平方链。与 Math.pow 在 IEEE 上不逐位等价，但 8bit 量化后与
 * 黄金图（参考实现用 Math.pow）无差异 —— 黄金对拍为仲裁（见 DEVLOG M8）。
 * 非整数指数回退 Math.pow。
 */
export function powInt(x: number, n: number): number {
  if (n !== (n | 0) || n < 0) return Math.pow(x, n);
  let result = 1;
  let base = x;
  let e = n | 0;
  while (e > 0) {
    if (e & 1) result *= base;
    base *= base;
    e >>>= 1;
  }
  return result;
}

export function shadePhongPixel(
  nx: number, ny: number, nz: number,
  px: number, py: number, pz: number,
  eye: Vec3,
  setup: LightSetup,
  ar: number, ag: number, ab: number,
  specular: number,
  shininess: number,
  out: Float64Array,
): void {
  let vx = eye.x - px, vy = eye.y - py, vz = eye.z - pz;
  let vl = Math.sqrt(vx * vx + vy * vy + vz * vz);
  if (vl === 0) vl = 1;
  vx /= vl; vy /= vl; vz /= vl;

  let r = ar * setup.ambient[0];
  let g = ag * setup.ambient[1];
  let b = ab * setup.ambient[2];

  const dir = setup.dir;
  if (dir !== null) {
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
          const s = specular * powInt(ndh, shininess);
          r += dir.color[0] * s;
          g += dir.color[1] * s;
          b += dir.color[2] * s;
        }
      }
    }
  }

  const pt = setup.point;
  if (pt !== null) {
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
            const s = specular * powInt(ndh, shininess);
            r += pt.color[0] * s * atten;
            g += pt.color[1] * s * atten;
            b += pt.color[2] * s * atten;
          }
        }
      }
    }
  }

  out[0] = r * 255;
  out[1] = g * 255;
  out[2] = b * 255;
}
