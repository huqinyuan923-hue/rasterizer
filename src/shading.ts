/**
 * 光照与着色（Lambert 漫反射 + Blinn-Phong 高光）—— 二期升级为多光源数组。
 *
 * 公式冻结（参考实现逐位复刻）：
 *   c = albedo⊙ambient + Σ_light [ albedo⊙lightColor·max(N·L,0)·atten·cone + lightColor·specular·max(N·H,0)^shininess·atten·cone ]
 *   V = normalize(eye - P)；H = normalize(L + V)，ndh = (N·(L+V))/|L+V|（先点积后除模长）
 *   方向光：L = dir（调用方保证已归一化、指向光源），atten = 1
 *   点光：L = normalize(pos - P)，atten = 1/(1 + k1·dist + k2·dist²)
 *   聚光：同点光衰减，另有锥形因子
 *     cosΘ = -(L·axis)（axis 为聚光轴，指向照射方向，此处按冻结公式 sqrt 归一化）
 *     恰在/超出外锥（cosΘ ≤ cosOuter）→ 无贡献；内锥内（cosΘ ≥ cosInner）→ 全亮；
 *     之间 t = (cosΘ - cosOuter)/(cosInner - cosOuter) 截断到 (0,1]；
 *     cosInner ≤ cosOuter（退化输入）按硬锥处理（in-cone 即全亮）；
 *     axis 为零向量 → 该光源无贡献。
 *
 * 累积顺序冻结：ambient → dirs（数组序）→ points（数组序）→ spots（数组序）。
 * 单 dir + 单 point 的场景与一期实现（ambient→dir→point）的浮点运算序列完全一致。
 *
 * Lambert 模式（specOn = false）：跳过全部高光项，只保留环境 + 漫反射。
 *
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

/** 聚光灯：点光衰减 + 锥形因子。 */
export interface SpotLight {
  pos: Vec3;
  /** 聚光轴：指向照射方向（渲染时按冻结公式归一化；零向量 → 无贡献）。 */
  dir: [number, number, number];
  color: [number, number, number];
  /** 内锥半角余弦（全亮边界），应大于 cosOuter。 */
  cosInner: number;
  /** 外锥半角余弦（外暗边界）。 */
  cosOuter: number;
  k1: number;
  k2: number;
}

export interface LightSetup {
  ambient: [number, number, number];
  dirs: DirectionalLight[];
  points: PointLight[];
  spots: SpotLight[];
}

export const NO_LIGHTS: LightSetup = { ambient: [0, 0, 0], dirs: [], points: [], spots: [] };

/** 旧版可空单光源形态（{dir, point}）→ 数组形态。迁移与测试夹具适配用。 */
export function legacyLights(
  ambient: [number, number, number],
  dir: DirectionalLight | null,
  point: PointLight | null,
): LightSetup {
  return { ambient, dirs: dir !== null ? [dir] : [], points: point !== null ? [point] : [], spots: [] };
}

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
  specOn: boolean = true,
): void {
  let vx = eye.x - px, vy = eye.y - py, vz = eye.z - pz;
  let vl = Math.sqrt(vx * vx + vy * vy + vz * vz);
  if (vl === 0) vl = 1;
  vx /= vl; vy /= vl; vz /= vl;

  let r = ar * setup.ambient[0];
  let g = ag * setup.ambient[1];
  let b = ab * setup.ambient[2];

  for (const dir of setup.dirs) {
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
            const s = specular * powInt(ndh, shininess);
            r += dir.color[0] * s;
            g += dir.color[1] * s;
            b += dir.color[2] * s;
          }
        }
      }
    }
  }

  for (const pt of setup.points) {
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
              const s = specular * powInt(ndh, shininess);
              r += pt.color[0] * s * atten;
              g += pt.color[1] * s * atten;
              b += pt.color[2] * s * atten;
            }
          }
        }
      }
    }
  }

  for (const sp of setup.spots) {
    const d0 = sp.dir[0], d1 = sp.dir[1], d2 = sp.dir[2];
    const dl = Math.sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    if (dl === 0) continue; // 冻结：零轴 → 无贡献
    const ax = d0 / dl, ay = d1 / dl, az = d2 / dl;
    const ldx = sp.pos.x - px, ldy = sp.pos.y - py, ldz = sp.pos.z - pz;
    const dist = Math.sqrt(ldx * ldx + ldy * ldy + ldz * ldz);
    if (dist > 0) {
      const lx = ldx / dist, ly = ldy / dist, lz = ldz / dist;
      const cosTheta = -(lx * ax + ly * ay + lz * az);
      if (cosTheta > sp.cosOuter) {
        const denom = sp.cosInner - sp.cosOuter;
        // 冻结：denom > 0 软锥（截断到 (0,1]），否则硬锥（in-cone 全亮）
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
                const s = specular * powInt(ndh, shininess);
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

  out[0] = r * 255;
  out[1] = g * 255;
  out[2] = b * 255;
}
