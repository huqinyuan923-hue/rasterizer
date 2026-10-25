/**
 * 近平面裁剪：对 clip 空间半空间 w >= eps 做 Sutherland–Hodgman。
 * 三角形输入 → 顶点数 0/3/4 的多边形输出 → 扇形化后恰为 0/1/2 个三角形。
 *
 * 冻结公式（参考实现逐位复刻）：
 *   距离 dist(v) = v.w - eps；交点参数 t = dist(a) / (dist(a) - dist(b))
 *   分量插值 out = a + (b - a) * t（clip 位置、视距 d、payload 全部同式）
 * 顶点恰在平面上（dist == 0）按内部处理。
 */

export interface ClipVert {
  x: number; y: number; z: number; w: number; // clip 空间位置
  d: number;                                  // 视距 d = -z_view
  a: Float64Array;                            // varying payload（clip 空间仿射，可线性插值）
}

// 快路径复用的输出数组（消费方同步使用，无重入）
const FAST_TRIPLE: ClipVert[] = new Array(3);

export function clipTriangleNear(v0: ClipVert, v1: ClipVert, v2: ClipVert, eps: number): ClipVert[] {
  const d0 = v0.w - eps, d1 = v1.w - eps, d2 = v2.w - eps;
  // 快路径：全部在近面前（原样返回，保持顶点序）
  if (d0 >= 0 && d1 >= 0 && d2 >= 0) {
    FAST_TRIPLE[0] = v0; FAST_TRIPLE[1] = v1; FAST_TRIPLE[2] = v2;
    return FAST_TRIPLE;
  }
  if (d0 < 0 && d1 < 0 && d2 < 0) return [];

  const verts = [v0, v1, v2];
  const dists = [d0, d1, d2];
  const out: ClipVert[] = [];
  for (let i = 0; i < 3; i++) {
    const cur = verts[i], nxt = verts[(i + 1) % 3];
    const dc = dists[i], dn = dists[(i + 1) % 3];
    if (dc >= 0) {
      out.push(cur);
      if (dn < 0) out.push(lerpVert(cur, nxt, dc / (dc - dn)));
    } else if (dn >= 0) {
      out.push(lerpVert(cur, nxt, dc / (dc - dn)));
    }
  }
  return out;
}

function lerpVert(a: ClipVert, b: ClipVert, t: number): ClipVert {
  const L = (x: number, y: number): number => x + (y - x) * t;
  const n = a.a.length;
  const payload = new Float64Array(n);
  for (let k = 0; k < n; k++) payload[k] = L(a.a[k], b.a[k]);
  return {
    x: L(a.x, b.x),
    y: L(a.y, b.y),
    z: L(a.z, b.z),
    w: L(a.w, b.w),
    d: L(a.d, b.d),
    a: payload,
  };
}
