/**
 * OBJ 模型加载（二期 M3）。零依赖、确定性解析。
 *
 * 冻结规则（参考实现逐行为复刻，黄金图对拍口径）：
 * - 逐行解析（\n 分割，容忍 \r）；token 按空白切分；`#` 开头为注释行，跳过。
 * - `v x y z [w]`：顶点，w 忽略。`vt u v [w]`：纹理坐标，w 忽略。`vn x y z`：法线（按原值存储）。
 * - `f ...`：面。索引形态四选一：`v`、`v/vt`、`v/vt/vn`、`v//vn`。
 *   索引 1 基；负索引 = 相对**原始 v/vt/vn 指令列表**末尾（-1 为最后一个"v"指令定义的
 *   顶点；f 行自身展开的角点不计入相对基准 —— OBJ 规范语义）。多边形按扇形三角化：(0, k, k+1)。
 * - 缺 vt 的顶点 → uv (0,0)（不延续"最后使用值"）；缺 vn 的三角形 → 该三角形的几何法线
 *   （normalize(cross(p1-p0, p2-p0))，逐三角形独立计算 → 平面着色语义；退化面兜底 (0,1,0)）。
 * - 其他指令（o/g/s/usemtl/mtllib/l/p/vp…）一律忽略；不支持 MTL 材质库（材质由调用方给）。
 * - 顶点索引越界 → 抛 Error（快速失败，不静默吞）。
 * - 顶点数 < 3 的 f 行跳过。
 */

import type { Mesh } from "./pipeline.js";

/** OBJ 顶点记录（解析中间态）：pos/uv/normal 各自独立索引。 */
interface ObjFaceCorner {
  v: number;  // 顶点索引（已转 0 基）
  vt: number; // 纹理坐标索引（已转 0 基）；-1 = 无
  vn: number; // 法线索引（已转 0 基）；-1 = 无
}

function parseCorner(token: string, posCount: number, uvCount: number, nCount: number): ObjFaceCorner {
  const parts = token.split("/");
  const v = Number(parts[0]);
  if (!Number.isFinite(v) || parts[0] === "") throw new Error(`OBJ: bad face corner "${token}"`);
  const vi = v > 0 ? v - 1 : posCount + v;
  if (vi < 0 || vi >= posCount) throw new Error(`OBJ: vertex index out of range (${v} of ${posCount})`);
  let vt = -1;
  if (parts.length >= 2 && parts[1] !== "") {
    const t = Number(parts[1]);
    if (!Number.isFinite(t)) throw new Error(`OBJ: bad vt index in "${token}"`);
    vt = t > 0 ? t - 1 : uvCount + t;
    if (vt < 0 || vt >= uvCount) throw new Error(`OBJ: vt index out of range (${t} of ${uvCount})`);
  }
  let vn = -1;
  if (parts.length >= 3 && parts[2] !== "") {
    const n = Number(parts[2]);
    if (!Number.isFinite(n)) throw new Error(`OBJ: bad vn index in "${token}"`);
    vn = n > 0 ? n - 1 : nCount + n;
    if (vn < 0 || vn >= nCount) throw new Error(`OBJ: vn index out of range (${n} of ${nCount})`);
  }
  return { v: vi, vt, vn };
}

/**
 * 解析 OBJ 文本。返回 Mesh（positions/normals/uvs/indices 四数组）。
 *
 * 布局冻结：输出数组 = f 行展开角点块（逐角点展开,扇形三角化）；indices 直接指向该块。
 * 负索引语义冻结：相对**原始 v/vt/vn 指令列表**（OBJ 规范"最后定义的顶点"），
 * 展开角点不计入相对基准。
 */
export function parseOBJ(text: string): Mesh {
  const rawPos: number[] = [];
  const rawUV: number[] = [];
  const rawNrm: number[] = [];
  const expPos: number[] = [];
  const expUV: number[] = [];
  const expNrm: number[] = [];
  const expIdx: number[] = [];

  const lines = text.split("\n");
  for (const raw of lines) {
    const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
    if (line.length === 0 || line.charCodeAt(0) === 0x23 /* # */) continue;
    const tokens = line.trim().split(/\s+/);
    const head = tokens[0];
    if (head === "v") {
      if (tokens.length < 4) throw new Error("OBJ: v line needs 3 coordinates");
      const x = Number(tokens[1]), y = Number(tokens[2]), z = Number(tokens[3]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        throw new Error(`OBJ: bad vertex "${line}"`);
      }
      rawPos.push(x, y, z);
    } else if (head === "vt") {
      if (tokens.length < 3) throw new Error("OBJ: vt line needs 2 coordinates");
      const u = Number(tokens[1]), v = Number(tokens[2]);
      if (!Number.isFinite(u) || !Number.isFinite(v)) throw new Error(`OBJ: bad uv "${line}"`);
      rawUV.push(u, v);
    } else if (head === "vn") {
      if (tokens.length < 4) throw new Error("OBJ: vn line needs 3 coordinates");
      const x = Number(tokens[1]), y = Number(tokens[2]), z = Number(tokens[3]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        throw new Error(`OBJ: bad normal "${line}"`);
      }
      rawNrm.push(x, y, z);
    } else if (head === "f") {
      if (tokens.length < 4) continue; // 少于 3 顶点的面：跳过
      const posCount = rawPos.length / 3;
      const uvCount = rawUV.length / 2;
      const nCount = rawNrm.length / 3;
      const corners: ObjFaceCorner[] = [];
      for (let k = 1; k < tokens.length; k++) {
        corners.push(parseCorner(tokens[k], posCount, uvCount, nCount));
      }
      // 扇形三角化；每个角点展开为独立顶点（OBJ 展开语义）
      for (let k = 1; k + 1 < corners.length; k++) {
        const tri = [corners[0], corners[k], corners[k + 1]];
        const p = tri.map((c) => ({
          x: rawPos[c.v * 3], y: rawPos[c.v * 3 + 1], z: rawPos[c.v * 3 + 2],
        }));
        // 缺 vn 的三角形 → 几何法线（冻结兜底：退化面取 (0,1,0)）
        let gnx = 0, gny = 1, gnz = 0;
        if (tri.some((c) => c.vn < 0)) {
          const ax = p[1].x - p[0].x, ay = p[1].y - p[0].y, az = p[1].z - p[0].z;
          const bx = p[2].x - p[0].x, by = p[2].y - p[0].y, bz = p[2].z - p[0].z;
          const cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
          const l = Math.sqrt(cx * cx + cy * cy + cz * cz);
          if (l > 0) { gnx = cx / l; gny = cy / l; gnz = cz / l; }
        }
        const base = expPos.length / 3;
        for (let m = 0; m < 3; m++) {
          const c = tri[m];
          expPos.push(p[m].x, p[m].y, p[m].z);
          if (c.vt >= 0) {
            expUV.push(rawUV[c.vt * 2], rawUV[c.vt * 2 + 1]);
          } else {
            expUV.push(0, 0); // 缺 vt → (0,0)
          }
          if (c.vn >= 0) {
            expNrm.push(rawNrm[c.vn * 3], rawNrm[c.vn * 3 + 1], rawNrm[c.vn * 3 + 2]);
          } else {
            expNrm.push(gnx, gny, gnz); // 缺 vn → 该三角形几何法线
          }
        }
        expIdx.push(base, base + 1, base + 2);
      }
    }
    // 其他指令一律忽略
  }

  // 输出 = 展开块（渲染所需的最小闭合网格；indices 直接指向展开块）
  return { positions: expPos, normals: expNrm, uvs: expUV, indices: expIdx };
}
