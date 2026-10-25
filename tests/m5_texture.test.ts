import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Framebuffer } from "../src/framebuffer.js";
import { Renderer, type Mesh, type Camera } from "../src/pipeline.js";
import { sampleNearest, sampleBilinear } from "../src/texture.js";
import { decodePPM, framebufferToRaw, type RawImage } from "../src/ppm.js";
import { diffRGB } from "../src/diff.js";
import { M5_QUAD, M5_W, M5_H, M5_FOV, projectVertex, makeM5Texture, renderM5 } from "../tools/reference/gen_m5.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "..", "..", "golden");

function loadGolden(name: string): RawImage {
  return decodePPM(new Uint8Array(readFileSync(join(GOLDEN_DIR, name))));
}

function quadMesh(): Mesh {
  const c = M5_QUAD.corners;
  return {
    positions: c.flatMap((p) => [p.x, p.y, p.z]),
    normals: c.flatMap(() => [0, 1, 0]),
    uvs: c.flatMap((p) => [p.u, p.v]),
    indices: [...M5_QUAD.indices],
  };
}

const CAM: Camera = {
  eye: { x: 0, y: 0, z: 0 }, target: { x: 0, y: 0, z: -1 }, up: { x: 0, y: 1, z: 0 },
  fovYRad: M5_FOV, aspect: 1, near: 0.1, far: 20,
};

function renderFormal(bilinear: boolean): Framebuffer {
  const fb = new Framebuffer(M5_W, M5_H);
  const t = makeM5Texture();
  const renderer = new Renderer(fb);
  renderer.render(CAM, [{
    mesh: quadMesh(),
    model: new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    material: {
      albedo: [1, 1, 1], specular: 0, shininess: 1,
      texture: { width: t.w, height: t.h, data: t.data },
      sampler: bilinear ? "bilinear" : "nearest",
      doubleSided: true,
    },
  }], "textured");
  return fb;
}

// ---------- 黄金对拍 ----------

test("M5 黄金: 正式透视校正管线（nearest）与参考实现逐像素一致", () => {
  const r = diffRGB(loadGolden("m5_tex_nearest.ppm"), framebufferToRaw(renderFormal(false)));
  assert.ok(r.ok, `nearest golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

test("M5 黄金: 正式透视校正管线（bilinear）与参考实现逐像素一致", () => {
  const r = diffRGB(loadGolden("m5_tex_bilinear.ppm"), framebufferToRaw(renderFormal(true)));
  assert.ok(r.ok, `bilinear golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

// ---------- 采样器单元 ----------

const T2 = {
  width: 2, height: 2,
  data: Uint8ClampedArray.from([
    255, 0, 0, 255, 0, 255, 0, 255,     // 行0: 红, 绿
    0, 0, 255, 255, 255, 255, 255, 255, // 行1: 蓝, 白
  ]),
};
const OUT = new Float64Array(3);

test("M5 采样: nearest 的格选择与 repeat 环绕", () => {
  sampleNearest(T2, 0.2, 0.2, OUT);
  assert.deepEqual([...OUT], [255, 0, 0]);
  sampleNearest(T2, 0.6, 0.2, OUT);
  assert.deepEqual([...OUT], [0, 255, 0]);
  sampleNearest(T2, 0.2, 0.6, OUT);
  assert.deepEqual([...OUT], [0, 0, 255]);
  sampleNearest(T2, 0.6, 0.6, OUT);
  assert.deepEqual([...OUT], [255, 255, 255]);
  sampleNearest(T2, 1.2, 0.2, OUT); // u=1.2 ≡ 0.2
  assert.deepEqual([...OUT], [255, 0, 0]);
  sampleNearest(T2, -0.3, 0.2, OUT); // u=-0.3 ≡ 0.7
  assert.deepEqual([...OUT], [0, 255, 0]);
});

test("M5 采样: bilinear 在纹素中心还原原色、在中心点为四纹素均值", () => {
  sampleBilinear(T2, 0.25, 0.25, OUT); // 红纹素中心
  assert.deepEqual([...OUT], [255, 0, 0]);
  sampleBilinear(T2, 0.5, 0.5, OUT);
  assert.ok(Math.abs(OUT[0] - 127.5) < 1e-12 && Math.abs(OUT[1] - 127.5) < 1e-12 && Math.abs(OUT[2] - 127.5) < 1e-12);
});

// ---------- 解析正确性（swimming 杀手）：光线-平面求交 vs 渲染输出 ----------

test("M5 解析: 深内部像素的渲染颜色 == 光线-平面求交解析 UV 的采样结果（逐位）", () => {
  const fb = renderFormal(false);
  const pv = M5_QUAD.corners.map(projectVertex);
  const idx = M5_QUAD.indices;
  const tris = [[pv[idx[0]], pv[idx[1]], pv[idx[2]]], [pv[idx[3]], pv[idx[4]], pv[idx[5]]]];
  const tanHalf = Math.tan(M5_FOV / 2);
  const tex = makeM5Texture();
  const expected = new Float64Array(3);
  let checked = 0;
  let maxULinearErr = 0;

  for (let j = 0; j < M5_H; j++) {
    for (let i = 0; i < M5_W; i++) {
      const ndcX = ((i + 0.5) / M5_W) * 2 - 1;
      const ndcY = 1 - ((j + 0.5) / M5_H) * 2;
      // 视空间光线（view = 恒等）：过像素中心
      const dx = ndcX * tanHalf, dy = ndcY * tanHalf;
      if (dy >= -1e-9) continue; // 打不到 y=-1 平面
      const t = -1 / dy;
      const wx = dx * t, wz = -t;
      if (wx < -2 || wx > 2 || wz < -8 || wz > -2) continue; // 落点在四边形外
      const uA = (wx + 2) / 4;
      const vA = (-wz - 2) / 6;

      // 落在纹素边界 ±1e-3 内的像素采样歧义（ULP 噪声可翻转格选择），跳过
      const fu = uA * tex.w, fv = vA * tex.h;
      const fuF = fu - Math.floor(fu), fvF = fv - Math.floor(fv);
      if (fuF < 1e-3 || fuF > 1 - 1e-3 || fvF < 1e-3 || fvF > 1 - 1e-3) continue;

      // 该像素深属于哪个三角形（min λ > 0.02 才比较，规避边界）
      for (const tri of tris) {
        const [a, b, c] = tri;
        const A2 = (b.sx - a.sx) * (c.sy - a.sy) - (b.sy - a.sy) * (c.sx - a.sx);
        const px = i + 0.5, py = j + 0.5;
        const w0 = (c.sx - b.sx) * (py - b.sy) - (c.sy - b.sy) * (px - b.sx);
        const w1 = (a.sx - c.sx) * (py - c.sy) - (a.sy - c.sy) * (px - c.sx);
        const w2 = (b.sx - a.sx) * (py - a.sy) - (b.sy - a.sy) * (px - a.sx);
        const l0 = w0 / A2, l1 = w1 / A2, l2 = w2 / A2;
        if (!(l0 > 0.02 && l1 > 0.02 && l2 > 0.02)) continue;

        // 屏幕空间线性 u（错误做法）与解析 u 的差距 —— 证明本场景对透视校正敏感
        const uL = l0 * (a.uOverW / a.invW) + l1 * (b.uOverW / b.invW) + l2 * (c.uOverW / c.invW);
        maxULinearErr = Math.max(maxULinearErr, Math.abs(uL - uA));

        sampleNearest({ width: tex.w, height: tex.h, data: tex.data }, uA, vA, expected);
        const o = (j * M5_W + i) * 4;
        assert.deepEqual(
          [fb.color[o], fb.color[o + 1], fb.color[o + 2]],
          [expected[0], expected[1], expected[2]],
          `pixel (${i},${j}) rendered != analytic (u=${uA}, v=${vA})`,
        );
        checked++;
        break;
      }
    }
  }
  assert.ok(checked > 2000, `should check plenty of pixels, got ${checked}`);
  assert.ok(maxULinearErr > 0.02, `screen-linear UV should deviate strongly here, max err ${maxULinearErr}`);
});

// ---------- 参考实现自检：与黄金图一致 ----------

test("M5 参考: 双采样器渲染与对应黄金图一致（夹具自检）", () => {
  assert.ok(diffRGB(loadGolden("m5_tex_nearest.ppm"), renderM5(false)).ok);
  assert.ok(diffRGB(loadGolden("m5_tex_bilinear.ppm"), renderM5(true)).ok);
});
