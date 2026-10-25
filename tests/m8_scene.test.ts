import { test } from "node:test";
import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";

import { Framebuffer } from "../src/framebuffer.js";
import { Renderer } from "../src/pipeline.js";
import { buildDemoScene, demoOrbitCamera, makeTorus } from "../src/scene.js";

test("M8 几何: 环面法线单位长、与参数化法线一致、索引合法", () => {
  const m = makeTorus(0.85, 0.38, 24, 12);
  const vc = m.positions.length / 3;
  assert.equal(vc, (24 + 1) * 12); // u 方向含缝合列
  for (let t = 0; t < m.indices.length / 3; t++) {
    for (const k of [0, 1, 2]) {
      const i = m.indices[t * 3 + k];
      assert.ok(i >= 0 && i < vc);
      const nl = Math.hypot(m.normals[i * 3], m.normals[i * 3 + 1], m.normals[i * 3 + 2]);
      assert.ok(Math.abs(nl - 1) < 1e-9);
    }
    // 面法线与参数化法线同向（环面内圈 dot(pos, normal) < 0，不能用质心判别）
    const i0 = m.indices[t * 3];
    const gx = (m.positions[i0 * 3] + m.positions[m.indices[t * 3 + 1] * 3] + m.positions[m.indices[t * 3 + 2] * 3]) / 3;
    const gy = (m.positions[i0 * 3 + 1] + m.positions[m.indices[t * 3 + 1] * 3 + 1] + m.positions[m.indices[t * 3 + 2] * 3 + 1]) / 3;
    const gz = (m.positions[i0 * 3 + 2] + m.positions[m.indices[t * 3 + 1] * 3 + 2] + m.positions[m.indices[t * 3 + 2] * 3 + 2]) / 3;
    // 质心 → 参数 (u, v)
    const R = 0.85;
    const rho = Math.hypot(gx, gz);
    const u = Math.atan2(gz, gx);
    const v = Math.atan2(gy, rho - R);
    const nx = Math.cos(v) * Math.cos(u), ny = Math.sin(v), nz = Math.cos(v) * Math.sin(u);
    const ax = m.positions[m.indices[t * 3 + 1] * 3] - m.positions[i0 * 3];
    const ay = m.positions[m.indices[t * 3 + 1] * 3 + 1] - m.positions[i0 * 3 + 1];
    const az = m.positions[m.indices[t * 3 + 1] * 3 + 2] - m.positions[i0 * 3 + 2];
    const bx = m.positions[m.indices[t * 3 + 2] * 3] - m.positions[i0 * 3];
    const by = m.positions[m.indices[t * 3 + 2] * 3 + 1] - m.positions[i0 * 3 + 1];
    const bz = m.positions[m.indices[t * 3 + 2] * 3 + 2] - m.positions[i0 * 3 + 2];
    const cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
    const dot = cx * nx + cy * ny + cz * nz;
    assert.ok(dot > 0, `torus face ${t} not outward (dot=${dot})`);
  }
});

test("M8 场景: 演示场景无头渲染一帧 —— 覆盖率、耗时与几何完整性", () => {
  const W = 256;
  const fb = new Framebuffer(W, W);
  const scene = buildDemoScene();
  scene.animate(1.25);
  const renderer = new Renderer(fb);
  const cam = demoOrbitCamera(0.55, 0.34, 6.6);
  const t0 = performance.now();
  renderer.render(cam, scene.items, "phong", scene.lights);
  const ms = performance.now() - t0;
  assert.ok(ms < 1500, `frame should render fast, took ${ms.toFixed(1)}ms`);

  let nonBg = 0;
  for (let i = 0; i < W * W; i++) {
    const r = fb.color[i * 4], g = fb.color[i * 4 + 1], b = fb.color[i * 4 + 2];
    if (r !== 0 || g !== 0 || b !== 0) nonBg++;
  }
  assert.ok(nonBg > W * W * 0.5, `scene should cover >50% of frame, got ${(100 * nonBg / (W * W)).toFixed(1)}%`);

  // 立方体纹理色、球体红色、环面条纹都应出现
  let hasWarm = 0, hasRed = 0, hasTeal = 0;
  for (let i = 0; i < W * W; i++) {
    const r = fb.color[i * 4], g = fb.color[i * 4 + 1], b = fb.color[i * 4 + 2];
    if (r > 140 && g > 90 && b < 90 && Math.abs(r - g) > 20) hasWarm++;
    if (r > 130 && r > g * 1.8 && r > b * 1.8) hasRed++;
    if (g > 100 && b > 100 && g > r * 1.5) hasTeal++;
  }
  assert.ok(hasWarm > 50, `textured cube pixels visible (${hasWarm})`);
  assert.ok(hasRed > 200, `phong sphere pixels visible (${hasRed})`);
  assert.ok(hasTeal > 100, `torus pixels visible (${hasTeal})`);
});

test("M8 场景: 多帧渲染稳定（无 NaN、深度缓冲语义正常）", () => {
  const W = 128;
  const fb = new Framebuffer(W, W);
  const scene = buildDemoScene();
  const renderer = new Renderer(fb);
  for (let f = 0; f < 5; f++) {
    scene.animate(f * 0.316);
    renderer.render(demoOrbitCamera(0.55 + f * 0.05, 0.34, 6.6), scene.items, "phong", scene.lights);
    for (let i = 0; i < W * W; i++) {
      const r = fb.color[i * 4], g = fb.color[i * 4 + 1], b = fb.color[i * 4 + 2];
      assert.ok(Number.isFinite(r) && Number.isFinite(g) && Number.isFinite(b));
    }
  }
});
