/**
 * Benchmark：512² 帧缓冲下的三角形吞吐量。
 * 两个口径：
 *  1) 屏幕空间纯填充（fillTriangleDepth，深度开）：随机种子三角形流；
 *  2) 完整管线（变换+裁剪+透视校正+Phong）：演示场景网格按帧渲染。
 * 用法: npm run bench
 */

import { Framebuffer } from "../src/framebuffer.js";
import { fillTriangleDepth } from "../src/raster.js";
import { Renderer, type DrawItem } from "../src/pipeline.js";
import { makeSphere } from "../src/scene.js";
import { mulberry32 } from "../src/rng.js";
import { mat4Identity } from "../src/math.js";

const W = 512;
const H = 512;
const WARMUP = 20;
const MEASURE_MS = 1500;

function runFor(fn: () => number, perCallUnits: number): { opsPerSec: number; unitsPerSec: number; msPerOp: number } {
  for (let i = 0; i < WARMUP; i++) fn();
  const t0 = performance.now();
  let ops = 0;
  while (performance.now() - t0 < MEASURE_MS) {
    fn();
    ops++;
  }
  const ms = performance.now() - t0;
  return {
    opsPerSec: ops / (ms / 1000),
    unitsPerSec: (ops * perCallUnits) / (ms / 1000),
    msPerOp: ms / ops,
  };
}

function fmt(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : n.toFixed(0);
}

console.log(`rasterizer benchmark — ${W}x${H} framebuffer, node ${process.version}\n`);

// ---- 1) 屏幕空间纯填充吞吐（三角形/秒） ----
{
  const fb = new Framebuffer(W, H);
  const rnd = mulberry32(42);
  // 一次调用渲染 256 个中小三角形（50~140px），混合深浅
  const tris: number[][] = [];
  for (let i = 0; i < 256; i++) {
    const cx = rnd() * W, cy = rnd() * H, s = 25 + rnd() * 45;
    tris.push([cx, cy - s, cx + s * 0.9, cy + s * 0.7, cx - s * 0.9, cy + s * 0.6, 2 + rnd() * 6]);
  }
  const res = runFor(() => {
    fb.clear(30, 30, 40);
    let drawn = 0;
    for (const t of tris) {
      drawn += fillTriangleDepth(fb, t[0], t[1], t[6], t[2], t[3], t[6] + 2, t[4], t[5], t[6] + 4, 200, 90, 60);
    }
    return drawn;
  }, 256);
  console.log(`[1] fillTriangleDepth（屏幕空间、开深度、含 clear；中大型三角形，像素填充主导）`);
  console.log(`    ${fmt(res.unitsPerSec)} tris/s   ${res.msPerOp.toFixed(2)} ms/256tris\n`);
}

// ---- 2) 完整管线吞吐（变换 + 裁剪 + 透视校正 + Phong + 深度） ----
{
  const fb = new Framebuffer(W, H);
  const renderer = new Renderer(fb);
  const mesh = makeSphere(1, 48, 24); // 2304 三角形
  const item: DrawItem = { mesh, model: mat4Identity(), material: { albedo: [0.85, 0.3, 0.22], specular: 0.6, shininess: 48 } };
  const cam = { eye: { x: 0, y: 0.25, z: 4.5 }, target: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 }, fovYRad: 55 * Math.PI / 180, aspect: 1, near: 0.1, far: 60 };
  const lights = { ambient: [0.12, 0.12, 0.14] as [number, number, number], dirs: [{ dir: [0.45, 0.8, 0.5] as [number, number, number], color: [0.85, 0.82, 0.75] as [number, number, number] }], points: [], spots: [] };
  const triCount = mesh.indices.length / 3;
  const res = runFor(() => {
    renderer.render(cam, [item], "phong", lights);
    return triCount;
  }, triCount);
  console.log(`[2] 完整管线 Phong（UV 球 48x24 = ${triCount} tris，占屏约一半）`);
  console.log(`    ${fmt(res.unitsPerSec)} tris/s   ${res.msPerOp.toFixed(2)} ms/frame   ${((W * H) / (res.msPerOp / 1000) / 1e6).toFixed(0)} Mpx/s equivalent\n`);
}

// ---- 3) 演示场景整帧 ----
{
  // 动态 import 避免 bench 对 scene 的额外职责
  const { buildDemoScene, demoOrbitCamera } = await import("../src/scene.js");
  const fb = new Framebuffer(W, H);
  const renderer = new Renderer(fb);
  const scene = buildDemoScene();
  scene.animate(1.0);
  const cam = demoOrbitCamera(0.55, 0.34, 6.6);
  const triCount = scene.items.reduce((s, it) => s + it.mesh.indices.length / 3, 0);
  const res = runFor(() => {
    renderer.render(cam, scene.items, "phong", scene.lights, { sortOpaque: true });
    return triCount;
  }, triCount);
  console.log(`[3] 演示场景整帧(${scene.items.length} 个物体,${triCount} tris,排序开)`);
  console.log(`    ${res.msPerOp.toFixed(2)} ms/frame ≈ ${(1000 / res.msPerOp).toFixed(0)} fps(不含 putImageData/合成)\n`);
}

// ---- 4) 1 万三角形场景(二期 M7 验收基准) ----
{
  const { makeSphere, makeGround, makeCube } = await import("../src/scene.js");
  const { checkerTexture } = await import("../src/texture.js");
  const { mat4Translation, mat4Identity } = await import("../src/math.js");
  const fb = new Framebuffer(W, H);
  const renderer = new Renderer(fb);
  // 4×UV 球(48×24 = 2304 tris/个 = 9216)+ 地面 + 8 立方体 ≈ 9.4k
  const tex = checkerTexture(128, 16, [190, 184, 172], [70, 76, 82]);
  const items: DrawItem[] = [];
  for (let i = 0; i < 4; i++) {
    items.push({
      mesh: makeSphere(0.9, 48, 24),
      model: mat4Translation({ x: -2.4 + i * 1.6, y: 0, z: 0 }),
      material: { albedo: [0.85 - i * 0.15, 0.3 + i * 0.1, 0.22], specular: 0.5, shininess: 32 },
    });
  }
  items.push({ mesh: makeGround(30, -1), model: mat4Identity(), material: { albedo: [1, 1, 1], specular: 0.06, shininess: 8, texture: tex, sampler: "bilinear", doubleSided: true, mip: true } });
  for (let i = 0; i < 8; i++) {
    items.push({
      mesh: makeCube(0.6),
      model: mat4Translation({ x: -3.5 + i, y: -0.2, z: i % 2 === 0 ? -1.2 : 1.2 }),
      material: { albedo: [0.3, 0.5, 0.7], specular: 0.2, shininess: 12, texture: tex, sampler: "bilinear", mip: true },
    });
  }
  const cam = { eye: { x: 0, y: 2.2, z: 7 }, target: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 }, fovYRad: 55 * Math.PI / 180, aspect: 1, near: 0.1, far: 60 };
  const lights = { ambient: [0.12, 0.12, 0.14] as [number, number, number], dirs: [{ dir: [0.45, 0.8, 0.5] as [number, number, number], color: [0.85, 0.82, 0.75] as [number, number, number] }], points: [], spots: [] };
  lights.dirs[0]!.dir[0] /= Math.hypot(0.45, 0.8, 0.5);
  lights.dirs[0]!.dir[1] /= Math.hypot(0.45, 0.8, 0.5);
  lights.dirs[0]!.dir[2] /= Math.hypot(0.45, 0.8, 0.5);
  const triCount = items.reduce((s, it) => s + it.mesh.indices.length / 3, 0);
  const res = runFor(() => {
    renderer.render(cam, items, "phong", lights, { sortOpaque: true });
    return triCount;
  }, triCount);
  console.log(`[4] 1 万三角形场景(${items.length} 个物体,${triCount} tris,视锥剔除+排序,mip 纹理,极端满屏特写)`);
  console.log(`    512²:${res.msPerOp.toFixed(2)} ms/frame ≈ ${(1000 / res.msPerOp).toFixed(0)} fps`);
  // 分辨率缩放扫描(M4 特性):像素成本按面积缩放
  for (const size of [448, 384, 256]) {
    const fb2 = new Framebuffer(size, size);
    const r2 = new Renderer(fb2);
    const r3 = runFor(() => {
      r2.render(cam, items, "phong", lights, { sortOpaque: true });
      return triCount;
    }, triCount);
    console.log(`    ${size}²:${r3.msPerOp.toFixed(2)} ms/frame ≈ ${(1000 / r3.msPerOp).toFixed(0)} fps`);
  }
  console.log();
}
