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
  const lights = { ambient: [0.12, 0.12, 0.14] as [number, number, number], dir: { dir: [0.45, 0.8, 0.5] as [number, number, number], color: [0.85, 0.82, 0.75] as [number, number, number] }, point: null };
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
    renderer.render(cam, scene.items, "phong", scene.lights);
    return triCount;
  }, triCount);
  console.log(`[3] 演示场景整帧（${scene.items.length} 个物体，${triCount} tris）`);
  console.log(`    ${res.msPerOp.toFixed(2)} ms/frame ≈ ${(1000 / res.msPerOp).toFixed(0)} fps（不含 putImageData/合成）`);
}
