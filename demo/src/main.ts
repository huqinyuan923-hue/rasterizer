/**
 * Demo 主程序：512² 画布、rAF 主循环、轨道相机（拖拽/滚轮）、FPS 显示。
 * 渲染全部走 src/ 的纯 CPU 管线 —— 页面只负责呈现与输入。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { Renderer } from "../../src/pipeline.js";
import { buildDemoScene, demoOrbitCamera, type DemoScene } from "../../src/scene.js";

const W = 512;
const H = 512;

const canvas = document.getElementById("view") as HTMLCanvasElement;
const ctx = canvas.getContext("2d", { alpha: false }) as CanvasRenderingContext2D;
const imageData = ctx.createImageData(W, H);
const fb = new Framebuffer(W, H, imageData.data);
const renderer = new Renderer(fb);
const scene: DemoScene = buildDemoScene();

// ---- 轨道相机状态 ----
let yaw = 0.55;
let pitch = 0.34;
let dist = 6.6;
let dragging = false;
let lastX = 0;
let lastY = 0;

canvas.addEventListener("mousedown", (e) => {
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
});
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  yaw -= (e.clientX - lastX) * 0.005;
  pitch += (e.clientY - lastY) * 0.005;
  pitch = Math.max(-0.15, Math.min(1.35, pitch));
  lastX = e.clientX;
  lastY = e.clientY;
});
window.addEventListener("mouseup", () => { dragging = false; });
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  dist *= Math.exp(e.deltaY * 0.001);
  dist = Math.max(3, Math.min(15, dist));
}, { passive: false });

// ---- 主循环 ----
const fpsEl = document.getElementById("fps") as HTMLSpanElement;
const msEl = document.getElementById("ms") as HTMLSpanElement;
let fpsEma = 60;
let msEma = 16;
let lastT = 0;
let hudT = 0;
let animT = 0;

function frame(now: number): void {
  if (lastT !== 0) {
    const dtMs = Math.min(100, now - lastT);
    fpsEma = fpsEma * 0.9 + (1000 / dtMs) * 0.1;
    msEma = msEma * 0.9 + dtMs * 0.1;
  }
  lastT = now;
  animT += 1 / 60; // 动画时间按帧推进（渲染耗时不应拖慢动画相位）

  scene.animate(animT);
  renderer.render(demoOrbitCamera(yaw, pitch, dist), scene.items, "phong", scene.lights);
  ctx.putImageData(imageData, 0, 0);

  if (now - hudT > 250) {
    hudT = now;
    fpsEl.textContent = `${fpsEma.toFixed(1)} fps`;
    msEl.textContent = `${msEma.toFixed(1)} ms`;
    fpsEl.style.color = fpsEma >= 55 ? "#7fd08a" : fpsEma >= 30 ? "#e2c06c" : "#e27a6c";
  }
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
