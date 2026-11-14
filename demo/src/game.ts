/**
 * 演示游戏(二期 M6)浏览器壳:第三人称收集玩法。
 * 逻辑全部在 src/game/logic.ts(纯逻辑、node 单测);本文件做场景搭建 + 输入 + 呈现。
 * 纯逻辑动画:腿部摆动/身体起伏全部由 transform 插值(sin 相位)驱动 —— 无骨骼。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { Renderer, type Camera, type DrawItem, type Material } from "../../src/pipeline.js";
import { makeCube, makeSphere, makeTorus, makeGround } from "../../src/scene.js";
import { checkerTexture, buildMip1 } from "../../src/texture.js";
import { SceneNode, makeTransform, updateWorldMatrices, collectDrawItems } from "../../src/scenegraph.js";
import {
  createGameState, stepGame, collectCamera, collectibleAnim, GAME_BOUNDS,
} from "../../src/game/logic.js";

const VIEW = 512;
const scale = 0.5; // 默认半分辨率(60fps 预算;M4 分辨率缩放特性)

const canvas = document.getElementById("view") as HTMLCanvasElement;
const ctx = canvas.getContext("2d", { alpha: false }) as CanvasRenderingContext2D;
const off = document.createElement("canvas");
const offCtx = off.getContext("2d") as CanvasRenderingContext2D;
const W = Math.round(VIEW * scale);
const H = W;
const imageData = offCtx.createImageData(W, H);
const fb = new Framebuffer(W, H, imageData.data);
const renderer = new Renderer(fb);

const fpsEl = document.getElementById("fps") as HTMLSpanElement;
const resEl = document.getElementById("res") as HTMLSpanElement;
const scoreEl = document.getElementById("score") as HTMLDivElement;
const overlay = document.getElementById("overlay") as HTMLDivElement;
const winTimeEl = document.getElementById("winTime") as HTMLDivElement;

// ---- 场景搭建 ----
const groundTex = checkerTexture(128, 16, [190, 184, 172], [70, 76, 82]);
groundTex.mip = buildMip1(groundTex);
const bodyTex = checkerTexture(32, 4, [90, 140, 220], [60, 90, 150]);

let state = createGameState(42);
let winAt = -1;

const root = new SceneNode("gameRoot");
const ground = new SceneNode("ground", {
  mesh: makeGround(GAME_BOUNDS * 2 + 4, 0),
  material: {
    albedo: [1, 1, 1], specular: 0.06, shininess: 8,
    texture: groundTex, sampler: "bilinear", doubleSided: true, mip: true,
  },
});
root.add(ground);

// 角色:身体 + 头 + 双腿(纯 transform 动画)
const player = new SceneNode("player", { transform: makeTransform({ x: 0, y: 0, z: 0 }) });
const bodyMat: Material = { albedo: [1, 1, 1], specular: 0.3, shininess: 24, texture: bodyTex, sampler: "bilinear" };
const body = new SceneNode("body", {
  transform: (() => { const t = makeTransform({ x: 0, y: 0.75, z: 0 }); t.scale = { x: 0.5, y: 0.7, z: 0.35 }; return t; })(),
  mesh: makeCube(1),
  material: bodyMat,
});
const head = new SceneNode("head", {
  transform: (() => { const t = makeTransform({ x: 0, y: 1.45, z: 0 }); t.scale = { x: 0.32, y: 0.32, z: 0.32 }; return t; })(),
  mesh: makeSphere(0.5, 24, 12),
  material: { albedo: [0.92, 0.75, 0.6], specular: 0.15, shininess: 12 },
});
const legL = new SceneNode("legL", {
  transform: (() => { const t = makeTransform({ x: -0.16, y: 0.25, z: 0 }); t.scale = { x: 0.16, y: 0.5, z: 0.16 }; return t; })(),
  mesh: makeCube(1),
  material: { albedo: [0.25, 0.28, 0.34], specular: 0.1, shininess: 8 },
});
const legR = new SceneNode("legR", {
  transform: (() => { const t = makeTransform({ x: 0.16, y: 0.25, z: 0 }); t.scale = { x: 0.16, y: 0.5, z: 0.16 }; return t; })(),
  mesh: makeCube(1),
  material: { albedo: [0.25, 0.28, 0.34], specular: 0.1, shininess: 8 },
});
player.add(body).add(head).add(legL).add(legR);
root.add(player);

// 边界墙(4 根矮柱,纯装饰)
const wallMat: Material = { albedo: [0.45, 0.42, 0.5], specular: 0.15, shininess: 10 };
for (const [wx, wz] of [[-GAME_BOUNDS, -GAME_BOUNDS], [GAME_BOUNDS, -GAME_BOUNDS], [-GAME_BOUNDS, GAME_BOUNDS], [GAME_BOUNDS, GAME_BOUNDS]] as Array<[number, number]>) {
  const t = makeTransform({ x: wx, y: 0.6, z: wz });
  t.scale = { x: 1, y: 1.2, z: 1 };
  root.add(new SceneNode(`wall_${wx}_${wz}`, { transform: t, mesh: makeCube(1), material: wallMat }));
}

// 收集物:10 个环面(金色)
const ringMesh = makeTorus(0.35, 0.12, 24, 12);
const ringMat: Material = { albedo: [1, 1, 1], specular: 0.8, shininess: 64 };
const ringNodes: SceneNode[] = [];
for (let i = 0; i < state.collectibles.length; i++) {
  const n = new SceneNode(`ring${i}`, {
    transform: makeTransform({ x: state.collectibles[i]!.x, y: 0.55, z: state.collectibles[i]!.z }),
    mesh: ringMesh,
    material: ringMat,
  });
  ringNodes.push(n);
  root.add(n);
}

// 装饰树(圆锥形 = 两层缩放立方体;散在场地四角区域)
const treeMat: Material = { albedo: [0.3, 0.55, 0.3], specular: 0.1, shininess: 8 };
const treeSpots: Array<[number, number]> = [
  [-8, -10], [-11, -4], [8, -11], [11, -5], [-9, 9], [9, 8], [0, -12], [12, 2], [-12, 1], [3, 11],
];
for (const [tx, tz] of treeSpots) {
  const t = makeTransform({ x: tx, y: 0.9, z: tz });
  t.scale = { x: 0.9, y: 1.8, z: 0.9 };
  root.add(new SceneNode(`tree_${tx}_${tz}`, { transform: t, mesh: makeCube(1), material: treeMat }));
}

const lights = {
  ambient: [0.14, 0.14, 0.17] as [number, number, number],
  dirs: [{ dir: [0.45, 0.8, 0.5] as [number, number, number], color: [0.85, 0.82, 0.75] as [number, number, number] }],
  points: [{ pos: { x: 0, y: 8, z: 0 }, color: [0.55, 0.6, 0.75] as [number, number, number], k1: 0.06, k2: 0.01 }],
  spots: [],
};
{
  const d = lights.dirs[0]!.dir;
  const l = Math.hypot(d[0], d[1], d[2]);
  d[0] /= l; d[1] /= l; d[2] /= l;
}

// ---- 输入 ----
const KEYMAP: Record<string, "up" | "down" | "left" | "right"> = {
  KeyW: "up", KeyS: "down", KeyA: "left", KeyD: "right",
  ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
};
window.addEventListener("keydown", (e) => {
  const k = KEYMAP[e.code];
  if (k !== undefined) {
    state.input[k] = true;
    e.preventDefault();
  }
});
window.addEventListener("keyup", (e) => {
  const k = KEYMAP[e.code];
  if (k !== undefined) state.input[k] = false;
});
document.getElementById("restart")!.addEventListener("click", () => {
  state = createGameState(42);
  winAt = -1;
  overlay.style.display = "none";
});

// ---- 主循环 ----
let fpsEma = 60;
let lastT = 0;
let hudT = 0;
let logicT = 0;

function applyState(): void {
  player.transform.position.x = state.player.x;
  player.transform.position.z = state.player.z;
  player.transform.rotationEuler.y = state.player.rotY;
  // 纯逻辑动画:移动时腿摆动(±0.55 rad @ 2.6 Hz)、身体轻微起伏
  const swing = state.player.moving ? Math.sin(state.t * 2.6 * Math.PI * 2) * 0.55 : 0;
  const bob = state.player.moving ? Math.abs(Math.sin(state.t * 2.6 * Math.PI)) * 0.05 : 0;
  legL.transform.rotationEuler.x = swing;
  legR.transform.rotationEuler.x = -swing;
  body.transform.position.y = 0.75 + bob;
  head.transform.position.y = 1.45 + bob;
  for (let i = 0; i < state.collectibles.length; i++) {
    const c = state.collectibles[i]!;
    const n = ringNodes[i]!;
    if (c.taken) {
      n.visible = false;
      continue;
    }
    const anim = collectibleAnim(state, c);
    n.transform.position.y = anim.y;
    n.transform.rotationEuler.y = anim.rotY;
    n.visible = true;
  }
}

function frame(now: number): void {
  const dtMs = lastT === 0 ? 16.7 : Math.min(100, now - lastT);
  lastT = now;
  const dt = Math.min(0.05, dtMs / 1000);

  stepGame(state, dt);
  applyState();
  logicT += dt;

  updateWorldMatrices(root);
  const items: DrawItem[] = collectDrawItems(root);
  const cam: Camera = collectCamera(state);
  fb.clear(38, 42, 54);
  renderer.render(cam, items, "phong", lights, { sortOpaque: true });
  offCtx.putImageData(imageData, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, VIEW, VIEW);

  // HUD
  fpsEma = fpsEma * 0.9 + (1000 / dtMs) * 0.1;
  if (now - hudT > 250) {
    hudT = now;
    fpsEl.textContent = `${fpsEma.toFixed(1)} fps`;
    resEl.textContent = `${W}×${H}`;
    fpsEl.style.color = fpsEma >= 50 ? "#7fd08a" : fpsEma >= 25 ? "#e2c06c" : "#e27a6c";
    scoreEl.textContent = `得分 ${state.score} / ${state.collectibles.length}`;
    if (state.won && winAt < 0) {
      winAt = logicT;
      winTimeEl.textContent = `用时 ${winAt.toFixed(1)} 秒`;
      overlay.style.display = "flex";
    }
  }
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
