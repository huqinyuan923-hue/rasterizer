import { test } from "node:test";
import assert from "node:assert/strict";

import {
  createGameState, stepGame, collectCamera, collectibleAnim,
  GAME_BOUNDS, COLLECT_COUNT, COLLECT_RADIUS, PLAYER_SPEED,
} from "../src/game/logic.js";

/** 直行 N 秒(按 60Hz 步进)。 */
function hold(s: ReturnType<typeof createGameState>, key: "up" | "down" | "left" | "right", seconds: number): void {
  s.input[key] = true;
  for (let i = 0; i < Math.round(seconds * 60); i++) stepGame(s, 1 / 60);
  s.input[key] = false;
}

test("P2 游戏: 初始状态 —— 10 个收集物、0 分、未胜、玩家在原点", () => {
  const s = createGameState(42);
  assert.equal(s.collectibles.length, COLLECT_COUNT);
  assert.equal(s.score, 0);
  assert.ok(!s.won);
  assert.ok(s.player.x === 0 && s.player.z === 0);
  for (const c of s.collectibles) {
    assert.ok(!c.taken);
    assert.ok(Math.abs(c.x) <= GAME_BOUNDS && Math.abs(c.z) <= GAME_BOUNDS);
  }
});

test("P2 游戏: 布局确定性 —— 同种子两次创建逐值一致;不同种子不同", () => {
  const a = createGameState(7);
  const b = createGameState(7);
  assert.deepEqual(a.collectibles, b.collectibles);
  const c = createGameState(8);
  assert.ok(JSON.stringify(a.collectibles) !== JSON.stringify(c.collectibles), "different seeds must differ");
});

test("P2 游戏: 移动 —— yaw=0 时 W 沿 +Z 前进,速度 4 单位/秒(锚定)", () => {
  const s = createGameState(1);
  hold(s, "up", 1);
  assert.ok(Math.abs(s.player.z - PLAYER_SPEED) < 1e-9, `z after 1s: ${s.player.z}`);
  assert.ok(Math.abs(s.player.x) < 1e-9);
  // 朝向:目标 atan2(0, 1) = 0,起始 0 → 保持 0
  assert.ok(Math.abs(s.player.rotY) < 1e-12, `rotY: ${s.player.rotY}`);
  assert.ok(s.player.moving);
});

test("P2 游戏: 转向平滑 —— 朝向向移动方向收敛(最短弧插值,指数收敛)", () => {
  const s = createGameState(1);
  s.player.rotY = 3; // 背对移动方向
  hold(s, "up", 2); // 2s ≈ (1 - 1/6)^120 → 残差 ~3e-10
  assert.ok(Math.abs(s.player.rotY) < 1e-6, `rotY converged: ${s.player.rotY}`);
});

test("P2 游戏: 边界钳位 —— 直行 10 秒不越界", () => {
  const s = createGameState(1);
  hold(s, "up", 10);
  assert.ok(Math.abs(s.player.z) <= GAME_BOUNDS, `z: ${s.player.z}`);
  const s2 = createGameState(1);
  s2.player.x = GAME_BOUNDS - 0.1;
  s2.input.up = false;
  s2.input.right = true;
  s2.camYaw = Math.PI / 2; // right 方向指向 +X?验证不越界即可
  for (let i = 0; i < 600; i++) stepGame(s2, 1 / 60);
  assert.ok(Math.abs(s2.player.x) <= GAME_BOUNDS + 1e-9);
});

test("P2 游戏: 收集 —— 走到收集物中心必收集;R·0.999 内收集、R·1.001 外不收集", () => {
  const s = createGameState(3);
  const c0 = s.collectibles[0]!;
  s.player.x = c0.x;
  s.player.z = c0.z;
  stepGame(s, 0.016);
  assert.equal(s.score, 1);
  assert.ok(c0.taken);
  // 半径内 0.1%:收集
  const c1 = s.collectibles[1]!;
  s.player.x = c1.x + COLLECT_RADIUS * 0.999;
  s.player.z = c1.z;
  stepGame(s, 0.016);
  assert.equal(s.score, 2, "just inside radius must collect");
  // 半径外 0.1%:不收集(严格 < R²;冻结语义在远离边界处验证,避免 FP 边界抖动)
  const c2 = s.collectibles[2]!;
  s.player.x = c2.x + COLLECT_RADIUS * 1.001;
  s.player.z = c2.z;
  stepGame(s, 0.016);
  assert.equal(s.score, 2, "just outside radius must not collect");
});

test("P2 游戏: 胜利与重开 —— 集齐 10 个 won=true;重新 createGameState 完整复位", () => {
  const s = createGameState(5);
  for (const c of s.collectibles) {
    s.player.x = c.x;
    s.player.z = c.z;
    stepGame(s, 0.016);
  }
  assert.equal(s.score, COLLECT_COUNT);
  assert.ok(s.won);
  // 重开 = 新状态(视图层 createGameState 同种子 → 同布局)
  const fresh = createGameState(5);
  assert.equal(fresh.score, 0);
  assert.ok(!fresh.won);
  assert.ok(fresh.collectibles.every((c) => !c.taken));
  assert.deepEqual(fresh.collectibles.map((c) => [c.x, c.z]), s.collectibles.map((c) => [c.x, c.z]), "same seed → same layout");
});

test("P2 游戏: 相机跟随 —— eye 在玩家后方 CAM_DIST、高 CAM_HEIGHT,看向玩家", () => {
  const s = createGameState(1);
  hold(s, "up", 1);
  const cam = collectCamera(s);
  const fx = Math.sin(s.camYaw), fz = Math.cos(s.camYaw);
  assert.ok(Math.abs(cam.eye.x - (s.player.x - fx * 6)) < 1e-9);
  assert.ok(Math.abs(cam.eye.z - (s.player.z - fz * 6)) < 1e-9);
  assert.ok(Math.abs(cam.eye.y - 3.2) < 1e-9);
  assert.ok(Math.abs(cam.target.x - s.player.x) < 1e-9);
  assert.ok(Math.abs(cam.target.z - s.player.z) < 1e-9);
});

test("P2 游戏: 收集物动画 —— 悬浮与自转是 t 的连续函数(纯 transform 插值,无骨骼)", () => {
  const s = createGameState(1);
  const c = s.collectibles[0]!;
  const a0 = collectibleAnim(s, c);
  s.t = 0.25;
  const a1 = collectibleAnim(s, c);
  assert.ok(a1.y !== a0.y && a1.rotY !== a0.rotY);
  // t 回退 → 动画确定回退(纯函数)
  s.t = 0;
  const a2 = collectibleAnim(s, c);
  assert.deepEqual(a2, a0);
});
