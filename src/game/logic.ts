/**
 * 演示游戏逻辑核心(二期 M6):第三人称收集玩法,纯逻辑、确定性、可无头测试。
 *
 * 冻结约定:
 * - 相机 yaw 定义:前向 f = (sin yaw, 0, cos yaw);相机位于 player - f·CAM_DIST + (0, CAM_HEIGHT, 0),
 *   看向玩家。W 沿 f 前进;右侧 right = f × up = (-cos yaw, 0, sin yaw);D 沿 right。
 * - 朝向 rotY = atan2(移动 x, 移动 z)(模型 rotY=0 时面向 +Z);转向平滑:每帧朝目标靠拢
 *   min(1, dt·TURN_RATE) 的比例插值(角度按最短弧)。
 * - 收集判定:dx² + dz² < COLLECT_RADIUS²(严格小于,冻结)。
 * - 收集物布局由 mulberry32(seed) 确定 —— 游戏可复现、可测。
 * - 场地边界 [-GAME_BOUNDS, +GAME_BOUNDS]²,越界钳位。
 */

import { mulberry32 } from "../rng.js";
import type { Camera } from "../pipeline.js";
import type { Vec3 } from "../math.js";

export const GAME_BOUNDS = 14;
export const PLAYER_SPEED = 4;
export const COLLECT_COUNT = 10;
export const COLLECT_RADIUS = 0.9;
export const CAM_DIST = 6;
export const CAM_HEIGHT = 3.2;
export const TURN_RATE = 10;

export interface GameInput {
  up: boolean;
  down: boolean;
  left: boolean;
  right: boolean;
}

export interface Collectible {
  x: number;
  z: number;
  /** 相位(动画偏移)。 */
  phase: number;
  taken: boolean;
}

export interface GameState {
  t: number;
  player: { x: number; z: number; rotY: number; moving: boolean };
  camYaw: number;
  /** WASD 输入状态(由浏览器键盘事件写入;逻辑只读)。 */
  input: GameInput;
  collectibles: Collectible[];
  score: number;
  won: boolean;
}

/** 确定性初始状态:收集物 = 双环布局(内环 5 + 外环 5,种子抖动)。 */
export function createGameState(seed: number): GameState {
  const rnd = mulberry32(seed);
  const collectibles: Collectible[] = [];
  for (let i = 0; i < COLLECT_COUNT; i++) {
    const ring = i < 5 ? 0 : 1;
    const baseAngle = (i % 5) * (2 * Math.PI / 5);
    const radius = ring === 0 ? 5 : 9;
    const jitter = (rnd() - 0.5) * 1.6;
    const a = baseAngle + jitter;
    collectibles.push({
      x: Math.cos(a) * radius,
      z: Math.sin(a) * radius,
      phase: rnd() * 2 * Math.PI,
      taken: false,
    });
  }
  return {
    t: 0,
    player: { x: 0, z: 0, rotY: 0, moving: false },
    camYaw: 0,
    input: { up: false, down: false, left: false, right: false },
    collectibles,
    score: 0,
    won: false,
  };
}

/** 推进一帧(dt 秒;渲染耗时不应拖慢玩法 —— 调用方钳位 dt)。 */
export function stepGame(s: GameState, dt: number): void {
  s.t += dt;
  const inp = s.input;
  const fy = s.camYaw;
  const fx = Math.sin(fy), fz = Math.cos(fy);
  const rx = -Math.cos(fy), rz = Math.sin(fy);
  const mvF = (inp.up ? 1 : 0) - (inp.down ? 1 : 0);
  const mvR = (inp.right ? 1 : 0) - (inp.left ? 1 : 0);
  let dx = fx * mvF + rx * mvR;
  let dz = fz * mvF + rz * mvR;
  const len = Math.sqrt(dx * dx + dz * dz);
  s.player.moving = len > 0;
  if (len > 0) {
    dx /= len; dz /= len;
    s.player.x += dx * PLAYER_SPEED * dt;
    s.player.z += dz * PLAYER_SPEED * dt;
    // 边界钳位
    if (s.player.x > GAME_BOUNDS) s.player.x = GAME_BOUNDS;
    if (s.player.x < -GAME_BOUNDS) s.player.x = -GAME_BOUNDS;
    if (s.player.z > GAME_BOUNDS) s.player.z = GAME_BOUNDS;
    if (s.player.z < -GAME_BOUNDS) s.player.z = -GAME_BOUNDS;
    // 朝向平滑转向(最短弧)
    const target = Math.atan2(dx, dz);
    let diff = target - s.player.rotY;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    const k = Math.min(1, dt * TURN_RATE);
    s.player.rotY += diff * k;
  }
  // 收集判定
  if (!s.won) {
    for (const c of s.collectibles) {
      if (c.taken) continue;
      const ddx = c.x - s.player.x, ddz = c.z - s.player.z;
      if (ddx * ddx + ddz * ddz < COLLECT_RADIUS * COLLECT_RADIUS) {
        c.taken = true;
        s.score++;
        if (s.score >= COLLECT_COUNT) s.won = true;
      }
    }
  }
}

/** 第三人称跟随相机(冻结公式)。 */
export function collectCamera(s: GameState): Camera {
  const p = s.player;
  const fx = Math.sin(s.camYaw), fz = Math.cos(s.camYaw);
  const eye: Vec3 = {
    x: p.x - fx * CAM_DIST,
    y: CAM_HEIGHT,
    z: p.z - fz * CAM_DIST,
  };
  return {
    eye,
    target: { x: p.x, y: 1.0, z: p.z },
    up: { x: 0, y: 1, z: 0 },
    fovYRad: (55 * Math.PI) / 180,
    aspect: 1,
    near: 0.1,
    far: 120,
  };
}

/** 收集物动画(供视图层查询):悬浮高度与自转角(冻结公式)。 */
export function collectibleAnim(s: GameState, c: Collectible): { y: number; rotY: number } {
  return {
    y: 0.55 + 0.15 * Math.sin(s.t * 2 + c.phase),
    rotY: s.t * 2 + c.phase,
  };
}
