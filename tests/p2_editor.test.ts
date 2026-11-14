import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Framebuffer } from "../src/framebuffer.js";
import { Renderer, type DrawItem, type Camera } from "../src/pipeline.js";
import { SceneNode, makeTransform, collectDrawItems, updateWorldMatrices } from "../src/scenegraph.js";
import { buildDemoScene, demoOrbitCamera, makeCube, makeSphere } from "../src/scene.js";
import { checkerTexture, type Texture } from "../src/texture.js";
import type { Mesh, Material } from "../src/pipeline.js";
import { screenRay, rayTriangle, raySphere, pickScene } from "../src/editor/picking.js";
import { projectPoint, pickGizmoMove, dragMoveDelta, pickGizmoRotate, dragRotateDelta } from "../src/editor/gizmo.js";
import { sceneToJSON, sceneFromJSON, type MeshJSON, type TextureJSON } from "../src/editor/serialize.js";
import { decodePPM, framebufferToRaw, type RawImage } from "../src/ppm.js";
import { diffRGB } from "../src/diff.js";
import type { LightSetup } from "../src/shading.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "..", "..", "golden");

function loadGolden(name: string): RawImage {
  return decodePPM(new Uint8Array(readFileSync(join(GOLDEN_DIR, name))));
}

/** 测试相机:原点,fov 90°(tanH=1),看向 -Z。 */
const CAM: Camera = {
  eye: { x: 0, y: 0, z: 0 },
  target: { x: 0, y: 0, z: -1 },
  up: { x: 0, y: 1, z: 0 },
  fovYRad: Math.PI / 2,
  aspect: 1,
  near: 0.1,
  far: 100,
};

// ---------- 拾取:射线构造 ----------

test("P2 拾取: screenRay 中心/偏轴锚定(fov 90°)", () => {
  const center = screenRay(CAM, 128, 128, 63.5, 63.5);
  assert.ok(center.dir.x === 0 && center.dir.y === 0 && Math.abs(center.dir.z + 1) < 1e-12);
  const right = screenRay(CAM, 128, 128, 127.5, 63.5);
  // ndcX = (127.5+0.5)/128*2-1 = 1 → dir ∝ (1,0,-1)/√2
  assert.ok(Math.abs(right.dir.x / -right.dir.z - 1) < 1e-12, `dx/dz: ${right.dir.x / -right.dir.z}`);
  assert.ok(right.dir.x > 0 && right.dir.z < 0);
  // 屏幕 y 向下:上方像素 → +Y 世界方向
  const up = screenRay(CAM, 128, 128, 63.5, 0.5 - 0.5); // py=0 → ndcY = 1-(0.5/128)*2 ≈ 0.9922
  assert.ok(up.dir.y > 0, "top pixel must look up");
});

test("P2 拾取: rayTriangle 锚定 —— 正中命中 t=2,错失 null,背后 null", () => {
  const o = { x: 0, y: 0, z: 0 };
  const d = { x: 0, y: 0, z: -1 };
  const p0 = { x: -1, y: -1, z: -2 };
  const p1 = { x: 1, y: -1, z: -2 };
  const p2 = { x: 0, y: 1, z: -2 };
  assert.ok(Math.abs(rayTriangle(o, d, p0, p1, p2)! - 2) < 1e-12);
  // 真退化(p0 == p1,面积零)→ null
  assert.equal(rayTriangle(o, d, p0, p0, p2), null);
  assert.equal(rayTriangle(o, d, { x: 3, y: 3, z: -2 }, { x: 5, y: 3, z: -2 }, { x: 4, y: 5, z: -2 }), null);
  assert.equal(rayTriangle(o, d, { x: -1, y: -1, z: 2 }, { x: 1, y: -1, z: 2 }, { x: 0, y: 1, z: 2 }), null, "behind → null");
  // 双面:背面也能命中
  const hitBack = rayTriangle(o, d, p0, p2, p1);
  assert.ok(hitBack !== null && Math.abs(hitBack - 2) < 1e-12);
});

test("P2 拾取: raySphere 锚定 —— 正穿/错失/背后;起源在球面 → 返回出射 t=2(冻结:首根 > 1e-9)", () => {
  const o = { x: 0, y: 0, z: 0 };
  const d = { x: 0, y: 0, z: -1 };
  assert.ok(Math.abs(raySphere(o, d, { x: 0, y: 0, z: -3 }, 1)! - 2) < 1e-12);
  // 起源恰在球面(球心 z=-1 r=1):入射根 t=0 被 eps 滤掉,返回出射根 t=2(冻结语义)
  assert.ok(Math.abs(raySphere(o, d, { x: 0, y: 0, z: -1 }, 1)! - 2) < 1e-12);
  assert.equal(raySphere(o, d, { x: 5, y: 0, z: -3 }, 1), null);
  assert.equal(raySphere(o, d, { x: 0, y: 0, z: 3 }, 1), null, "sphere behind → null");
});

// ---------- 拾取:场景级 ----------

function cubeScene(): SceneNode {
  const mat: Material = { albedo: [1, 1, 1], specular: 0, shininess: 1 };
  const root = new SceneNode("root");
  const far = new SceneNode("far", { transform: makeTransform({ x: 0, y: 0, z: -6 }), mesh: makeCube(1), material: mat });
  const near = new SceneNode("near", { transform: makeTransform({ x: 0, y: 0, z: -2 }), mesh: makeCube(1), material: mat });
  root.add(far);
  root.add(near);
  return root;
}

test("P2 拾取: pickScene 命中最近物体并返回节点;遮挡正确;未命中 null", () => {
  const cam: Camera = { ...CAM, eye: { x: 0, y: 0, z: 5 }, target: { x: 0, y: 0, z: 0 } };
  const root = cubeScene();
  const hit = pickScene(root, cam, 128, 128, 63.5, 63.5);
  assert.ok(hit !== null, "center pixel must hit");
  assert.equal(hit.node.name, "near");
  // 偏移到两立方体之间的空隙?cube(1) 在 x=0 — 两侧未命中点
  const miss = pickScene(root, cam, 128, 128, 63.5, 2.5); // 屏幕最上方 → 视线上方,应 miss(z=-6 处)
  void miss;
  // 命中远方的点:把 near 隐藏
  const root2 = cubeScene();
  root2.children[1]!.visible = false;
  const hit2 = pickScene(root2, cam, 128, 128, 63.5, 63.5);
  assert.ok(hit2 !== null && hit2.node.name === "far", "with near hidden, ray hits far");
});

// ---------- Gizmo ----------

test("P2 gizmo: projectPoint 锚定(fov 90°) —— 中心/±X 边缘", () => {
  const c = projectPoint(CAM, 128, 128, { x: 0, y: 0, z: -2 });
  assert.ok(Math.abs(c.x - 64) < 1e-12 && Math.abs(c.y - 64) < 1e-12);
  const e = projectPoint(CAM, 128, 128, { x: 2, y: 0, z: -2 });
  assert.ok(Math.abs(e.x - 128) < 1e-12, `edge x: ${e.x}`);
  const behind = projectPoint(CAM, 128, 128, { x: 0, y: 0, z: 2 });
  assert.ok(behind.behind);
});

test("P2 gizmo: pickGizmoMove 命中三轴/未命中", () => {
  // gizmo 在 z=-2:轴段 X:(64,64)→(128,64);Y:(64,64)→(64,0);Z:(64,64)→(64,128)(朝相机,投影重合 — 透视下 Z 轴指向观察者,投影为一点 → 命中靠距离)
  const origin = { x: 0, y: 0, z: -2 };
  assert.equal(pickGizmoMove(origin, CAM, 128, 128, 100, 64), 0, "X arrow");
  assert.equal(pickGizmoMove(origin, CAM, 128, 128, 64, 20), 1, "Y arrow");
  assert.equal(pickGizmoMove(origin, CAM, 128, 128, 10, 10), null, "miss");
});

test("P2 gizmo: dragMoveDelta 锚定 —— +28px 右拖 = +1.09375 世界单位(X)", () => {
  const origin = { x: 0, y: 0, z: 0 };
  const cam: Camera = { ...CAM, eye: { x: 0, y: 0, z: 5 }, target: { x: 0, y: 0, z: 0 } };
  // px0=63.5 → ndcX0 = 1/128;px1=77.5 → ndcX1 = 29/128;平面 z=0 距相机 5 → Δx = 5·(28/128) = 1.09375
  const d = dragMoveDelta(0, origin, cam, 128, 128, 63.5, 63.5, 77.5, 63.5);
  assert.ok(d !== null && Math.abs(d.x - 1.09375) < 1e-9, `dx: ${d?.x}`);
  assert.ok(Math.abs(d!.y) < 1e-12 && Math.abs(d!.z) < 1e-12);
  // Y 轴拖拽:向下拖 → 世界 -Y?屏幕 y 向下 → 下方像素 ndcY 负 → Δy 负
  const dy = dragMoveDelta(1, origin, cam, 128, 128, 63.5, 63.5, 63.5, 77.5);
  assert.ok(dy !== null && Math.abs(dy.y + 1.09375) < 1e-9, `dy: ${dy?.y}`);
});

test("P2 gizmo: pickGizmoRotate 命中环(俯视相机)+ dragRotateDelta 锚定 —— 90° 旋转", () => {
  // 拖拽锚定:相机 (0,0,5) fov 90°,Z 环(XY 平面):射线与 z=0 平面交点 = (5·ndcX, 5·ndcY, 0)
  const cam0: Camera = { ...CAM, eye: { x: 0, y: 0, z: 5 }, target: { x: 0, y: 0, z: 0 } };
  const center = { x: 0, y: 0, z: 0 };
  const angle = dragRotateDelta(2, center, cam0, 128, 128, 101.9, 63.5, 63.5, 25.1);
  assert.ok(angle !== null && Math.abs(angle - Math.PI / 2) < 1e-6, `angle: ${angle}`);
  // 拾取:俯视相机使三环都投成可辨椭圆;点击"Z 环 45° 采样点的投影"(避开三环共用的轴交点)
  const cam1: Camera = { ...CAM, eye: { x: 2.5, y: 2.5, z: 2.5 }, target: { x: 0, y: 0, z: 0 } };
  const onRing = projectPoint(cam1, 128, 128, { x: Math.SQRT1_2, y: Math.SQRT1_2, z: 0 }); // Z 环 45° 点
  assert.ok(!onRing.behind);
  const axis = pickGizmoRotate(center, cam1, 128, 128, onRing.x, onRing.y);
  assert.equal(axis, 2, "Z ring picked at its 45° sample point");
});

// ---------- 序列化 round-trip ----------

test("P2 序列化: 材质/网格/纹理 JSON 往返后渲染逐像素一致(合成场景)", () => {
  const root = new SceneNode("root");
  const tex = checkerTexture(32, 4, [200, 150, 80], [90, 60, 40]);
  tex.mip = undefined;
  const meshJSONs = new Map<Mesh, MeshJSON>();
  const texJSONs = new Map<Texture, TextureJSON>();
  const box = new SceneNode("box", {
    transform: makeTransform({ x: -1, y: 0, z: 0 }),
    mesh: makeCube(1),
    material: { albedo: [1, 1, 1], specular: 0.3, shininess: 16, texture: tex, sampler: "bilinear" },
  });
  const ball = new SceneNode("ball", {
    transform: makeTransform({ x: 1, y: 0.2, z: 0.3 }),
    mesh: makeSphere(0.6, 24, 12),
    material: { albedo: [0.85, 0.3, 0.22], specular: 0.5, shininess: 32, shadingModel: "lambert", cullBackface: false },
  });
  root.add(box).add(ball);
  meshJSONs.set(box.mesh!, { kind: "cube", size: 1 });
  meshJSONs.set(ball.mesh!, { kind: "sphere", radius: 0.6, segU: 24, segV: 12 });
  texJSONs.set(tex, { kind: "checker", size: 32, cells: 4, a: [200, 150, 80], b: [90, 60, 40] });

  const cam: Camera = demoOrbitCamera(0.55, 0.34, 6.6);
  const lights: LightSetup = {
    ambient: [0.12, 0.12, 0.14],
    dirs: [{ dir: [0.45, 0.8, 0.5], color: [0.85, 0.82, 0.75] }],
    points: [{ pos: { x: 3.2, y: 2.4, z: 0 }, color: [1, 0.78, 0.55], k1: 0.08, k2: 0.02 }],
    spots: [],
  };

  const render = (r: SceneNode): RawImage => {
    const fb = new Framebuffer(128, 128);
    const items: DrawItem[] = collectDrawItems(r);
    new Renderer(fb).render(cam, items, "phong", lights);
    return framebufferToRaw(fb);
  };

  const json = sceneToJSON({ root, camera: cam, lights, meshRegistry: meshJSONs, textureRegistry: texJSONs });
  // 走真实 JSON 字符串往返(验收 #3 的严格口径)
  const text = JSON.stringify(json);
  const rebuilt = sceneFromJSON(JSON.parse(text) as typeof json);
  const r = diffRGB(render(root), render(rebuilt.root));
  assert.ok(r.ok, `round-trip render mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

test("P2 序列化: 演示场景 → JSON → 重建 → 与升级前锚点黄金图逐像素一致(验收 #3)", () => {
  const scene = buildDemoScene();
  scene.animate(1.25);
  const meshJSONs = new Map<Mesh, MeshJSON>();
  const texJSONs = new Map<Texture, TextureJSON>();
  // 演示场景节点树(DemoScene.root;DFS 先序 ground → cube → sphere → torus)
  const byName = new Map<string, SceneNode>();
  const visit = (n: SceneNode): void => { byName.set(n.name, n); for (const c of n.children) visit(c); };
  visit(scene.root);
  const ground = byName.get("ground")!;
  const cube = byName.get("cube")!;
  const sphere = byName.get("sphere")!;
  const torus = byName.get("torus")!;
  meshJSONs.set(ground.mesh!, { kind: "ground", size: 30, y: -1.3 });
  meshJSONs.set(cube.mesh!, { kind: "cube", size: 0.9 });
  meshJSONs.set(sphere.mesh!, { kind: "sphere", radius: 0.95, segU: 48, segV: 24 });
  meshJSONs.set(torus.mesh!, { kind: "torus", R: 0.85, r: 0.38, segU: 48, segV: 24 });
  texJSONs.set(ground.material!.texture!, { kind: "checker", size: 128, cells: 12, a: [190, 184, 172], b: [70, 76, 82] });
  texJSONs.set(cube.material!.texture!, { kind: "checker", size: 64, cells: 4, a: [205, 150, 80], b: [138, 94, 50] });
  texJSONs.set(torus.material!.texture!, { kind: "stripe", size: 64, stripes: 8, a: [36, 158, 148], b: [232, 230, 224] });

  const cam = demoOrbitCamera(0.55, 0.34, 6.6);
  const json = sceneToJSON({ root: scene.root, camera: cam, lights: scene.lights, meshRegistry: meshJSONs, textureRegistry: texJSONs });
  const text = JSON.stringify(json);
  const rebuilt = sceneFromJSON(JSON.parse(text) as typeof json);

  const fbA = new Framebuffer(256, 256);
  new Renderer(fbA).render(cam, scene.items, "phong", scene.lights);
  const fbB = new Framebuffer(256, 256);
  // 重建场景必须先求世界矩阵(items getter 内置了这一步;这里显式调用)
  updateWorldMatrices(rebuilt.root);
  new Renderer(fbB).render(cam, collectDrawItems(rebuilt.root), "phong", rebuilt.lights);
  const r1 = diffRGB(framebufferToRaw(fbA), framebufferToRaw(fbB));
  assert.ok(r1.ok, `demo round-trip mismatch: ${r1.message} first=${JSON.stringify(r1.first)}`);
  // 与升级前锚点对拍(验收 #3 的最终口径:编辑器改出的场景 JSON 重新加载渲染逐像素一致)
  const r2 = diffRGB(loadGolden("p2_scenegraph_anchor.ppm"), framebufferToRaw(fbB));
  assert.ok(r2.ok, `rebuilt vs anchor mismatch: ${r2.message} first=${JSON.stringify(r2.first)}`);
});
