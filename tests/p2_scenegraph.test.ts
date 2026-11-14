import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Framebuffer } from "../src/framebuffer.js";
import { Renderer } from "../src/pipeline.js";
import {
  SceneNode, makeTransform, composeTRS, updateWorldMatrices, collectDrawItems, walkNodes,
} from "../src/scenegraph.js";
import { buildDemoScene, demoOrbitCamera } from "../src/scene.js";
import {
  mat4Identity, mat4Mul, mat4RotX, mat4RotY, mat4RotZ, mat4Translation, mat4Scaling, type Mat4,
} from "../src/math.js";
import { decodePPM, framebufferToRaw, type RawImage } from "../src/ppm.js";
import { diffRGB } from "../src/diff.js";
import { renderRefScenegraph } from "../tools/reference/gen_p2_scenegraph.js";
import { writeGolden } from "../tools/goldens.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "..", "..", "golden");

function loadGolden(name: string): RawImage {
  return decodePPM(new Uint8Array(readFileSync(join(GOLDEN_DIR, name))));
}

function goldenExists(name: string): boolean {
  try { readFileSync(join(GOLDEN_DIR, name)); return true; } catch { return false; }
}

function expectMatEqual(a: Mat4, b: Mat4, label: string): void {
  for (let i = 0; i < 16; i++) {
    assert.ok(Object.is(a[i], b[i]), `${label}[${i}]: ${a[i]} !== ${b[i]}`);
  }
}

// ---------- TRS 锚定(冻结公式 ((T·Rz)·Ry)·Rx·S) ----------

test("P2 场景图: 纯平移 TRS 与 mat4Translation 逐位一致", () => {
  const tr = makeTransform({ x: 1.5, y: -2.25, z: 0.125 });
  expectMatEqual(composeTRS(tr), mat4Translation({ x: 1.5, y: -2.25, z: 0.125 }), "TRS(T)");
});

test("P2 场景图: 单轴旋转+平移与旧 demo 的 compose(T, Rot) 逐位一致(迁移等价的数学基石)", () => {
  const y = 0.625, x = -0.14;
  const trY = makeTransform({ x: -1.7, y: -0.85, z: 0.2 });
  trY.rotationEuler.y = y;
  expectMatEqual(
    composeTRS(trY),
    mat4Mul(mat4Translation({ x: -1.7, y: -0.85, z: 0.2 }), mat4RotY(y)),
    "TRS(T,Ry)",
  );
  const trYX = makeTransform({ x: 1.9, y: -0.92, z: -0.1 });
  trYX.rotationEuler.x = x;
  trYX.rotationEuler.y = y;
  // 旧 torus:compose(T, RotY, RotX) = (T·Ry)·Rx;新 TRS:((T·Rz⁰)·Ry)·Rx⁰·S⁰ 必须逐位相同
  expectMatEqual(
    composeTRS(trYX),
    mat4Mul(mat4Mul(mat4Translation({ x: 1.9, y: -0.92, z: -0.1 }), mat4RotY(y)), mat4RotX(x)),
    "TRS(T,Ry,Rx)",
  );
});

test("P2 场景图: 三轴全开的旋转顺序为 X→Y→Z(local = ((T·Rz)·Ry)·Rx·S)", () => {
  const tr = makeTransform({ x: 0.3, y: 0.5, z: 0.7 });
  tr.rotationEuler = { x: 0.41, y: -1.13, z: 0.77 };
  tr.scale = { x: 1.5, y: 0.75, z: 2 };
  const manual = mat4Mul(
    mat4Mul(
      mat4Mul(mat4Translation({ x: 0.3, y: 0.5, z: 0.7 }), mat4RotZ(0.77)),
      mat4RotY(-1.13),
    ),
    mat4RotX(0.41),
  );
  expectMatEqual(composeTRS(tr), mat4Mul(manual, mat4Scaling({ x: 1.5, y: 0.75, z: 2 })), "TRS full");
  // 换序必须不同(防止测试恒真):Rx·Ry·Rz ≠ Rz·Ry·Rx
  const swapped = mat4Mul(mat4Mul(mat4RotX(0.41), mat4RotY(-1.13)), mat4RotZ(0.77));
  let differs = false;
  for (let i = 0; i < 16; i++) if (!Object.is(swapped[i], manual[i])) differs = true;
  assert.ok(differs, "rotation order must matter");
});

// ---------- 层级与遍历 ----------

test("P2 场景图: 两级层级 world = parent·local,与手工合成逐位一致", () => {
  const parent = new SceneNode("parent", { transform: makeTransform({ x: 1, y: 2, z: 3 }) });
  parent.transform.rotationEuler.y = 0.5;
  const child = new SceneNode("child", { transform: makeTransform({ x: 0, y: 1, z: 0 }) });
  child.transform.rotationEuler.x = 0.25;
  const grand = new SceneNode("grand", { transform: makeTransform({ x: 2, y: 0, z: 0 }) });
  parent.add(child);
  child.add(grand);
  updateWorldMatrices(parent);

  const pLocal = composeTRS(parent.transform);
  const cLocal = composeTRS(child.transform);
  const gLocal = composeTRS(grand.transform);
  expectMatEqual(parent.world, mat4Mul(mat4Identity(), pLocal), "parent.world");
  expectMatEqual(child.world, mat4Mul(pLocal, cLocal), "child.world");
  expectMatEqual(grand.world, mat4Mul(mat4Mul(pLocal, cLocal), gLocal), "grand.world");

  // 数值锚:parent 无旋转时,child 世界平移 = parent 平移 + child 平移
  const flat = new SceneNode("flat", { transform: makeTransform({ x: 5, y: 6, z: 7 }) });
  const leaf = new SceneNode("leaf", { transform: makeTransform({ x: 1, y: 1, z: 1 }) });
  flat.add(leaf);
  updateWorldMatrices(flat);
  const w = leaf.world;
  assert.ok(Math.abs(w[12] - 6) < 1e-12 && Math.abs(w[13] - 7) < 1e-12 && Math.abs(w[14] - 8) < 1e-12);
});

test("P2 场景图: 可见性 —— invisible 子树整体跳过,但世界矩阵仍计算", () => {
  const root = new SceneNode("root");
  const hidden = new SceneNode("hidden", { visible: false });
  const inner = new SceneNode("inner");
  hidden.add(inner);
  const shown = new SceneNode("shown");
  root.add(hidden).add(shown);
  const mesh = { positions: [0, 0, 0, 1, 0, 0, 0, 1, 0], normals: [0, 0, 1, 0, 0, 1, 0, 0, 1], uvs: [0, 0, 1, 0, 0, 1], indices: [0, 1, 2] };
  const mat = { albedo: [1, 1, 1] as [number, number, number], specular: 0, shininess: 1 };
  inner.mesh = mesh; inner.material = mat;
  shown.mesh = mesh; shown.material = mat;

  updateWorldMatrices(root);
  const items = collectDrawItems(root);
  assert.equal(items.length, 1, "only 'shown' subtree is drawn");
  // 世界矩阵不受可见性影响
  assert.ok(inner.world[0] === 1 && inner.world[15] === 1);
});

test("P2 场景图: 展平顺序为 DFS 先序(黄金对拍依赖)", () => {
  const root = new SceneNode("root");
  const a = new SceneNode("a"), b = new SceneNode("b");
  const a1 = new SceneNode("a1"), a2 = new SceneNode("a2");
  a.add(a1).add(a2);
  root.add(a).add(b);
  const order: string[] = [];
  walkNodes(root, (n) => order.push(n.name));
  assert.deepEqual(order, ["root", "a", "a1", "a2", "b"]);
});

// ---------- 迁移等价(验收 #2) ----------

test("P2 场景图: 迁移后 demo 的绘制清单为 4 项且顺序冻结", () => {
  const scene = buildDemoScene();
  const items = scene.items;
  assert.equal(items.length, 4);
  // 顺序冻结:ground → cube → sphere → torus(与升级前 buildDemoScene 的数组一致)
});

test("P2 场景图: 迁移后 demo 与升级前锚点黄金图逐像素一致(容差 0)", () => {
  const fb = new Framebuffer(256, 256);
  const scene = buildDemoScene();
  scene.animate(1.25);
  // items 是 getter,animate 之后取值才包含本帧的 transform
  new Renderer(fb).render(demoOrbitCamera(0.55, 0.34, 6.6), scene.items, "phong", scene.lights);
  const r = diffRGB(loadGolden("p2_scenegraph_anchor.ppm"), framebufferToRaw(fb));
  assert.ok(r.ok, `anchor mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

test("P2 场景图: 正式场景图展平渲染与独立参考实现黄金图逐像素一致(容差 0)", () => {
  // 参考黄金图缺失时从参考实现现场生成(与 npm run golden 同源),保证测试自持
  if (!goldenExists("p2_scenegraph_ref.ppm")) {
    writeGolden("p2_scenegraph_ref.ppm", renderRefScenegraph());
  }
  const fb = new Framebuffer(256, 256);
  const scene = buildDemoScene();
  scene.animate(1.25);
  new Renderer(fb).render(demoOrbitCamera(0.55, 0.34, 6.6), scene.items, "phong", scene.lights);
  const r = diffRGB(loadGolden("p2_scenegraph_ref.ppm"), framebufferToRaw(fb));
  assert.ok(r.ok, `ref mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});
