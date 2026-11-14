/**
 * 关卡编辑器(二期 M5):场景树面板 / 物体拾取 / 移动-旋转 gizmo / 材质面板 / JSON 导入导出。
 * 渲染走 src/ 的纯 CPU 管线;拾取与 gizmo 数学在 src/editor/(node 单测覆盖)。
 * 本文件只做 DOM 绑定与绘制编排(任务书:四功能可用为限,不追求商业级手感)。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { Renderer, type Camera, type DrawItem, type Mesh, type Material } from "../../src/pipeline.js";
import { buildDemoScene, makeCube, makeSphere, makeTorus, makeGround } from "../../src/scene.js";
import { checkerTexture, buildMip1, type Texture } from "../../src/texture.js";
import { SceneNode, makeTransform, updateWorldMatrices, collectDrawItems, walkNodes } from "../../src/scenegraph.js";
import { pickScene } from "../../src/editor/picking.js";
import {
  projectPoint,
  pickGizmoMove, dragMoveDelta, pickGizmoRotate, dragRotateDelta, GIZMO_LEN,
  AXIS_VECTORS, RING_RADIUS,
} from "../../src/editor/gizmo.js";
import { sceneToJSON, sceneFromJSON, type MeshJSON, type TextureJSON, type EditorSceneJSON } from "../../src/editor/serialize.js";
import { drawLine } from "../../src/raster.js";
import type { LightSetup } from "../../src/shading.js";
import { mat4ApplyPoint, type Vec3 } from "../../src/math.js";
import { mulberry32 } from "../../src/rng.js";

/** 新增节点的摆放抖动(确定性;禁 Math.random)。 */
const placeRnd = mulberry32(20261115);

const VIEW = 512;

const canvas = document.getElementById("view") as HTMLCanvasElement;
const ctx = canvas.getContext("2d", { alpha: false }) as CanvasRenderingContext2D;
let W = VIEW;
let H = VIEW;
let imageData = ctx.createImageData(W, H);
let fb = new Framebuffer(W, H, imageData.data);

// ---- 离屏画布(分辨率缩放:内部小缓冲 → 整数放大到画布) ----
const off = document.createElement("canvas");
const offCtx = off.getContext("2d") as CanvasRenderingContext2D;
off.width = W;
off.height = H;
const rendererRef = { r: new Renderer(fb) };

function setScale(s: number): void {
  W = Math.max(64, Math.round(VIEW * s));
  H = W;
  off.width = W;
  off.height = H;
  imageData = offCtx.createImageData(W, H);
  fb = new Framebuffer(W, H, imageData.data);
  rendererRef.r = new Renderer(fb);
}

// ---- 场景 ----
const demo = buildDemoScene();
let root: SceneNode = demo.root;
let lights: LightSetup = demo.lights;
let editorCamera: Camera = {
  eye: { x: 0, y: 4.2, z: 9 },
  target: { x: 0, y: -0.6, z: 0 },
  up: { x: 0, y: 1, z: 0 },
  fovYRad: (50 * Math.PI) / 180,
  aspect: 1,
  near: 0.1,
  far: 120,
};

// 网格/纹理参数登记(JSON 导出需要;导入时重建)
const meshJSONs = new Map<Mesh, MeshJSON>();
const texJSONs = new Map<Texture, TextureJSON>();

function registerDemo(): void {
  const byName = new Map<string, SceneNode>();
  walkNodes(root, (n) => byName.set(n.name, n));
  const g = byName.get("ground")!;
  const c = byName.get("cube")!;
  const s = byName.get("sphere")!;
  const t = byName.get("torus")!;
  meshJSONs.set(g.mesh!, { kind: "ground", size: 30, y: -1.3 });
  meshJSONs.set(c.mesh!, { kind: "cube", size: 0.9 });
  meshJSONs.set(s.mesh!, { kind: "sphere", radius: 0.95, segU: 48, segV: 24 });
  meshJSONs.set(t.mesh!, { kind: "torus", R: 0.85, r: 0.38, segU: 48, segV: 24 });
  texJSONs.set(g.material!.texture!, { kind: "checker", size: 128, cells: 12, a: [190, 184, 172], b: [70, 76, 82] });
  texJSONs.set(c.material!.texture!, { kind: "checker", size: 64, cells: 4, a: [205, 150, 80], b: [138, 94, 50] });
  texJSONs.set(t.material!.texture!, { kind: "stripe", size: 64, stripes: 8, a: [36, 158, 148], b: [232, 230, 224] });
}
registerDemo();

// ---- 选中与交互状态 ----
let selected: SceneNode | null = null;
type Drag =
  | { kind: "move"; axis: 0 | 1 | 2; lastX: number; lastY: number }
  | { kind: "rotate"; axis: 0 | 1 | 2; lastX: number; lastY: number }
  | null;
let drag: Drag = null;

// 相机轨道(右键)
let camYaw = Math.atan2(editorCamera.eye.x, editorCamera.eye.z);
let camPitch = 0.42;
let camDist = 10;
function syncCamera(): void {
  const cp = Math.cos(camPitch);
  editorCamera.eye = {
    x: editorCamera.target.x + camDist * Math.sin(camYaw) * cp,
    y: editorCamera.target.y + camDist * Math.sin(camPitch),
    z: editorCamera.target.z + camDist * Math.cos(camYaw) * cp,
  };
}

// ---- 渲染循环 ----
const fpsEl = document.getElementById("fps") as HTMLSpanElement;
const msEl = document.getElementById("ms") as HTMLSpanElement;
const selInfo = document.getElementById("selinfo") as HTMLSpanElement;
let fpsEma = 60;
let lastT = 0;
let hudT = 0;

function selectedOrigin(): Vec3 | null {
  if (selected === null) return null;
  updateWorldMatrices(root);
  return mat4ApplyPoint(selected.world, { x: 0, y: 0, z: 0 });
}

function drawGizmo(): void {
  const o = selectedOrigin();
  if (o === null) return;
  const colors: Array<[number, number, number]> = [
    [220, 70, 70], [90, 220, 100], [90, 130, 230],
  ];
  // 移动箭头
  for (let axis = 0; axis < 3; axis++) {
    const dir = AXIS_VECTORS[axis]!;
    const a = projectPoint(editorCamera, W, H, o);
    const b = projectPoint(editorCamera, W, H, { x: o.x + dir.x * GIZMO_LEN, y: o.y + dir.y * GIZMO_LEN, z: o.z + dir.z * GIZMO_LEN });
    if (a.behind || b.behind) continue;
    drawLine(fb, Math.round(a.x), Math.round(a.y), Math.round(b.x), Math.round(b.y), colors[axis]![0], colors[axis]![1], colors[axis]![2]);
  }
  // 旋转环(48 段折线)
  for (let axis = 0; axis < 3; axis++) {
    let prev = projectPoint(editorCamera, W, H, ringPoint(axis, o, 0));
    for (let i = 1; i <= 48; i++) {
      const cur = projectPoint(editorCamera, W, H, ringPoint(axis, o, i / 48));
      if (!prev.behind && !cur.behind) {
        drawLine(fb, Math.round(prev.x), Math.round(prev.y), Math.round(cur.x), Math.round(cur.y), colors[axis]![0], colors[axis]![1], colors[axis]![2]);
      }
      prev = cur;
    }
  }
}

function ringPoint(axis: number, c: Vec3, theta: number): Vec3 {
  const a = Math.cos(theta * 2 * Math.PI) * RING_RADIUS;
  const b = Math.sin(theta * 2 * Math.PI) * RING_RADIUS;
  if (axis === 0) return { x: c.x, y: c.y + a, z: c.z + b };
  if (axis === 1) return { x: c.x + a, y: c.y, z: c.z + b };
  return { x: c.x + a, y: c.y + b, z: c.z };
}

function frame(now: number): void {
  const dtMs = lastT === 0 ? 16 : Math.min(100, now - lastT);
  lastT = now;
  demo.animate(now / 1000 % 60);
  syncCamera();
  updateWorldMatrices(root);
  const items: DrawItem[] = collectDrawItems(root);
  fb.clear(24, 26, 32);
  rendererRef.r.render(editorCamera, items, "phong", lights, { sortOpaque: true });
  drawGizmo();
  offCtx.putImageData(imageData, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, VIEW, VIEW);

  fpsEma = fpsEma * 0.9 + (1000 / dtMs) * 0.1;
  if (now - hudT > 250) {
    hudT = now;
    fpsEl.textContent = `${fpsEma.toFixed(1)} fps`;
    msEl.textContent = `${dtMs.toFixed(1)} ms`;
    fpsEl.style.color = fpsEma >= 50 ? "#7fd08a" : fpsEma >= 25 ? "#e2c06c" : "#e27a6c";
  }
  requestAnimationFrame(frame);
}

// ---- 鼠标交互 ----
let buttonDown = -1;

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

canvas.addEventListener("mousedown", (e) => {
  const rect = canvas.getBoundingClientRect();
  const px = ((e.clientX - rect.left) / rect.width) * W;
  const py = ((e.clientY - rect.top) / rect.height) * H;
  buttonDown = e.button;
  if (e.button === 2) return; // 右键:轨道相机
  if (selected !== null) {
    const o = selectedOrigin()!;
    const mv = pickGizmoMove(o, editorCamera, W, H, px, py);
    if (mv !== null) {
      drag = { kind: "move", axis: mv, lastX: px, lastY: py };
      return;
    }
    const rv = pickGizmoRotate(o, editorCamera, W, H, px, py);
    if (rv !== null) {
      drag = { kind: "rotate", axis: rv, lastX: px, lastY: py };
      return;
    }
  }
  // 拾取物体
  const hit = pickScene(root, editorCamera, W, H, px, py);
  select(hit !== null ? hit.node : null);
});

window.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  const px = ((e.clientX - rect.left) / rect.width) * W;
  const py = ((e.clientY - rect.top) / rect.height) * H;
  if (buttonDown === 2) {
    camYaw -= (e.movementX) * 0.005;
    camPitch = Math.max(0.05, Math.min(1.35, camPitch + e.movementY * 0.005));
    return;
  }
  if (drag === null || selected === null) return;
  if (drag.kind === "move") {
    const d = dragMoveDelta(drag.axis, selectedOrigin()!, editorCamera, W, H, drag.lastX, drag.lastY, px, py);
    if (d !== null) {
      selected.transform.position.x += d.x;
      selected.transform.position.y += d.y;
      selected.transform.position.z += d.z;
    }
  } else {
    const ang = dragRotateDelta(drag.axis, selectedOrigin()!, editorCamera, W, H, drag.lastX, drag.lastY, px, py);
    if (ang !== null) {
      const tr = selected.transform.rotationEuler;
      if (drag.axis === 0) tr.x += ang;
      else if (drag.axis === 1) tr.y += ang;
      else tr.z += ang;
    }
  }
  if (drag !== null) {
    drag.lastX = px;
    drag.lastY = py;
  }
  refreshTree();
  refreshMaterial();
});

window.addEventListener("mouseup", () => {
  buttonDown = -1;
  drag = null;
});

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  camDist = Math.max(3, Math.min(30, camDist * Math.exp(e.deltaY * 0.001)));
}, { passive: false });

window.addEventListener("keydown", (e) => {
  if (selected === null) return;
  if (e.key === "v" || e.key === "V") {
    selected.visible = !selected.visible;
    refreshTree();
  }
  if (e.key === "Delete") {
    deleteSelected();
  }
});

// ---- 场景树面板 ----
const treeEl = document.getElementById("tree") as HTMLDivElement;

function refreshTree(): void {
  treeEl.textContent = "";
  walkNodes(root, (n, depth) => {
    const div = document.createElement("div");
    div.className = "tnode" + (n === selected ? " sel" : "") + (n.visible ? "" : " hid");
    div.style.paddingLeft = `${6 + depth * 14}px`;
    div.textContent = (n.mesh !== null ? "◆ " : "· ") + n.name;
    div.addEventListener("click", () => select(n));
    treeEl.appendChild(div);
  });
}

function select(n: SceneNode | null): void {
  selected = n;
  selInfo.textContent = n === null ? "未选中" : `选中:${n.name}`;
  refreshTree();
  refreshMaterial();
}

// ---- 材质面板 ----
const matPanel = document.getElementById("matPanel") as HTMLDivElement;

function refreshMaterial(): void {
  matPanel.textContent = "";
  if (selected === null || selected.material === null) {
    const em = document.createElement("em");
    em.style.cssText = "color:#667; font-size:12px;";
    em.textContent = selected === null ? "选中一个物体后可编辑" : "该节点无材质(纯分组节点)";
    matPanel.appendChild(em);
    return;
  }
  const m = selected.material;
  const slider = (name: string, get: () => number, set: (v: number) => void, min: number, max: number, step: number): void => {
    const lb = document.createElement("label");
    const sp = document.createElement("span");
    sp.className = "name";
    sp.textContent = name;
    const inp = document.createElement("input");
    inp.type = "range";
    inp.min = String(min);
    inp.max = String(max);
    inp.step = String(step);
    inp.value = String(get());
    const val = document.createElement("span");
    val.style.minWidth = "36px";
    val.textContent = get().toFixed(2);
    inp.addEventListener("input", () => {
      set(Number(inp.value));
      val.textContent = Number(inp.value).toFixed(2);
    });
    lb.append(sp, inp, val);
    matPanel.appendChild(lb);
  };
  slider("albedo R", () => m.albedo[0], (v) => { m.albedo[0] = v; }, 0, 1, 0.01);
  slider("albedo G", () => m.albedo[1], (v) => { m.albedo[1] = v; }, 0, 1, 0.01);
  slider("albedo B", () => m.albedo[2], (v) => { m.albedo[2] = v; }, 0, 1, 0.01);
  slider("specular", () => m.specular, (v) => { m.specular = v; }, 0, 1, 0.01);
  slider("shininess", () => m.shininess, (v) => { m.shininess = v; }, 1, 128, 1);
  slider("mip", () => (m.mip === true ? 1 : 0), (v) => {
    m.mip = v === 1;
    if (m.texture !== undefined && m.mip) m.texture.mip = buildMip1(m.texture);
  }, 0, 1, 1);

  const modelLb = document.createElement("label");
  modelLb.innerHTML = `<span class="name">着色模型</span>`;
  const modelSel = document.createElement("select");
  modelSel.innerHTML = `<option value="blinnPhong">Blinn-Phong</option><option value="lambert">Lambert</option>`;
  modelSel.value = m.shadingModel ?? "blinnPhong";
  modelSel.addEventListener("change", () => { m.shadingModel = modelSel.value as "lambert" | "blinnPhong"; });
  modelLb.appendChild(modelSel);
  matPanel.appendChild(modelLb);

  const cullLb = document.createElement("label");
  cullLb.innerHTML = `<span class="name">背面剔除</span><input type="checkbox" ${m.cullBackface !== false ? "checked" : ""}>`;
  const cb = cullLb.querySelector("input") as HTMLInputElement;
  cb.addEventListener("change", () => { m.cullBackface = cb.checked; });
  matPanel.appendChild(cullLb);
}

// ---- 增删节点 ----
function uniqueName(base: string): string {
  const names = new Set<string>();
  walkNodes(root, (n) => names.add(n.name));
  if (!names.has(base)) return base;
  let i = 2;
  while (names.has(`${base}${i}`)) i++;
  return `${base}${i}`;
}

function addPrimitive(kind: "cube" | "sphere" | "torus"): void {
  const name = uniqueName(kind);
  let mesh: Mesh;
  let spec: MeshJSON;
  if (kind === "cube") { mesh = makeCube(1); spec = { kind: "cube", size: 1 }; }
  else if (kind === "sphere") { mesh = makeSphere(0.7, 32, 16); spec = { kind: "sphere", radius: 0.7, segU: 32, segV: 16 }; }
  else { mesh = makeTorus(0.6, 0.25, 32, 16); spec = { kind: "torus", R: 0.6, r: 0.25, segU: 32, segV: 16 }; }
  const mat: Material = { albedo: [0.7, 0.7, 0.75], specular: 0.3, shininess: 24, texture: checkerTexture(32, 4, [190, 150, 90], [80, 62, 44]), sampler: "bilinear" };
  // 新节点摆放抖动用确定性 PRNG(项目纪律:禁 Math.random)
  const node = new SceneNode(name, {
    transform: makeTransform({ x: Math.floor(placeRnd() * 8 - 4), y: 0.5, z: Math.floor(placeRnd() * 8 - 4) }),
    mesh, material: mat,
  });
  meshJSONs.set(mesh, spec);
  texJSONs.set(mat.texture!, { kind: "checker", size: 32, cells: 4, a: [190, 150, 90], b: [80, 62, 44] });
  root.add(node);
  select(node);
}

function deleteSelected(): void {
  if (selected === null || selected === root) return;
  const remove = (parent: SceneNode): boolean => {
    const idx = parent.children.indexOf(selected!);
    if (idx >= 0) {
      parent.children.splice(idx, 1);
      return true;
    }
    return parent.children.some(remove);
  };
  remove(root);
  select(null);
}

document.getElementById("addCube")!.addEventListener("click", () => addPrimitive("cube"));
document.getElementById("addSphere")!.addEventListener("click", () => addPrimitive("sphere"));
document.getElementById("addTorus")!.addEventListener("click", () => addPrimitive("torus"));
document.getElementById("toggleVis")!.addEventListener("click", () => {
  if (selected !== null) {
    selected.visible = !selected.visible;
    refreshTree();
  }
});
document.getElementById("delNode")!.addEventListener("click", deleteSelected);

// ---- JSON 导入导出 ----
const jsonBox = document.getElementById("jsonBox") as HTMLTextAreaElement;

document.getElementById("exportBtn")!.addEventListener("click", () => {
  const json = sceneToJSON({ root, camera: editorCamera, lights, meshRegistry: meshJSONs, textureRegistry: texJSONs });
  jsonBox.value = JSON.stringify(json, null, 2);
});

document.getElementById("importBtn")!.addEventListener("click", () => {
  try {
    const json = JSON.parse(jsonBox.value) as EditorSceneJSON;
    const built = sceneFromJSON(json);
    root = built.root;
    lights = built.lights;
    editorCamera = built.camera;
    camYaw = Math.atan2(editorCamera.eye.x, editorCamera.eye.z);
    camPitch = Math.asin((editorCamera.eye.y - editorCamera.target.y) / Math.max(0.001, Math.hypot(editorCamera.eye.x - editorCamera.target.x, editorCamera.eye.y - editorCamera.target.y, editorCamera.eye.z - editorCamera.target.z)));
    camDist = Math.hypot(editorCamera.eye.x - editorCamera.target.x, editorCamera.eye.y - editorCamera.target.y, editorCamera.eye.z - editorCamera.target.z);
    // 重建登记表:遍历新树,与 JSON 描述对齐(同名路径)
    registerFromJSON(json, built.root);
    select(null);
  } catch (err) {
    alert(`导入失败:${err instanceof Error ? err.message : String(err)}`);
  }
});

function registerFromJSON(json: EditorSceneJSON, node: SceneNode, nj = json.root): void {
  if (node.mesh !== null && nj.mesh !== null) meshJSONs.set(node.mesh, nj.mesh);
  if (node.material?.texture !== undefined && nj.material?.texture != null) texJSONs.set(node.material.texture, nj.material.texture);
  for (let i = 0; i < node.children.length; i++) {
    const cj = nj.children[i];
    if (cj !== undefined) registerFromJSON(json, node.children[i]!, cj);
  }
}

document.getElementById("downloadBtn")!.addEventListener("click", () => {
  const json = sceneToJSON({ root, camera: editorCamera, lights, meshRegistry: meshJSONs, textureRegistry: texJSONs });
  const blob = new Blob([JSON.stringify(json, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "scene.json";
  a.click();
  URL.revokeObjectURL(a.href);
});

document.getElementById("loadFileBtn")!.addEventListener("click", () => {
  (document.getElementById("fileInput") as HTMLInputElement).click();
});
document.getElementById("fileInput")!.addEventListener("change", async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file === undefined) return;
  jsonBox.value = await file.text();
  (document.getElementById("importBtn") as HTMLButtonElement).click();
});

// ---- 分辨率缩放 ----
document.getElementById("scale")!.addEventListener("change", (e) => {
  setScale(Number((e.target as HTMLSelectElement).value));
});

// ---- 启动 ----
refreshTree();
refreshMaterial();
requestAnimationFrame(frame);
void makeGround;
