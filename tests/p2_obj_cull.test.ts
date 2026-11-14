import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Framebuffer } from "../src/framebuffer.js";
import { Renderer, type DrawItem } from "../src/pipeline.js";
import { parseOBJ } from "../src/obj.js";
import { meshBoundingSphere, transformSphere, sphereOutsidePlanes, frustumPlanes, shouldCull } from "../src/culling.js";
import { checkerTexture } from "../src/texture.js";
import { mat4Identity, mat4Translation, mat4Scaling, mat4LookAt, mat4Perspective } from "../src/math.js";
import { decodePPM, framebufferToRaw, type RawImage } from "../src/ppm.js";
import { diffRGB } from "../src/diff.js";
import { makeSphere } from "../src/scene.js";
import type { LightSetup } from "../src/shading.js";
import {
  P2O_CAMERA, P2O_LIGHTS, P2O_TEX, P2O_MATERIAL, loadFixtureText,
} from "../tools/reference/gen_p2_obj.js";
import { P2C_CAMERA, P2C_LIGHTS, P2C_BOX_MATERIAL, P2C_SPHERE_MATERIAL, openBoxMesh } from "../tools/reference/gen_p2_cull.js";
import { P2F_CAMERA, P2F_LIGHTS, P2F_ITEMS } from "../tools/reference/gen_p2_frustum.js";
import type { RefLights } from "../tools/reference/ref_render.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "..", "..", "golden");

function loadGolden(name: string): RawImage {
  return decodePPM(new Uint8Array(readFileSync(join(GOLDEN_DIR, name))));
}

/** 参考侧光照(RefLights)→ 正式 LightSetup(补 spots 空数组)。 */
function lightsOf(l: RefLights): LightSetup {
  return { ambient: l.ambient, dirs: l.dirs, points: l.points, spots: [] };
}

function P2C_SPHERE_MESH(): ReturnType<typeof makeSphere> {
  return makeSphere(0.8, 32, 16);
}

// ---------- OBJ 解析器 ----------

test("P2 OBJ: 夹具解析 —— 四种索引形态、负索引、扇形三角化、顶点展开", () => {
  const m = parseOBJ(loadFixtureText());
  // 输出 = 展开块:底面 2 tri + 4 侧面 + 贴图 2 tri = 8 tri × 3 = 24 顶点
  assert.equal(m.indices.length, 24);
  assert.equal(m.positions.length / 3, 24);
  // 负索引 -1 解析到第 5 个"v"指令(尖顶):侧面展开的第三个角点是 (0,1,0)
  // 展开块布局:底面 6(0-5)、侧面 12(6-17)、贴图 6(18-23);第一个侧面第三角点 = 顶点 8
  const apex = m.positions.slice(24, 27);
  assert.ok(apex[0] === 0 && apex[1] === 1 && apex[2] === 0, `apex via negative index: ${apex}`);
  // 索引合法
  const vc = m.positions.length / 3;
  for (const i of m.indices) assert.ok(i >= 0 && i < vc);
  // 无 vn 的四边形用几何法线(0,0,1) —— 朝相机
  const quadTri0 = m.indices[18];
  assert.ok(Math.abs(m.normals[quadTri0 * 3]) < 1e-9 && Math.abs(m.normals[quadTri0 * 3 + 1]) < 1e-9);
  assert.ok(Math.abs(m.normals[quadTri0 * 3 + 2] - 1) < 1e-9);
});

test("P2 OBJ: 缺 vt 补 (0,0);越界索引抛错;空面跳过;未知指令忽略", () => {
  const m1 = parseOBJ("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
  // 展开语义:3 原始顶点保留,3 个展开角点追加,uvs = 3 原始(缺 vt 行→空)+3 展开(0,0)
  assert.equal(m1.indices.length, 3);
  assert.deepEqual([...m1.uvs.slice(-6)], [0, 0, 0, 0, 0, 0]);
  assert.throws(() => parseOBJ("v 0 0 0\nf 1 2 5\n"), /out of range/);
  assert.throws(() => parseOBJ("f 1 2 3\n"), /out of range/);
  const m2 = parseOBJ("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2\nusemtl x\nl 1 2\np 1\n");
  assert.equal(m2.indices.length, 0);
  const m3 = parseOBJ("v 0 0 0\nv 1 0 0\nv 0 1 0\nv -1 0 0\nf -4 -3 -2\n"); // 负索引
  assert.equal(m3.indices.length, 3);
  // 展开角点引用原始顶点 0,1,2(-4→0, -3→1, -2→2)
  const c0 = m3.indices[0];
  assert.equal(m3.positions[c0 * 3], 0);
});

test("P2 OBJ: 解析结果可直接渲染(索引/法线/uv 形态合法)", () => {
  const m = parseOBJ(loadFixtureText());
  const vc = m.positions.length / 3;
  for (let i = 0; i < vc; i++) {
    const nl = Math.hypot(m.normals[i * 3], m.normals[i * 3 + 1], m.normals[i * 3 + 2]);
    // 夹具 vn 是工程近似值(如 0.894,0.447,0),容差放宽到 1e-3;几何法线精确单位
    assert.ok(Math.abs(nl - 1) < 1e-3, `normal unit at ${i}: ${nl}`);
  }
});

// ---------- OBJ 黄金对拍 ----------

test("P2 OBJ: 加载夹具渲染与独立参考实现逐像素一致(容差 0)", () => {
  const fb = new Framebuffer(128, 128);
  const mesh = parseOBJ(loadFixtureText());
  const tex = checkerTexture(P2O_TEX.size, P2O_TEX.cells, P2O_TEX.a, P2O_TEX.b);
  const items: DrawItem[] = [{ mesh, model: mat4Identity(), material: { ...P2O_MATERIAL, texture: tex, sampler: "bilinear" } }];
  new Renderer(fb).render(P2O_CAMERA, items, "phong", lightsOf(P2O_LIGHTS));
  const r = diffRGB(loadGolden("p2_obj.ppm"), framebufferToRaw(fb));
  assert.ok(r.ok, `obj golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
});

// ---------- 背面剔除开关 ----------

test("P2 剔除: cull ON 只见红球,cull OFF 内壁可见 —— 两态黄金各自对拍且两图不同", () => {
  const itemsFor = (cull: boolean): DrawItem[] => [
    { mesh: openBoxMesh(), model: mat4Identity(), material: { ...P2C_BOX_MATERIAL, sampler: "bilinear", cullBackface: cull } },
    { mesh: P2C_SPHERE_MESH(), model: mat4Translation({ x: 0, y: 0, z: -2 }), material: { ...P2C_SPHERE_MATERIAL, cullBackface: true } },
  ];
  const renderWith = (cull: boolean): RawImage => {
    const fb = new Framebuffer(128, 128);
    new Renderer(fb).render(P2C_CAMERA, itemsFor(cull), "phong", lightsOf(P2C_LIGHTS));
    return framebufferToRaw(fb);
  };
  const on = diffRGB(loadGolden("p2_cull_on.ppm"), renderWith(true));
  const off = diffRGB(loadGolden("p2_cull_off.ppm"), renderWith(false));
  assert.ok(on.ok, `cull-on golden mismatch: ${on.message} first=${JSON.stringify(on.first)}`);
  assert.ok(off.ok, `cull-off golden mismatch: ${off.message} first=${JSON.stringify(off.first)}`);
  assert.ok(!diffRGB(loadGolden("p2_cull_on.ppm"), loadGolden("p2_cull_off.ppm")).ok, "on/off goldens must differ");
});

// ---------- 视锥剔除 ----------

test("P2 剔除: 视锥剔除默认开 —— 与参考全量渲染逐像素一致(剔除只跳过 0 像素项)", () => {
  const fb = new Framebuffer(128, 128);
  const items: DrawItem[] = P2F_ITEMS.map((it) => ({ mesh: it.mesh, model: it.model, material: it.material }));
  new Renderer(fb).render(P2F_CAMERA, items, "phong", lightsOf(P2F_LIGHTS));
  const r = diffRGB(loadGolden("p2_frustum.ppm"), framebufferToRaw(fb));
  assert.ok(r.ok, `frustum golden mismatch: ${r.message} first=${JSON.stringify(r.first)}`);
  // 且确实剔除了:左外球(第 4 项)与远外球(第 5 项)中恰一项被剔除
  const view = mat4LookAt(P2F_CAMERA.eye, P2F_CAMERA.target, P2F_CAMERA.up);
  const proj = mat4Perspective(P2F_CAMERA.fovYRad, P2F_CAMERA.aspect, P2F_CAMERA.near, P2F_CAMERA.far);
  const culls = P2F_ITEMS.map((it) => shouldCull(it.mesh, it.model, view, proj));
  assert.equal(culls[3], true, "left-outside sphere must be culled");
  assert.equal(culls[4], false, "beyond-far sphere must NOT be culled (far plane is not clipped)");
  assert.equal(culls[0], false, "center sphere kept");
  assert.equal(culls[1], false, "right cube kept");
  assert.equal(culls[2], false, "straddling sphere kept");
});

test("P2 剔除: 包围球数学锚定 —— AABB 中心、最远顶点半径、仿射变换、平面外判定", () => {
  const mesh = { positions: [1, 0, 0, -1, 0, 0, 0, 2, 0, 0, -2, 0], normals: [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], uvs: [0, 0, 0, 0, 0, 0, 0, 0], indices: [0, 1, 2] };
  const s = meshBoundingSphere(mesh);
  // AABB:y ∈ [-2,2],x ∈ [-1,1] → 中心 (0,0,0);最远顶点 (0,±2,0) → r = 2
  assert.ok(s.cx === 0 && s.cy === 0 && s.cz === 0);
  assert.ok(Math.abs(s.r - 2) < 1e-12);
  const m = mat4Scaling({ x: 2, y: 2, z: 2 });
  const ws = transformSphere(s, m);
  assert.ok(Math.abs(ws.r - 2 * 2 * Math.sqrt(3)) < 1e-9); // Frobenius(2I)=2√3
  // 平面外判定:单位球在 +x 远处,看向 -z 的相机 left/right 平面
  const view = mat4LookAt({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: -1 }, { x: 0, y: 1, z: 0 });
  const proj = mat4Perspective((50 * Math.PI) / 180, 1, 0.1, 100);
  const planes = frustumPlanes(view, proj);
  assert.equal(sphereOutsidePlanes({ cx: 50, cy: 0, cz: -50, r: 1 }, planes), true, "off-right sphere culled");
  assert.equal(sphereOutsidePlanes({ cx: 0, cy: 0, cz: -5, r: 1 }, planes), false, "center sphere kept");
  assert.equal(sphereOutsidePlanes({ cx: 0, cy: 0, cz: 5, r: 1 }, planes), true, "behind-camera sphere culled");
});
