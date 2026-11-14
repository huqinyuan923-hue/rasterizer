/**
 * M1(二期)参考实现:场景图 —— TRS 合成、层级世界矩阵、DFS 展平,全部独立编码。
 * 已被一期黄金图仲裁的部分(数学库 mat4*、光栅管线 Renderer)按纪律复用为夹具。
 *
 * 黄金图:p2_scenegraph_ref.ppm —— 参考场景图展平出的绘制清单喂给 Renderer,
 * 与正式实现(ScrapeNode/composeTRS/collectDrawItems)展平出的清单渲染必须逐像素一致。
 * 另有 p2_scenegraph_anchor.ppm(升级前硬编码场景,由 gen_p2_anchor 一次性生成)作迁移等价锚。
 */

import { Framebuffer } from "../../src/framebuffer.js";
import { Renderer, type DrawItem, type Material, type Mesh } from "../../src/pipeline.js";
import {
  mat4Mul, mat4RotX, mat4RotY, mat4RotZ, mat4Scaling, mat4Translation, type Mat4, type Vec3,
} from "../../src/math.js";
import { makeCube, makeSphere, makeTorus, stripeTexture } from "../../src/scene.js";
import { checkerTexture } from "../../src/texture.js";
import { framebufferToRaw, type RawImage } from "../../src/ppm.js";
import { writeGolden } from "../goldens.js";

const W = 256;
const H = 256;
const ANIM_T = 1.25;

// ---- 参考场景图节点(独立于 src/scenegraph.ts 的结构) ----

interface RefNode {
  name: string;
  pos: Vec3;
  rot: Vec3; // 欧拉角(弧度),应用顺序 X→Y→Z
  scl: Vec3;
  mesh: Mesh | null;
  material: Material | null;
  visible: boolean;
  kids: RefNode[];
  world: Mat4;
}

function refNode(name: string, opts: Partial<RefNode> = {}): RefNode {
  return {
    name,
    pos: opts.pos ?? { x: 0, y: 0, z: 0 },
    rot: opts.rot ?? { x: 0, y: 0, z: 0 },
    scl: opts.scl ?? { x: 1, y: 1, z: 1 },
    mesh: opts.mesh ?? null,
    material: opts.material ?? null,
    visible: opts.visible ?? true,
    kids: opts.kids ?? [],
    world: new Float64Array(16),
  };
}

/** 参考 TRS:local = ((T·Rz)·Ry)·Rx·S(冻结公式,独立写法)。 */
function refComposeLocal(n: RefNode): Mat4 {
  const T = mat4Translation({ x: n.pos.x, y: n.pos.y, z: n.pos.z });
  const Rz = mat4RotZ(n.rot.z);
  const Ry = mat4RotY(n.rot.y);
  const Rx = mat4RotX(n.rot.x);
  const S = mat4Scaling({ x: n.scl.x, y: n.scl.y, z: n.scl.z });
  const tz = mat4Mul(T, Rz);
  const zy = mat4Mul(tz, Ry);
  const yx = mat4Mul(zy, Rx);
  return mat4Mul(yx, S);
}

/** 参考遍历:先序 DFS,world = parent.world · local。 */
function refUpdateWorld(n: RefNode, parent: Mat4): void {
  n.world = mat4Mul(parent, refComposeLocal(n));
  for (const k of n.kids) refUpdateWorld(k, n.world);
}

/** 参考展平:visible=false 整棵子树跳过;先序收集可绘制节点。 */
function refCollect(n: RefNode, out: DrawItem[]): void {
  if (!n.visible) return;
  if (n.mesh && n.material) out.push({ mesh: n.mesh, model: n.world, material: n.material });
  for (const k of n.kids) refCollect(k, out);
}

// ---- 参考演示场景(与升级前 demo 逐值一致,动画时刻冻结在 ANIM_T) ----

function refScene(): { root: RefNode; lights: Parameters<Renderer["render"]>[3] } {
  const groundTex = checkerTexture(128, 12, [190, 184, 172], [70, 76, 82]);
  const crateTex = checkerTexture(64, 4, [205, 150, 80], [138, 94, 50]);
  const torusTex = stripeTexture(64, 8, [36, 158, 148], [232, 230, 224]);

  const groundMesh: Mesh = {
    positions: [-15, -1.3, -15, 15, -1.3, -15, 15, -1.3, 15, -15, -1.3, 15],
    normals: [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    uvs: [0, 0, 1, 0, 1, 1, 0, 1],
    indices: [0, 2, 1, 0, 3, 2],
  };

  const ground = refNode("ground", {
    mesh: groundMesh,
    material: { albedo: [1, 1, 1], specular: 0.06, shininess: 8, texture: groundTex, sampler: "bilinear", doubleSided: true },
  });
  const cube = refNode("cube", {
    pos: { x: -1.7, y: -0.85, z: 0.2 },
    rot: { x: 0, y: ANIM_T * 0.5, z: 0 },
    mesh: makeCube(0.9),
    material: { albedo: [1, 1, 1], specular: 0.35, shininess: 24, texture: crateTex, sampler: "nearest" },
  });
  const sphere = refNode("sphere", {
    pos: { x: 0.05, y: -0.35, z: 0 },
    mesh: makeSphere(0.95, 48, 24),
    material: { albedo: [0.85, 0.3, 0.22], specular: 0.6, shininess: 48 },
  });
  const torus = refNode("torus", {
    pos: { x: 1.9, y: -0.92, z: -0.1 },
    rot: { x: -0.14, y: ANIM_T * 0.7, z: 0 },
    mesh: makeTorus(0.85, 0.38, 48, 24),
    material: { albedo: [1, 1, 1], specular: 0.85, shininess: 64, texture: torusTex, sampler: "bilinear" },
  });
  const root = refNode("root", { kids: [ground, cube, sphere, torus] });

  const lights = {
    ambient: [0.12, 0.12, 0.14] as [number, number, number],
    dirs: [{ dir: [0.45, 0.8, 0.5] as [number, number, number], color: [0.85, 0.82, 0.75] as [number, number, number] }],
    points: [{ pos: { x: 3.2, y: 2.4, z: 0 }, color: [1.0, 0.78, 0.55] as [number, number, number], k1: 0.08, k2: 0.02 }],
    spots: [] as never[],
  };
  {
    const d = lights.dirs[0].dir;
    const l = Math.hypot(d[0], d[1], d[2]);
    d[0] /= l; d[1] /= l; d[2] /= l;
  }
  lights.points[0].pos.x = 3.2 * Math.cos(ANIM_T * 0.6);
  lights.points[0].pos.y = 2.4 + 0.5 * Math.sin(ANIM_T * 0.9);
  lights.points[0].pos.z = 3.2 * Math.sin(ANIM_T * 0.6);

  return { root, lights };
}

// ---- 相机(demoOrbitCamera(0.55, 0.34, 6.6) 冻结值,独立构造) ----

function refCamera() {
  const yaw = 0.55, pitch = 0.34, dist = 6.6;
  const target = { x: 0, y: -0.3, z: 0 };
  const cp = Math.cos(pitch);
  return {
    eye: {
      x: target.x + dist * Math.sin(yaw) * cp,
      y: target.y + dist * Math.sin(pitch),
      z: target.z + dist * Math.cos(yaw) * cp,
    },
    target,
    up: { x: 0, y: 1, z: 0 },
    fovYRad: (55 * Math.PI) / 180,
    aspect: 1,
    near: 0.1,
    far: 60,
  };
}

export function renderRefScenegraph(): RawImage {
  const fb = new Framebuffer(W, H);
  const { root, lights } = refScene();
  refUpdateWorld(root, new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]));
  const items: DrawItem[] = [];
  refCollect(root, items);
  new Renderer(fb).render(refCamera(), items, "phong", lights);
  return framebufferToRaw(fb);
}

export function run(): void {
  writeGolden("p2_scenegraph_ref.ppm", renderRefScenegraph());
  console.log("p2_scenegraph_ref.ppm written");
}
