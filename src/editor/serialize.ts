/**
 * 场景序列化（二期 M5）：场景树 + 相机 + 光照 ↔ JSON。
 *
 * 冻结规则：
 * - 数字经 JSON.stringify/parse 往返精确（IEEE 754 最短表示可逆）。
 * - 网格只支持程序化种类（cube/sphere/torus/ground）与内嵌 OBJ 文本（kind:"obj"）。
 * - 纹理只支持程序化种类（checker/stripe）；material.mip = true 时重建时附 mip 一档。
 * - 重建全部走与运行时相同的生成器（确定性），round-trip 渲染逐像素一致（验收 #3）。
 */

import { makeCube, makeSphere, makeTorus, makeGround, stripeTexture } from "../scene.js";
import { checkerTexture, buildMip1, type Texture } from "../texture.js";
import { parseOBJ } from "../obj.js";
import type { Camera, Material, Mesh } from "../pipeline.js";
import type { LightSetup } from "../shading.js";
import { SceneNode, makeTransform, type NodeTransform } from "../scenegraph.js";
import type { Vec3 } from "../math.js";

// ---------- JSON 形态 ----------

export type MeshJSON =
  | { kind: "cube"; size: number }
  | { kind: "sphere"; radius: number; segU: number; segV: number }
  | { kind: "torus"; R: number; r: number; segU: number; segV: number }
  | { kind: "ground"; size: number; y: number }
  | { kind: "obj"; obj: string };

export type TextureJSON =
  | { kind: "checker"; size: number; cells: number; a: [number, number, number]; b: [number, number, number] }
  | { kind: "stripe"; size: number; stripes: number; a: [number, number, number]; b: [number, number, number] };

export interface MaterialJSON {
  albedo: [number, number, number];
  specular: number;
  shininess: number;
  shadingModel?: "lambert" | "blinnPhong";
  sampler?: "nearest" | "bilinear";
  cullBackface?: boolean;
  mip?: boolean;
  texture?: TextureJSON | null;
  doubleSided?: boolean; // 遗留别名(读入兼容;写出用 cullBackface)
}

export interface NodeJSON {
  name: string;
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
  visible: boolean;
  mesh: MeshJSON | null;
  material: MaterialJSON | null;
  children: NodeJSON[];
}

export interface CameraJSON {
  eye: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  fovYRad: number;
  aspect: number;
  near: number;
  far: number;
  projection?: "perspective" | "ortho";
  left?: number;
  right?: number;
  bottom?: number;
  top?: number;
}

export interface LightsJSON {
  ambient: [number, number, number];
  dirs: Array<{ dir: [number, number, number]; color: [number, number, number] }>;
  points: Array<{ pos: [number, number, number]; color: [number, number, number]; k1: number; k2: number }>;
  spots: Array<{ pos: [number, number, number]; dir: [number, number, number]; color: [number, number, number]; cosInner: number; cosOuter: number; k1: number; k2: number }>;
}

export interface EditorSceneJSON {
  version: 2;
  camera: CameraJSON;
  lights: LightsJSON;
  root: NodeJSON;
}

// ---------- 编码 ----------

function vec3ToArray(v: Vec3): [number, number, number] {
  return [v.x, v.y, v.z];
}

function transformToJSON(tr: NodeTransform): Pick<NodeJSON, "position" | "rotation" | "scale"> {
  return {
    position: vec3ToArray(tr.position),
    rotation: vec3ToArray(tr.rotationEuler),
    scale: vec3ToArray(tr.scale),
  };
}

/** 纹理无法从运行时对象反推参数,由调用方随材质登记;此处按"程序化描述"编码。 */
export interface TextureRegistryEntry {
  spec: TextureJSON;
  texture: Texture;
}

/**
 * 编码材质纹理:在 registry 里按对象引用查找对应参数描述。
 * 未登记的纹理 → null(材质保留其余参数,纹理丢弃 —— round-trip 会缺纹理,调用方应登记)。
 */
function textureToJSON(texture: Texture | undefined, registry: Map<Texture, TextureJSON>): TextureJSON | null {
  if (texture === undefined) return null;
  return registry.get(texture) ?? null;
}

export function materialToJSON(m: Material, registry: Map<Texture, TextureJSON>): MaterialJSON {
  const out: MaterialJSON = {
    albedo: [m.albedo[0], m.albedo[1], m.albedo[2]],
    specular: m.specular,
    shininess: m.shininess,
  };
  if (m.shadingModel !== undefined) out.shadingModel = m.shadingModel;
  if (m.sampler !== undefined) out.sampler = m.sampler;
  if (m.cullBackface !== undefined) out.cullBackface = m.cullBackface;
  else if (m.doubleSided !== undefined) out.cullBackface = !m.doubleSided;
  if (m.mip !== undefined) out.mip = m.mip;
  out.texture = textureToJSON(m.texture, registry);
  return out;
}

function nodeToJSON(node: SceneNode, registry: Map<Texture, TextureJSON>): NodeJSON {
  const json: NodeJSON = {
    name: node.name,
    ...transformToJSON(node.transform),
    visible: node.visible,
    mesh: null,
    material: node.material !== null ? materialToJSON(node.material, registry) : null,
    children: node.children.map((c) => nodeToJSON(c, registry)),
  };
  return json;
}

/**
 * 编码网格:mesh 来源必须是本模块 buildNode 的程序化网格或 OBJ 解析结果。
 * 网格运行时对象不携带参数,由调用方用 registry(Map<Mesh, MeshJSON>)登记。
 */
export function meshToJSON(mesh: Mesh, registry: Map<Mesh, MeshJSON>): MeshJSON | null {
  return registry.get(mesh) ?? null;
}

export interface SerializeInput {
  root: SceneNode;
  camera: Camera;
  lights: LightSetup;
  /** mesh/texture 运行时对象 → 参数描述(调用方在构建场景时登记)。 */
  meshRegistry: Map<Mesh, MeshJSON>;
  textureRegistry: Map<Texture, TextureJSON>;
}

export function sceneToJSON(input: SerializeInput): EditorSceneJSON {
  const root = nodeToJSON(input.root, input.textureRegistry);
  // 网格 registry 单独回填(避免 nodeToJSON 泛化)
  fillMeshes(input.root, input.meshRegistry, root);
  const cam = input.camera;
  return {
    version: 2,
    camera: {
      eye: vec3ToArray(cam.eye),
      target: vec3ToArray(cam.target),
      up: vec3ToArray(cam.up),
      fovYRad: cam.fovYRad,
      aspect: cam.aspect,
      near: cam.near,
      far: cam.far,
      ...(cam.projection !== undefined ? { projection: cam.projection } : {}),
      ...(cam.left !== undefined ? { left: cam.left } : {}),
      ...(cam.right !== undefined ? { right: cam.right } : {}),
      ...(cam.bottom !== undefined ? { bottom: cam.bottom } : {}),
      ...(cam.top !== undefined ? { top: cam.top } : {}),
    },
    lights: {
      ambient: [input.lights.ambient[0], input.lights.ambient[1], input.lights.ambient[2]],
      dirs: input.lights.dirs.map((d) => ({ dir: [d.dir[0], d.dir[1], d.dir[2]] as [number, number, number], color: [d.color[0], d.color[1], d.color[2]] as [number, number, number] })),
      points: input.lights.points.map((p) => ({ pos: vec3ToArray(p.pos), color: [p.color[0], p.color[1], p.color[2]] as [number, number, number], k1: p.k1, k2: p.k2 })),
      spots: input.lights.spots.map((s) => ({ pos: vec3ToArray(s.pos), dir: [s.dir[0], s.dir[1], s.dir[2]] as [number, number, number], color: [s.color[0], s.color[1], s.color[2]] as [number, number, number], cosInner: s.cosInner, cosOuter: s.cosOuter, k1: s.k1, k2: s.k2 })),
    },
    root,
  };
}

/** 与 nodeToJSON 平行的第二遍(按同名路径回填 mesh) —— 树结构确定性保证对齐。 */
function fillMeshes(node: SceneNode, registry: Map<Mesh, MeshJSON>, json: NodeJSON): void {
  json.mesh = node.mesh !== null ? (registry.get(node.mesh) ?? null) : null;
  for (let i = 0; i < node.children.length; i++) {
    const childJSON = json.children[i];
    if (childJSON !== undefined) fillMeshes(node.children[i]!, registry, childJSON);
  }
}

// ---------- 解码 ----------

function buildMesh(spec: MeshJSON): Mesh {
  if (spec.kind === "cube") return makeCube(spec.size);
  if (spec.kind === "sphere") return makeSphere(spec.radius, spec.segU, spec.segV);
  if (spec.kind === "torus") return makeTorus(spec.R, spec.r, spec.segU, spec.segV);
  if (spec.kind === "ground") return makeGround(spec.size, spec.y);
  if (spec.kind === "obj") return parseOBJ(spec.obj);
  throw new Error(`serialize: unknown mesh kind ${(spec as { kind?: string }).kind}`);
}

export function buildTexture(spec: TextureJSON): Texture {
  const t = spec.kind === "checker"
    ? checkerTexture(spec.size, spec.cells, spec.a, spec.b)
    : stripeTexture(spec.size, spec.stripes, spec.a, spec.b);
  return t;
}

function buildMaterial(json: MaterialJSON): Material {
  const m: Material = {
    albedo: [json.albedo[0], json.albedo[1], json.albedo[2]],
    specular: json.specular,
    shininess: json.shininess,
  };
  if (json.shadingModel !== undefined) m.shadingModel = json.shadingModel;
  if (json.sampler !== undefined) m.sampler = json.sampler;
  if (json.cullBackface !== undefined) m.cullBackface = json.cullBackface;
  else if (json.doubleSided !== undefined) m.cullBackface = !json.doubleSided;
  if (json.mip !== undefined) m.mip = json.mip;
  if (json.texture != null) {
    const tex = buildTexture(json.texture);
    if (json.mip === true) tex.mip = buildMip1(tex);
    m.texture = tex;
  }
  return m;
}

function buildNode(json: NodeJSON): SceneNode {
  const node = new SceneNode(json.name, {
    transform: makeTransform({ x: json.position[0], y: json.position[1], z: json.position[2] }),
    visible: json.visible,
    mesh: json.mesh !== null ? buildMesh(json.mesh) : null,
    material: json.material !== null ? buildMaterial(json.material) : null,
  });
  node.transform.rotationEuler = { x: json.rotation[0], y: json.rotation[1], z: json.rotation[2] };
  node.transform.scale = { x: json.scale[0], y: json.scale[1], z: json.scale[2] };
  for (const child of json.children) {
    node.add(buildNode(child));
  }
  return node;
}

export function sceneFromJSON(json: EditorSceneJSON): { root: SceneNode; camera: Camera; lights: LightSetup } {
  const c = json.camera;
  const camera: Camera = {
    eye: { x: c.eye[0], y: c.eye[1], z: c.eye[2] },
    target: { x: c.target[0], y: c.target[1], z: c.target[2] },
    up: { x: c.up[0], y: c.up[1], z: c.up[2] },
    fovYRad: c.fovYRad,
    aspect: c.aspect,
    near: c.near,
    far: c.far,
  };
  if (c.projection !== undefined) camera.projection = c.projection;
  if (c.left !== undefined) camera.left = c.left;
  if (c.right !== undefined) camera.right = c.right;
  if (c.bottom !== undefined) camera.bottom = c.bottom;
  if (c.top !== undefined) camera.top = c.top;

  const lights: LightSetup = {
    ambient: [json.lights.ambient[0], json.lights.ambient[1], json.lights.ambient[2]],
    dirs: json.lights.dirs.map((d) => ({ dir: [d.dir[0], d.dir[1], d.dir[2]] as [number, number, number], color: [d.color[0], d.color[1], d.color[2]] as [number, number, number] })),
    points: json.lights.points.map((p) => ({ pos: { x: p.pos[0], y: p.pos[1], z: p.pos[2] }, color: [p.color[0], p.color[1], p.color[2]] as [number, number, number], k1: p.k1, k2: p.k2 })),
    spots: json.lights.spots.map((s) => ({ pos: { x: s.pos[0], y: s.pos[1], z: s.pos[2] }, dir: [s.dir[0], s.dir[1], s.dir[2]] as [number, number, number], color: [s.color[0], s.color[1], s.color[2]] as [number, number, number], cosInner: s.cosInner, cosOuter: s.cosOuter, k1: s.k1, k2: s.k2 })),
  };

  return { root: buildNode(json.root), camera, lights };
}
