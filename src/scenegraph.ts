/**
 * 场景图（二期 M1）：节点树、transform 层级、可见性、遍历求世界矩阵。
 *
 * 冻结约定（黄金图对拍口径）：
 * - 节点局部变换为 TRS：local = ((T·Rz)·Ry)·Rx·S。欧拉角 rotationEuler = (x,y,z)
 *   表示"先绕 X，再绕 Y，再绕 Z"。矩阵乘法结合顺序固定为从左到右逐对相乘，
 *   与一期 demo 的 compose(T, Rz, Ry, Rx, S) 逐位一致（迁移等价的基石）。
 * - 世界矩阵 world = parent.world · local（mat4Mul(parent, local)，先应用 local）。
 *   根节点的父世界矩阵为恒等矩阵，故根的 world = I · local。
 * - 遍历：深度优先、先序、children 按数组顺序。
 * - 可见性：visible=false 的节点连同整个子树不进入绘制清单；世界矩阵仍会被计算。
 * - 绘制清单：按遍历序收集有 mesh 且可见的节点，model 直接取其 world 矩阵
 *   （同一 Mat4 引用，节点树与渲染共享，无拷贝）。
 */

import {
  mat4Identity, mat4Mul, mat4RotX, mat4RotY, mat4RotZ, mat4Scaling, mat4Translation,
  type Mat4, type Vec3,
} from "./math.js";
import type { DrawItem, Material, Mesh } from "./pipeline.js";

/** 节点局部变换（TRS 三元组）。 */
export interface NodeTransform {
  position: Vec3;
  rotationEuler: Vec3; // 弧度
  scale: Vec3;
}

/** 新建单位变换（每次新对象，节点可独立修改）。 */
export function makeTransform(position: Vec3 = { x: 0, y: 0, z: 0 }): NodeTransform {
  return { position, rotationEuler: { x: 0, y: 0, z: 0 }, scale: { x: 1, y: 1, z: 1 } };
}

/**
 * 冻结的 TRS 合成：local = ((T·Rz)·Ry)·Rx·S。
 * 结合顺序与一期 compose(T, Rz, Ry, Rx, S) 完全一致 —— 对同一组标量输入
 * （含恒等旋转 RotZ(0)/恒等缩放）产出逐位相同的矩阵。
 */
export function composeTRS(tr: NodeTransform): Mat4 {
  const t = mat4Translation(tr.position);
  const rz = mat4RotZ(tr.rotationEuler.z);
  const ry = mat4RotY(tr.rotationEuler.y);
  const rx = mat4RotX(tr.rotationEuler.x);
  const s = mat4Scaling(tr.scale);
  let m = mat4Mul(t, rz);
  m = mat4Mul(m, ry);
  m = mat4Mul(m, rx);
  m = mat4Mul(m, s);
  return m;
}

/** 场景节点。mesh+material 齐备的节点是可绘制节点。 */
export class SceneNode {
  readonly name: string;
  readonly transform: NodeTransform;
  visible: boolean;
  mesh: Mesh | null;
  material: Material | null;
  readonly children: SceneNode[];
  /** 世界矩阵（updateWorldMatrices 后有效；初始为单位矩阵）。 */
  readonly world: Mat4;

  constructor(
    name: string,
    opts: {
      transform?: NodeTransform;
      visible?: boolean;
      mesh?: Mesh | null;
      material?: Material | null;
    } = {},
  ) {
    this.name = name;
    this.transform = opts.transform ?? makeTransform();
    this.visible = opts.visible ?? true;
    this.mesh = opts.mesh ?? null;
    this.material = opts.material ?? null;
    this.children = [];
    this.world = mat4Identity();
  }

  /** 追加子节点（不检测环 —— 场景树由构造方保证无环）。 */
  add(child: SceneNode): this {
    this.children.push(child);
    return this;
  }
}

/**
 * 遍历并计算整棵树的世界矩阵（深度优先、先序）。
 * parentWorld 缺省为单位矩阵（根节点）。
 */
export function updateWorldMatrices(root: SceneNode, parentWorld: Mat4 = mat4Identity()): void {
  const local = composeTRS(root.transform);
  const world = mat4Mul(parentWorld, local);
  // 原地写入（world 是 readonly 引用，内容可更新 —— 避免每帧分配）
  root.world.set(world);
  for (const child of root.children) {
    updateWorldMatrices(child, root.world);
  }
}

/**
 * 深度优先先序收集绘制清单：visible 且 mesh 非空的节点。
 * visible=false 的子树整体跳过。顺序冻结（黄金对拍依赖）。
 */
export function collectDrawItems(root: SceneNode, out: DrawItem[] = []): DrawItem[] {
  if (!root.visible) return out;
  if (root.mesh !== null && root.material !== null) {
    out.push({ mesh: root.mesh, model: root.world, material: root.material });
  }
  for (const child of root.children) {
    collectDrawItems(child, out);
  }
  return out;
}

/** 深度优先先序枚举全部节点（含不可见与根自身；编辑器树面板用）。 */
export function walkNodes(root: SceneNode, visit: (node: SceneNode, depth: number) => void, depth = 0): void {
  visit(root, depth);
  for (const child of root.children) {
    walkNodes(child, visit, depth + 1);
  }
}
