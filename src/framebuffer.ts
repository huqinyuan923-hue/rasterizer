/**
 * 渲染目标：RGBA Uint8ClampedArray 颜色缓冲 + Float32Array 深度缓冲。
 * 第 0 行为屏幕顶部。深度存"视空间距离" d = -z_view（正值，越小越近），初始 +Infinity。
 */

export class Framebuffer {
  readonly width: number;
  readonly height: number;
  readonly color: Uint8ClampedArray; // RGBA8，长度 width*height*4
  readonly depth: Float32Array;      // 长度 width*height

  constructor(width: number, height: number, color?: Uint8ClampedArray) {
    this.width = width;
    this.height = height;
    this.color = color ?? new Uint8ClampedArray(width * height * 4);
    this.depth = new Float32Array(width * height);
    // alpha 通道恒为 255（不透明），PPM 输出忽略 alpha
    for (let i = 3; i < this.color.length; i += 4) this.color[i] = 255;
    // 深度初始 +Infinity：空缓冲接受任意片段（clear() 亦会重置）
    this.depth.fill(Infinity);
  }

  /** 清屏：r/g/b ∈ [0,255]，深度重置为 +Infinity。 */
  clear(r: number, g: number, b: number): void {
    const { color, depth } = this;
    for (let i = 0; i < this.width * this.height; i++) {
      const o = i * 4;
      color[o] = r; color[o + 1] = g; color[o + 2] = b; color[o + 3] = 255;
      depth[i] = Infinity;
    }
  }

  /** 带边界 guard 的单像素写入（越界静默丢弃）。 */
  setPixel(x: number, y: number, r: number, g: number, b: number): void {
    if (x < 0 || y < 0 || x >= this.width || y >= this.height) return;
    const o = (y * this.width + x) * 4;
    this.color[o] = r;
    this.color[o + 1] = g;
    this.color[o + 2] = b;
  }
}
