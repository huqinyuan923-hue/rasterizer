/**
 * PPM (P6) 编解码 —— 黄金图的落盘格式。手写实现，确定性、零依赖。
 * P6 头部：`P6\n<width> <height>\n<maxval>\n`，随后按行优先（顶行在前）存储 RGB 各 1 字节。
 * 我们的帧缓冲第 0 行即屏幕顶部，因此直接顺序写出。
 */

export interface RawImage {
  width: number;
  height: number;
  rgb: Uint8Array; // 长度 width*height*3
}

function asciiBytes(s: string): number[] {
  const out: number[] = [];
  for (let i = 0; i < s.length; i++) out.push(s.charCodeAt(i) & 0x7f);
  return out;
}

export function encodePPM(img: RawImage): Uint8Array {
  const header = asciiBytes(`P6\n${img.width} ${img.height}\n255\n`);
  const out = new Uint8Array(header.length + img.rgb.length);
  out.set(header, 0);
  out.set(img.rgb, header.length);
  return out;
}

/** 帧缓冲 → RGB（去 alpha，行序不变）。 */
export function framebufferToRGB(fb: { width: number; height: number; color: Uint8ClampedArray }): Uint8Array {
  const n = fb.width * fb.height;
  const rgb = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    rgb[i * 3] = fb.color[i * 4];
    rgb[i * 3 + 1] = fb.color[i * 4 + 1];
    rgb[i * 3 + 2] = fb.color[i * 4 + 2];
  }
  return rgb;
}

export function framebufferToRaw(fb: { width: number; height: number; color: Uint8ClampedArray }): RawImage {
  return { width: fb.width, height: fb.height, rgb: framebufferToRGB(fb) };
}

/**
 * 解析 P6。头部允许任意空白与 `#` 注释（PPM 规范），maxval 必须 ≤ 255。
 * 解析失败抛 Error（信息含原因）。
 */
export function decodePPM(bytes: Uint8Array): RawImage {
  let pos = 0;
  const tokens: string[] = [];
  // 头部一共需要 4 个 token：P6, width, height, maxval
  while (tokens.length < 4 && pos < bytes.length) {
    const c = bytes[pos];
    if (c === 0x23) { // '#' 注释，吞到行尾
      while (pos < bytes.length && bytes[pos] !== 0x0a) pos++;
    } else if (c === 0x20 || c === 0x09 || c === 0x0d || c === 0x0a) {
      pos++;
    } else {
      const start = pos;
      while (pos < bytes.length) {
        const d = bytes[pos];
        if (d === 0x20 || d === 0x09 || d === 0x0d || d === 0x0a || d === 0x23) break;
        pos++;
      }
      tokens.push(String.fromCharCode(...bytes.subarray(start, pos)));
    }
  }
  if (tokens.length < 4) throw new Error("PPM: header truncated");
  if (tokens[0] !== "P6") throw new Error(`PPM: expected magic P6, got ${tokens[0]}`);
  const width = Number(tokens[1]);
  const height = Number(tokens[2]);
  const maxval = Number(tokens[3]);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error(`PPM: bad dimensions ${tokens[1]}x${tokens[2]}`);
  }
  if (!Number.isInteger(maxval) || maxval <= 0 || maxval > 255) {
    throw new Error(`PPM: unsupported maxval ${tokens[3]}`);
  }
  // maxval 之后恰好一个空白字节，然后是二进制数据
  pos++; // 跳过第 4 个 token 后的分隔符（解析循环保证它存在）
  const need = width * height * 3;
  if (bytes.length - pos < need) {
    throw new Error(`PPM: pixel data truncated (need ${need}, have ${bytes.length - pos})`);
  }
  return { width, height, rgb: bytes.slice(pos, pos + need) };
}
