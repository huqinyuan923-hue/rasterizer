/**
 * 黄金图写盘工具：所有 gen_mX 脚本统一经由它落盘到 golden/。
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { encodePPM, type RawImage } from "../src/ppm.js";

const HERE = dirname(fileURLToPath(import.meta.url));
// 编译后位于 build/tools/，仓库根为上两级
export const GOLDEN_DIR = join(HERE, "..", "..", "golden");

export function writeGolden(name: string, img: RawImage): void {
  mkdirSync(GOLDEN_DIR, { recursive: true });
  writeFileSync(join(GOLDEN_DIR, name), encodePPM(img));
}
