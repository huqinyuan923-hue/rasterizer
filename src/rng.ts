/**
 * mulberry32 种子 PRNG。所有"随机"场景必须由它驱动（禁止 Math.random），保证黄金图确定性。
 * 返回 [0, 1) 的 float64。
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
