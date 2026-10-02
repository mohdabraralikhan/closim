// Minimal vector helpers over Float64/Float32 arrays (plain functions, no allocs in hot paths where possible).

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a: ArrayLike<number>): number {
  return Math.sqrt(dot(a, a));
}

export function axpy(y: Float64Array, alpha: number, x: ArrayLike<number>): void {
  for (let i = 0; i < y.length; i++) y[i] += alpha * x[i];
}

export function copyInto(dst: Float64Array | Float32Array, src: ArrayLike<number>): void {
  for (let i = 0; i < dst.length; i++) dst[i] = src[i];
}
