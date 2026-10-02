// Jacobi-preconditioned Conjugate Gradient, matrix-free.
// H(x) is provided as hvp(p) -> Hp. M2 = diag preconditioner (e.g. lumped M/h^2 + beta).

export interface PcgResult {
  x: Float64Array;
  iters: number;
  residual: number;
  converged: boolean;
}

export function pcg(
  b: Float64Array,
  hvp: (p: Float64Array, out: Float64Array) => void,
  diag: Float64Array,
  opts: {
    maxIters?: number; tol?: number; x0?: Float64Array; filter?: (v: Float64Array) => void;
    /** G5B/C/D: custom preconditioner z = M^-1 r (replaces Jacobi when set). */
    applyPreconditioner?: (r: Float64Array, z: Float64Array) => void;
  } = {},
): PcgResult {
  const n = b.length;
  const maxIters = opts.maxIters ?? 100;
  const tol = opts.tol ?? 1e-6;
  const x = opts.x0 ? Float64Array.from(opts.x0) : new Float64Array(n);
  const filter = opts.filter;

  const r = new Float64Array(n);
  const z = new Float64Array(n);
  const p = new Float64Array(n);
  const Ap = new Float64Array(n);

  // r = b - A x0 ; x0 = 0 -> r = b
  if (opts.x0) {
    hvp(x, Ap);
    for (let i = 0; i < n; i++) r[i] = b[i] - Ap[i];
  } else {
    r.set(b);
  }
  if (filter) filter(r);

  if (opts.applyPreconditioner) {
    opts.applyPreconditioner(r, z);
  } else {
    for (let i = 0; i < n; i++) {
      z[i] = diag[i] > 1e-12 ? r[i] / diag[i] : r[i];
    }
  }
  if (filter) filter(z);
  p.set(z);

  let rz = 0;
  for (let i = 0; i < n; i++) rz += r[i] * z[i];

  const bNorm = Math.sqrt(dotSelf(b));
  const absTol = Math.max(tol * Math.max(bNorm, 1e-12), 1e-12);

  let residual = Math.sqrt(dotSelf(r));
  if (residual < absTol) return { x, iters: 0, residual, converged: true };

  for (let k = 0; k < maxIters; k++) {
    if (filter) filter(p);
    hvp(p, Ap);
    if (filter) filter(Ap);

    let pAp = 0;
    for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
    if (!(pAp > 1e-30)) {
      // Non-positive curvature: stop, return current x.
      return { x, iters: k, residual, converged: residual < absTol };
    }
    const alpha = rz / pAp;
    for (let i = 0; i < n; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }
    if (filter) { filter(x); filter(r); }

    residual = Math.sqrt(dotSelf(r));
    if (residual < absTol) return { x, iters: k + 1, residual, converged: true };

    if (opts.applyPreconditioner) {
      opts.applyPreconditioner(r, z);
    } else {
      for (let i = 0; i < n; i++) z[i] = diag[i] > 1e-12 ? r[i] / diag[i] : r[i];
    }
    if (filter) filter(z);

    let rzNew = 0;
    for (let i = 0; i < n; i++) rzNew += r[i] * z[i];
    const beta = rzNew / rz;
    for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    rz = rzNew;
  }
  return { x, iters: maxIters, residual, converged: false };
}

function dotSelf(a: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return s;
}
