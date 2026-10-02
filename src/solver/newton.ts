// Variational implicit Euler + Newton + Jacobi-PCG + backtracking line search,
// with variational contact: barrier + lagged friction participate in the same
// Newton/Armijo solve (no post-hoc projection). CCD gates line-search trials.
import { pcg } from "../math/pcg.js";
import type { ClothScene } from "../physics/scene.js";
import { enforcePins, makePinFilter } from "../physics/scene.js";
import { evalInternal, internalEnergyOnly, membraneHvp } from "../physics/fem.js";
import type { ContactDiagnostics } from "../collision/types.js";

export interface StepStats {
  newtonIters: number;
  pcgIters: number;
  gradNorm: number;
  energy: number;
  maxStrain: number;
  converged: boolean;
  contact: ContactDiagnostics | null;
}

export interface StepperOptions {
  newtonIters?: number;
  pcgMaxIters?: number;
  gradTol?: number;
  lineSearchIters?: number;
}

export function implicitStep(scene: ClothScene, h: number, opts: StepperOptions = {}): StepStats {
  // Iteration budget: stiff barrier contact needs ~10 inexact-Newton steps to
  // walk out of deep penetration (0.1mm steps against 2.7mm/step gravity drop).
  const newtonIters = opts.newtonIters ?? 10;
  const pcgMaxIters = opts.pcgMaxIters ?? 60;
  const gradTol = opts.gradTol ?? 1e-5;
  const lsIters = opts.lineSearchIters ?? 10;

  const n3 = scene.mesh.count * 3;
  const M = scene.mesh.masses;
  const x0 = Float64Array.from(scene.positions);
  const v0 = scene.velocities;
  const [gx, gy, gz] = scene.gravity;

  // predictor y_hat = x + h v + h^2 g
  const yHat = new Float64Array(n3);
  for (let i = 0; i < scene.mesh.count; i++) {
    yHat[i * 3] = x0[i * 3] + h * v0[i * 3] + h * h * gx;
    yHat[i * 3 + 1] = x0[i * 3 + 1] + h * v0[i * 3 + 1] + h * h * gy;
    yHat[i * 3 + 2] = x0[i * 3 + 2] + h * v0[i * 3 + 2] + h * h * gz;
  }

  const x = Float64Array.from(yHat);
  enforcePins(scene, x);
  const filter = makePinFilter(scene.pinned);
  const contact = scene.contact ?? null;
  if (contact) {
    contact.beginStep(x0);
    // Predictor validation: the explicit predictor can tunnel OVER the thin
    // barrier zone in one step (endpoint-only active sets would miss it and
    // Newton would "converge" instantly on a penetrating state). Bisect back
    // along (x0 -> yHat) to the largest CCD-valid fraction as the initial
    // guess; the variational target yHat is unchanged.
    if (!contact.checkTrial(x).valid) {
      let lo = 0, hi = 1;
      const probe = new Float64Array(n3);
      for (let b = 0; b < 8; b++) {
        const mid = 0.5 * (lo + hi);
        for (let i = 0; i < n3; i++) probe[i] = x0[i] + mid * (x[i] - x0[i]);
        enforcePins(scene, probe);
        if (contact.checkTrial(probe).valid) lo = mid;
        else hi = mid;
        if (hi - lo < 1e-4) break;
      }
      for (let i = 0; i < n3; i++) x[i] = x0[i] + lo * (x[i] - x0[i]);
      enforcePins(scene, x);
    }
  }

  const totalEnergy = (xx: Float64Array): number => {
    let e = 0;
    for (let i = 0; i < scene.mesh.count; i++) {
      const m = M[i];
      for (let k = 0; k < 3; k++) {
        const d = xx[i * 3 + k] - yHat[i * 3 + k];
        e += 0.5 * m * d * d / (h * h);
      }
    }
    e += internalEnergyOnly(xx, scene.mesh, scene.material);
    if (contact) e += contact.meritEnergy(xx);
    return e;
  };

  let energy = 0;
  {
    // Seed the frozen contact set BEFORE the baseline energy: otherwise the
    // Armijo base excludes the barrier while every trial includes it, and the
    // line search can never accept once contact is active.
    if (contact) contact.updateActiveSet(x);
    energy = totalEnergy(x);
  }
  let gradNorm = Infinity;
  let maxStrain = 0;
  let pcgItersTotal = 0;
  let converged = false;

  const invH2 = 1 / (h * h);
  // Jacobi diag: M/h^2 + beta (material-scale shift so PCG stays well conditioned)
  const beta = Math.max(scene.material.stretchWarp, scene.material.stretchWeft, scene.material.shear) * scene.material.thickness * 0.1 + 1e-6;
  const diag = new Float64Array(n3);
  for (let i = 0; i < scene.mesh.count; i++) {
    for (let k = 0; k < 3; k++) diag[i * 3 + k] = M[i] * invH2 + beta;
  }
  const cHp = new Float64Array(n3); // contact HVP scratch

  for (let ni = 0; ni < newtonIters; ni++) {
    const ev = evalInternal(x, scene.mesh, scene.material);
    maxStrain = ev.maxStrain;
    // g = M(x-yHat)/h^2 + gradE_internal + gradE_barrier + g_friction_lagged
    const g = new Float64Array(n3);
    for (let i = 0; i < n3; i++) {
      const vi = Math.floor(i / 3);
      g[i] = M[vi] * (x[i] - yHat[i]) * invH2 + ev.grad[i];
    }
    if (contact) {
      contact.updateActiveSet(x);
      const ce = contact.energyGrad(x);
      for (let i = 0; i < n3; i++) g[i] += ce.grad[i];
      contact.addDiagEstimate(diag, x, M, invH2, beta);
    }
    filter(g);
    gradNorm = Math.sqrt(g.reduce((s, v) => s + v * v, 0));
    if (!isFinite(gradNorm)) throw new Error("non-finite gradient — blow up; reduce dt/stiffness");
    if (gradNorm < gradTol) { converged = true; break; }

    const b = new Float64Array(n3);
    for (let i = 0; i < n3; i++) b[i] = -g[i];

    const hvpFull = (p: Float64Array, out: Float64Array) => {
      // Inexact Newton: Hessian ≈ M/h^2 + H_membrane (bending kept in gradient only).
      const Hp = membraneHvp(x, p, scene.mesh, scene.material);
      for (let i = 0; i < n3; i++) {
        const vi = Math.floor(i / 3);
        out[i] = M[vi] * p[i] * invH2 + Hp[i];
      }
      if (contact) {
        contact.applyHvp(x, p, cHp);
        for (let i = 0; i < n3; i++) out[i] += cHp[i];
      }
    };

    const res = pcg(b, hvpFull, diag, { maxIters: pcgMaxIters, tol: 1e-3, filter });
    let dx = res.x;
    pcgItersTotal += res.iters;

    // Safeguard: PCG on indefinite StVK Hessian can return a non-descent
    // direction. Fall back to Jacobi-scaled steepest descent (guaranteed descent).
    let gtdx = 0;
    for (let i = 0; i < n3; i++) gtdx += g[i] * dx[i];
    if (!isFinite(gtdx) || gtdx >= 0) {
      for (let i = 0; i < n3; i++) dx[i] = -g[i] / diag[i];
      gtdx = 0;
      for (let i = 0; i < n3; i++) gtdx += g[i] * dx[i];
    }

    // Trust region: cap the Newton displacement per iteration. PCG on the
    // indefinite/contact-stiffened system can return huge (but descent)
    // steps whose line-search validation would cost more than the progress
    // is worth; scaling preserves the descent direction (Armijo intact).
    // 2mm >> legitimate contact steps (~0.1mm) and free-flight steps (~0).
    {
      let maxDx = 0;
      for (let i = 0; i < n3; i++) {
        const m = Math.abs(dx[i]);
        if (m > maxDx) maxDx = m;
      }
      const TRUST = 0.002;
      if (maxDx > TRUST) {
        const s = TRUST / maxDx;
        for (let i = 0; i < n3; i++) dx[i] *= s;
        gtdx *= s;
      }
    }

    // backtracking (Armijo). dx is guaranteed descent (gtdx < 0) here.
    // Every trial is ALSO gated by CCD against the Newton segment: a trial
    // that tunnels through contact is rejected even if Armijo would accept.
    let alpha = 1;
    let accepted = false;
    for (let li = 0; li < lsIters; li++) {
      const xt = new Float64Array(n3);
      for (let i = 0; i < n3; i++) xt[i] = x[i] + alpha * dx[i];
      enforcePins(scene, xt);
      if (contact) {
        const chk = contact.checkTrial(xt);
        if (!chk.valid) {
          contact.addRejection();
          alpha *= 0.5;
          continue;
        }
      }
      const eNew = totalEnergy(xt);
      if (!isFinite(eNew)) {
        if (contact) contact.addRejection();
        alpha *= 0.5;
        continue;
      }
      if (eNew <= energy + 1e-4 * alpha * gtdx) {
        x.set(xt);
        energy = eNew;
        accepted = true;
        if (contact) contact.commit(xt);
        break;
      }
      if (contact) contact.addRejection();
      alpha *= 0.5;
    }
    if (!accepted) {
      // Line search failed: keep current x (already energy-decreasing
      // up to here) and stop Newton — do NOT inject energy with a blind step.
      break;
    }
    if (alpha < 1e-6) break;
  }

  // velocity update + light damping
  const damp = 1 - Math.min(Math.max(scene.material.damping, 0), 0.1);
  for (let i = 0; i < n3; i++) {
    scene.velocities[i] = ((x[i] - x0[i]) / h) * damp;
  }
  if (contact) killApproachVelocity(contact, scene, x);
  scene.positions.set(x);
  enforcePins(scene, scene.positions);

  return { newtonIters, pcgIters: pcgItersTotal, gradNorm, energy, maxStrain, converged, contact: contact ? { ...contact.diag } : null };
}

/** Restitution-0 impact handling (velocity level, after the variational solve):
 *  remove approaching normal velocity at deep contacts so impacts do not
 *  bounce elastically off the barrier. Single deterministic sweep; separating
 *  velocities are untouched. */
function killApproachVelocity(
  contact: NonNullable<ClothScene["contact"]>,
  scene: ClothScene,
  x: Float64Array,
): void {
  contact.updateActiveSet(x);
  const v = scene.velocities;
  for (const c of contact.activeList()) {
    if (c.kind === 2) {
      const vy = v[c.p * 3 + 1];
      if (vy < 0) v[c.p * 3 + 1] = 0;
      continue;
    }
    if (c.kind === 0) {
      // relative normal velocity of p vs triangle
      const vp = [v[c.p * 3], v[c.p * 3 + 1], v[c.p * 3 + 2]];
      let tx = 0, ty = 0, tz = 0;
      const ws = [c.w0, c.w1, c.w2];
      const vs = [c.a, c.b, c.c];
      for (let k = 0; k < 3; k++) {
        if (vs[k] < 0) continue;
        tx += ws[k] * v[vs[k] * 3];
        ty += ws[k] * v[vs[k] * 3 + 1];
        tz += ws[k] * v[vs[k] * 3 + 2];
      }
      const rel = (vp[0] - tx) * c.nx + (vp[1] - ty) * c.ny + (vp[2] - tz) * c.nz;
      if (rel < 0) {
        const j = -0.5 * rel;
        v[c.p * 3] += j * c.nx;
        v[c.p * 3 + 1] += j * c.ny;
        v[c.p * 3 + 2] += j * c.nz;
        for (let k = 0; k < 3; k++) {
          if (vs[k] < 0) continue;
          v[vs[k] * 3] -= j * ws[k] * c.nx;
          v[vs[k] * 3 + 1] -= j * ws[k] * c.ny;
          v[vs[k] * 3 + 2] -= j * ws[k] * c.nz;
        }
      }
      continue;
    }
    // EE: edge-ab average vs edge-cd average
    const avg = (a: number, b: number): [number, number, number] => {
      const xa = a >= 0 ? [v[a * 3], v[a * 3 + 1], v[a * 3 + 2]] : [0, 0, 0];
      const xb = b >= 0 ? [v[b * 3], v[b * 3 + 1], v[b * 3 + 2]] : [0, 0, 0];
      return [(xa[0] + xb[0]) / 2, (xa[1] + xb[1]) / 2, (xa[2] + xb[2]) / 2];
    };
    const [ax, ay, az] = avg(c.a, c.b);
    const [cx, cy, cz] = avg(c.c, c.d);
    const rel = (ax - cx) * c.nx + (ay - cy) * c.ny + (az - cz) * c.nz;
    if (rel < 0) {
      const j = -0.25 * rel;
      for (const vv of [c.a, c.b]) {
        if (vv < 0) continue;
        v[vv * 3] += j * c.nx;
        v[vv * 3 + 1] += j * c.ny;
        v[vv * 3 + 2] += j * c.nz;
      }
      for (const vv of [c.c, c.d]) {
        if (vv < 0) continue;
        v[vv * 3] -= j * c.nx;
        v[vv * 3 + 1] -= j * c.ny;
        v[vv * 3 + 2] -= j * c.nz;
      }
    }
  }
}
