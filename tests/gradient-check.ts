// Finite-difference gradient check for the membrane element.
import { triangleEnergyGradient } from "../src/physics/membrane.js";
import { DEFAULT_MATERIAL } from "../src/physics/types.js";

export function checkTriGradient(): { maxAbs: number; maxRel: number } {
  const x0 = [0.01, 0.002, 0.03];
  const x1 = [0.03, -0.001, 0.028];
  const x2 = [0.012, 0.004, 0.05];
  const inv = [50, -5, 3, 40];
  const area = 0.5 * 0.02 * 0.02;
  const mat = DEFAULT_MATERIAL;
  const xs = [x0, x1, x2];
  const { grad } = triangleEnergyGradient(x0, x1, x2, inv, area, mat);
  const eps = 1e-7;
  let maxAbs = 0, maxRel = 0;
  for (let v = 0; v < 3; v++) {
    for (let i = 0; i < 3; i++) {
      const pert = xs.map((p) => [...p]);
      pert[v][i] += eps;
      const ep = triangleEnergyGradient(pert[0], pert[1], pert[2], inv, area, mat).energy;
      pert[v][i] -= 2 * eps;
      const em = triangleEnergyGradient(pert[0], pert[1], pert[2], inv, area, mat).energy;
      const num = (ep - em) / (2 * eps);
      const ana = grad[v * 3 + i];
      const abs = Math.abs(num - ana);
      const rel = abs / (Math.abs(num) + 1e-12);
      maxAbs = Math.max(maxAbs, abs);
      maxRel = Math.max(maxRel, rel);
    }
  }
  return { maxAbs, maxRel };
}

if (import.meta.url === `file://${process.argv[1]?.replace(/\\/g, "/")}` || process.argv[1]?.endsWith("gradient-check.ts")) {
  const r = checkTriGradient();
  console.log(`membrane gradient check: maxAbs=${r.maxAbs.toExponential(3)} maxRel=${r.maxRel.toExponential(3)}`);
  if (r.maxRel > 1e-4) {
    console.error("GRADIENT CHECK FAILED");
    process.exit(1);
  } else {
    console.log("GRADIENT CHECK PASSED");
  }
}
