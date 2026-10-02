// Reference case: pinned strip under gravity. Records tip deflection baseline.
import { buildGrid, preprocess } from "../src/mesh/mesh.js";
import { createScene, pinColumn } from "../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../src/physics/types.js";
import { CpuSolver } from "../src/backend/cpu-solver.js";
import { evalInternal } from "../src/physics/fem.js";

function main(): void {
  const { positions, uv, indices } = buildGrid(12, 6, 0.2, 0.1);
  const mesh = preprocess(positions, uv, indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  pinColumn(scene, (x) => x < 1e-9);
  const solver = new CpuSolver();
  solver.initialize(scene);
  const h = 1 / 60;
  console.log(`verts=${mesh.count} tris=${mesh.triCount} hinges=${mesh.hinges.length} pinned=${scene.pinned.size}`);
  for (let s = 0; s < 60; s++) {
    solver.step(h);
    const st = solver.lastStats!;
    if (s % 15 === 0 || s === 59) {
      console.log(
        `step ${s}: E=${st.energy.toExponential(4)} |g|=${st.gradNorm.toExponential(3)} ` +
        `pcg=${st.pcgIters} maxStrain=${st.maxStrain.toExponential(3)}`,
      );
    }
  }
  const pos = solver.getPositions();
  // Tip = material last column (i == nx), NOT current x: a hanging strip
  // swings down so tip x collapses toward 0.
  const nx = 12, ny = 6;
  let minY = Infinity, tipY = 0, tipN = 0;
  for (let j = 0; j <= ny; j++) {
    const id = j * (nx + 1) + nx;
    tipY += pos[id * 3 + 1]; tipN++;
  }
  for (let i = 0; i < mesh.count; i++) minY = Math.min(minY, pos[i * 3 + 1]);
  const ev = evalInternal(Float64Array.from(pos), mesh, scene.material);
  console.log(`tipMeanY=${(tipY / tipN).toFixed(5)} minY=${minY.toFixed(5)} maxStrain=${ev.maxStrain.toFixed(5)} finalE=${ev.energy.toExponential(5)}`);
  console.log("CHECKS: sag<−0.01:", tipY / tipN < -0.01, "strain<0.05:", ev.maxStrain < 0.05);
}

main();
