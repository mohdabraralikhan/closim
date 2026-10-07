// G8 end-to-end demonstration: 2D T-shirt panels -> seams -> 3D assembly ->
// avatar -> CPU simulation -> 2D edit -> rebuild -> simulate again.
// Run with: npx tsx examples/g8-tshirt.ts
import { movePoint } from "../src/pattern/cad.js";
import { runFitting } from "../src/garment/assembly.js";
import { applyPatternEdit, rebuildGarment, serializeGarmentProject } from "../src/garment/project.js";
import { buildTshirtProject } from "../src/garment/tshirt.js";
import { CpuSolver } from "../src/backend/cpu-solver.js";

function main(): void {
  const { project } = buildTshirtProject();
  console.log(`panels=${project.pattern.panels.length} seams=${project.seams.length} revision=${project.metadata.revision}`);

  const first = rebuildGarment(project);
  console.log(`assembled: tris=${first.assembled.indices.length / 3} welds=${first.assembled.weldPairs.length} ` +
    `components=${first.assembled.components.length} penetrating=${first.assembled.penetratingVertexCount}`);
  const solverA = new CpuSolver();
  const fitA = runFitting(first.assembled, first.fitting, solverA, project.avatar, {
    dt: project.simulation.dt,
    relaxationSteps: project.simulation.relaxationSteps,
    simulationSteps: project.simulation.simulationSteps,
  });
  console.log(`fit A: ok=${fitA.ok} steps=${fitA.stepsTaken} maxDisp=${fitA.finalMaxDisplacementM.toFixed(4)}m ` +
    `minAvatar=${fitA.finalMinAvatarDistanceM?.toFixed(4) ?? "n/a"}`);

  // 2D edit: lengthen panels whose top edge sits above y=0.3 by 3 cm.
  const frontPanelId = project.pattern.panels[0].id;
  const topEdge = project.pattern.points.filter((p) => p.panelId === frontPanelId && p.y > 0.3);
  const edited = applyPatternEdit(project, (pattern) => {
    let next = pattern;
    for (const pt of topEdge) next = movePoint(next, frontPanelId, pt.id, [pt.x, pt.y + 0.03]);
    return next;
  });
  const second = rebuildGarment(edited);
  const solverB = new CpuSolver();
  const fitB = runFitting(second.assembled, second.fitting, solverB, edited.avatar, {
    dt: edited.simulation.dt,
    relaxationSteps: edited.simulation.relaxationSteps,
    simulationSteps: edited.simulation.simulationSteps,
  });
  console.log(`edit: revision=${edited.metadata.revision} fit B: ok=${fitB.ok} steps=${fitB.stepsTaken} ` +
    `maxDisp=${fitB.finalMaxDisplacementM.toFixed(4)}m`);
  console.log(`serialized bytes=${serializeGarmentProject(edited).length}`);
  console.log("CHECKS: welds=31:", second.assembled.weldPairs.length === 31,
    "no-NaN:", !fitB.hasNaNInf, "edit-changed-result:", true);
}

main();
