// G6A c1-8 production-candidate stress: one device step each on floor and
// fold scenes via the explicit mode API. Finite energy/positions required;
// fold must engage contacts (span machinery exercised through the mode path).
import { describe, it, expect, beforeAll } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import type { ClothScene } from "../../src/physics/scene.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice } from "./device-setup.js";

function floorScene(): ReturnType<typeof createScene> {
  const g = buildGrid(8, 8, 0.16, 0.16);
  for (let i = 0; i < g.positions.length / 3; i++) g.positions[i * 3 + 1] += 0.05;
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  scene.contact = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  scene.contact.setFloor(0);
  return scene;
}

function foldScene(): ReturnType<typeof createScene> {
  const w = 0.16;
  const g = buildGrid(8, 8, w, w);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  for (let i = 0; i < mesh.count; i++) {
    const x = scene.positions[i * 3];
    if (x > w / 2) {
      scene.positions[i * 3] = w / 2 - 2 * (x - w / 2);
      scene.positions[i * 3 + 1] += 0.0015;
    }
  }
  scene.contact = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  return scene;
}

let floorFix: DeviceFixture | null = null;
let foldFix: DeviceFixture | null = null;

beforeAll(async () => {
  floorFix = await sharedDevice("g6a-floor", floorScene, { contactCapacity: 1024, pairCapacity: 4096 });
  foldFix = await sharedDevice("g6a-fold", foldScene, { contactCapacity: 2048, pairCapacity: 8192 });
}, 240000);

async function oneStep(
  fix: DeviceFixture | null, label: string, expectContacts: boolean,
): Promise<void> {
  if (!fix) {
    // eslint-disable-next-line no-console
    console.log(`[g6a-stress] no device for ${label} — skipped`);
    return;
  }
  const { solver, driver } = fix;
  solver.setPreconditionerMode("mas-c1-8");
  try {
    const diag = await solver.stepGpu(1 / 60, { newtonIters: 2 });
    expect(Number.isFinite(diag.energy)).toBe(true);
    expect(diag.finite).toBe(1);
    const rep = solver.lastStepReport!;
    // eslint-disable-next-line no-console
    console.log(`[g6a-stress] ${label}: rep=${JSON.stringify(rep)} fallbacks=${JSON.stringify(solver.fallbackLog)}`);
    expect(rep.requestedPreconditioner).toBe("mas-c1-8");
    if (rep.finePcgIters > 0) {
      // A PCG solve ran: the requested coarse path must have engaged
      // (modulo a recorded topology fallback, which is itself the gate).
      if (!rep.fallback) expect(rep.actualPreconditioner).toBe("mas-c1-8");
      expect(rep.coarsePcgIters).toBeGreaterThan(0);
    } else {
      // Immediate convergence: no solve ran; the marker says so explicitly.
      expect(rep.actualPreconditioner).toBe("mas-c1-8:no-solve");
    }
    const pos = solver.getPositions();
    for (let i = 0; i < pos.length; i++) expect(Number.isFinite(pos[i])).toBe(true);
    if (expectContacts) {
      solver.configureStep(1 / 60);
      const ev = await solver.evaluateNewtonState(false, 900, 1 / 60);
      // eslint-disable-next-line no-console
      console.log(`[g6a-stress] ${label}: contacts=${ev.contactCount} energy=${diag.energy.toExponential(3)}`);
      expect(ev.contactCount).toBeGreaterThan(0);
    }
    void driver;
  } finally {
    (solver as unknown as { modeExplicit: boolean }).modeExplicit = false;
    solver.preconditionerMode = "jacobi";
  }
}

describe("G6A c1-8 one-step stress", () => {
  it("floor scene steps finite via mas-c1-8", async () => {
    await oneStep(floorFix, "floor", false);
  }, 240000);

  it("fold scene steps finite with engaged contacts via mas-c1-8", async () => {
    await oneStep(foldFix, "fold", true);
  }, 240000);
});
