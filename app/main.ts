// G10 3D garment workspace app shell.
// The render loop never starts a solve: SimulationSession.tick() is the only
// stepping path, called explicitly each frame; camera movement only touches
// the pure camera state.

import { CpuSolver } from "../src/backend/cpu-solver.js";
import { movePoint, type PatternDocument } from "../src/pattern/cad.js";
import { rebuildGarment, type GarmentProject } from "../src/garment/project.js";
import type { Seam } from "../src/garment/sewing.js";
import { buildTshirtProject } from "../src/garment/tshirt.js";
import {
  DragController,
  PinManager,
  defaultPlacements,
  movePanelPlacement,
  repositionAroundAvatar,
  rotatePanelPlacement,
  transformGarmentPlacements,
  withPlacements,
} from "../src/view/manipulation.js";
import {
  pickAvatar,
  pickPanel,
  pickSeam,
  pickVertex,
  screenRay,
} from "../src/view/pick.js";
import { SimulationSession } from "../src/view/sim-session.js";
import { rebuildWithPlan } from "../src/view/sync.js";
import { boundsCenter, WorkspaceError } from "../src/view/types.js";
import { GarmentWorkspace } from "../src/view/viewport.js";
import { ViewportRenderer } from "./renderer.js";
import { mountProductionDashboard } from "./production-dashboard.js";

const canvas = document.getElementById("viewport") as HTMLCanvasElement;
const workspace = new GarmentWorkspace();
const solver = new CpuSolver();
const session = new SimulationSession(solver, (positions) => workspace.publishSimPositions(positions));
const pins = new PinManager();
const drag = new DragController(pins, workspace);
pins.attach(workspace, solver);
const renderer = new ViewportRenderer(canvas);

let currentProject: GarmentProject | null = null;
let currentSeams: Seam[] = [];
let statusMessage = "ready";
let lastPickLabel = "—";
const refreshProductionDashboard = mountProductionDashboard(
  document.getElementById("production-dashboard")!,
  document.getElementById("production-toggle") as HTMLButtonElement,
  () => currentProject,
);

function applyProject(project: GarmentProject, reason: "initial-load" | "rebuild" | "project-load"): void {
  const { assembled, fitting } = rebuildGarment(project);
  workspace.setGarment(project, assembled, reason);
  session.attach(fitting, project.simulation);
  currentProject = project;
  currentSeams = project.seams;
  renderer.rebuild(workspace, project.seams, project.pattern, project.placements);
  renderer.refreshSelection(workspace);
  renderer.updatePins(pins);
  workspace.frameGarment(canvasAspect());
  refreshOutliner();
  refreshInfo();
  refreshProductionDashboard();
}

function rebuildFromCurrentProject(): void {
  if (!currentProject) return;
  applyProject(currentProject, "rebuild");
}

function canvasAspect(): number {
  return canvas.clientWidth / Math.max(canvas.clientHeight, 1);
}

function guarded(action: () => void): void {
  try {
    action();
  } catch (error) {
    if (error instanceof WorkspaceError) {
      statusMessage = `${error.code}: ${error.message}`;
    } else if (error instanceof Error) {
      statusMessage = error.message;
    } else {
      statusMessage = String(error);
    }
    refreshStatusBar();
  }
}

// ---------------------------------------------------------------------------
// Camera input (never triggers a solve)
// ---------------------------------------------------------------------------

let pointerState: { id: number; mode: "orbit" | "pan" | "drag" | "click"; x: number; y: number } | null = null;

function ndcFromEvent(event: PointerEvent): [number, number] {
  const rect = canvas.getBoundingClientRect();
  return [
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -(((event.clientY - rect.top) / rect.height) * 2 - 1),
  ];
}

function pickRay(event: PointerEvent) {
  const [nx, ny] = ndcFromEvent(event);
  return screenRay(workspace.camera.viewProjectionMatrix(canvasAspect()), nx, ny);
}

canvas.addEventListener("pointerdown", (event) => {
  canvas.setPointerCapture(event.pointerId);
  if (event.button === 1 || (event.button === 0 && event.altKey)) {
    pointerState = { id: event.pointerId, mode: event.shiftKey ? "pan" : "orbit", x: event.clientX, y: event.clientY };
  } else if (event.button === 2) {
    pointerState = { id: event.pointerId, mode: "orbit", x: event.clientX, y: event.clientY };
  } else if (event.button === 0) {
    guarded(() => {
      const ray = pickRay(event);
      const vertexHit = workspace.assembled
        ? pickVertex(workspace.assembled, workspace.renderPositions, ray, vertexPickRadius())
        : null;
      if (vertexHit) {
        const panelHit = workspace.assembled ? pickPanel(workspace.assembled, workspace.renderPositions, ray) : null;
        const grabPoint = panelHit ? panelHit.point : ([
          workspace.renderPositions[vertexHit.vertex * 3],
          workspace.renderPositions[vertexHit.vertex * 3 + 1],
          workspace.renderPositions[vertexHit.vertex * 3 + 2],
        ] as [number, number, number]);
        drag.begin(vertexHit.vertex, grabPoint);
        pointerState = { id: event.pointerId, mode: "drag", x: event.clientX, y: event.clientY };
        lastPickLabel = `dragging vertex ${vertexHit.vertex} (${workspace.vertexPanelId(vertexHit.vertex) ?? "?"})`;
        return;
      }
      pointerState = { id: event.pointerId, mode: "click", x: event.clientX, y: event.clientY };
    });
  }
  refreshStatusBar();
});

canvas.addEventListener("pointermove", (event) => {
  if (!pointerState || pointerState.id !== event.pointerId) return;
  const dx = event.clientX - pointerState.x;
  const dy = event.clientY - pointerState.y;
  pointerState.x = event.clientX;
  pointerState.y = event.clientY;
  if (pointerState.mode === "orbit") {
    workspace.camera.orbit(dx * 0.008, dy * 0.008);
  } else if (pointerState.mode === "pan") {
    const wpp = workspace.camera.worldPerPixel(canvas.clientHeight);
    workspace.camera.pan(-dx * wpp, dy * wpp);
  } else if (pointerState.mode === "drag") {
    guarded(() => {
      const ray = pickRay(event);
      const planeHit = rayOriginPlaneIntersection(ray, workspace.camera.basis().eye);
      if (planeHit) drag.moveTo(planeHit);
    });
  }
});

canvas.addEventListener("pointerup", (event) => {
  if (!pointerState || pointerState.id !== event.pointerId) return;
  const mode = pointerState.mode;
  pointerState = null;
  canvas.releasePointerCapture(event.pointerId);
  if (mode === "drag") {
    guarded(() => drag.end(false));
    refreshStatusBar();
    return;
  }
  if (mode === "click") {
    const moved = Math.abs(event.movementX) + Math.abs(event.movementY);
    guarded(() => handleClickSelect(event, moved < 4));
  }
});

canvas.addEventListener("contextmenu", (event) => event.preventDefault());

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const factor = Math.exp(-event.deltaY * 0.0012);
  workspace.camera.zoom(factor);
}, { passive: false });

function vertexPickRadius(): number {
  return Math.max(0.006, workspace.camera.worldPerPixel(canvas.clientHeight) * 9);
}

/** Intersect the ray with a camera-facing plane through the orbit target: stable drag depth. */
function rayOriginPlaneIntersection(ray: { origin: [number, number, number]; dir: [number, number, number] }, eye: [number, number, number]): [number, number, number] | null {
  const target = workspace.camera.state.target;
  const n: [number, number, number] = [
    eye[0] - target[0],
    eye[1] - target[1],
    eye[2] - target[2],
  ];
  const len = Math.hypot(n[0], n[1], n[2]) || 1;
  n[0] /= len; n[1] /= len; n[2] /= len;
  const denom = ray.dir[0] * n[0] + ray.dir[1] * n[1] + ray.dir[2] * n[2];
  if (Math.abs(denom) < 1e-9) return null;
  const t = ((target[0] - ray.origin[0]) * n[0] + (target[1] - ray.origin[1]) * n[1] + (target[2] - ray.origin[2]) * n[2]) / denom;
  if (t <= 0) return null;
  return [ray.origin[0] + t * ray.dir[0], ray.origin[1] + t * ray.dir[1], ray.origin[2] + t * ray.dir[2]];
}

function handleClickSelect(event: PointerEvent, isClick: boolean): void {
  if (!workspace.assembled) return;
  const ray = pickRay(event);
  if (!isClick) return;
  const seamHit = pickSeam(workspace.assembled, workspace.renderPositions, ray);
  const panelHit = pickPanel(workspace.assembled, workspace.renderPositions, ray);
  // Seam wins only when the ray clearly targets a stitch/edge rather than a panel face.
  if (seamHit && (!panelHit || seamHit.distance <= panelHit.distance + 0.003)) {
    toggleSeamSelection(seamHit.seamId);
    lastPickLabel = `seam ${seamHit.seamId}`;
  } else if (panelHit) {
    togglePanelSelection(panelHit.panelId);
    lastPickLabel = `panel ${panelHit.panelId}`;
  } else if (workspace.project?.avatar) {
    const avatarHit = pickAvatar(workspace.project.avatar, ray);
    if (avatarHit) {
      workspace.selection.avatar = !workspace.selection.avatar;
      lastPickLabel = `avatar ${workspace.project.avatar.id}`;
    } else {
      workspace.selection.garment = false;
      workspace.selection.avatar = false;
      workspace.selection.panelIds = [];
      workspace.selection.seamIds = [];
      lastPickLabel = "cleared selection";
    }
  }
  renderer.refreshSelection(workspace);
  refreshOutliner();
  refreshStatusBar();
}

function togglePanelSelection(panelId: string): void {
  if (workspace.selection.panelIds.includes(panelId)) {
    workspace.selection.panelIds = workspace.selection.panelIds.filter((id) => id !== panelId);
  } else {
    workspace.selection.panelIds.push(panelId);
  }
}

function toggleSeamSelection(seamId: string): void {
  if (workspace.selection.seamIds.includes(seamId)) {
    workspace.selection.seamIds = workspace.selection.seamIds.filter((id) => id !== seamId);
  } else {
    workspace.selection.seamIds.push(seamId);
  }
}

// ---------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  switch (event.key) {
    case "f": case "F":
      if (!workspace.frameSelection(canvasAspect())) workspace.frameGarment(canvasAspect());
      break;
    case "g": case "G": workspace.frameGarment(canvasAspect()); break;
    case "a": case "A": if (currentProject?.avatar) workspace.frameAvatar(currentProject.avatar, canvasAspect()); break;
    case "r": case "R": workspace.camera.reset(); break;
    case "o": case "O": workspace.camera.setMode(workspace.camera.state.mode === "perspective" ? "orthographic" : "perspective"); break;
    case " ": event.preventDefault(); guarded(() => session.toggle()); break;
    case "Escape":
      guarded(() => drag.cancel());
      workspace.selection.garment = false;
      workspace.selection.avatar = false;
      workspace.selection.panelIds = [];
      workspace.selection.seamIds = [];
      renderer.refreshSelection(workspace);
      refreshOutliner();
      break;
    case "1": workspace.camera.setView("front"); break;
    case "2": workspace.camera.setView("back"); break;
    case "3": workspace.camera.setView("left"); break;
    case "4": workspace.camera.setView("right"); break;
    case "5": workspace.camera.setView("top"); break;
    case "6": workspace.camera.setView("bottom"); break;
    default: return;
  }
  refreshViewTools();
  refreshStatusBar();
});

// ---------------------------------------------------------------------------
// Manipulation + rebuild actions
// ---------------------------------------------------------------------------

function requireProject(): GarmentProject {
  if (!currentProject) throw new WorkspaceError("no-garment", "no project loaded");
  return currentProject;
}

function selectedPanelId(): string {
  const id = workspace.selection.panelIds[0];
  if (!id) throw new WorkspaceError("invalid-entity", "select a panel first");
  return id;
}

const PANEL_MOVE_STEP = 0.02;
const PANEL_YAW_STEP = Math.PI / 12;

const actions = {
  playPause: () => { session.toggle(); },
  step1: () => { session.stepOnce(1); },
  step10: () => { session.stepOnce(10); },
  restoreStable: () => {
    if (session.restoreStable()) statusMessage = "restored last stable state";
    else statusMessage = "no stable state available";
  },
  resetSim: () => {
    if (currentProject) {
      const { fitting } = rebuildGarment(currentProject);
      session.attach(fitting, currentProject.simulation);
      statusMessage = "simulation reset";
    }
  },
  recenterOnAvatar: () => {
    const project = requireProject();
    applyProject(repositionAroundAvatar(project, workspace.currentBounds(), project.avatar ?? {
      id: "none", bodyPart: "none", positions: [0, 0, 0], indices: [0, 0, 0], thicknessM: 0,
    }), "rebuild");
    statusMessage = "garment recentered on avatar";
  },
  resetPlacement: () => {
    const project = requireProject();
    applyProject(withPlacements(project, defaultPlacements(project)), "rebuild");
    statusMessage = "placement reset";
  },
  garmentTranslate: (axis: 0 | 1 | 2, sign: number) => {
    const delta: [number, number, number] = [0, 0, 0];
    delta[axis] = sign * PANEL_MOVE_STEP * 2;
    applyProject(transformGarmentPlacements(requireProject(), delta, 0, [0, 0, 0]), "rebuild");
  },
  garmentYaw: (sign: number) => {
    const project = requireProject();
    applyProject(transformGarmentPlacements(project, [0, 0, 0], sign * PANEL_YAW_STEP, boundsCenter(workspace.currentBounds())), "rebuild");
  },
  panelTranslate: (axis: 0 | 1 | 2, sign: number) => {
    const project = requireProject();
    const panelId = selectedPanelId();
    const delta: [number, number, number] = [0, 0, 0];
    delta[axis] = sign * PANEL_MOVE_STEP;
    applyProject(movePanelPlacement(project, panelId, delta), "rebuild");
    statusMessage = `moved panel ${panelId}`;
  },
  panelYaw: (sign: number) => {
    const project = requireProject();
    const panelId = selectedPanelId();
    applyProject(rotatePanelPlacement(project, panelId, sign * PANEL_YAW_STEP), "rebuild");
    statusMessage = `rotated panel ${panelId}`;
  },
  lengthenTopEdge: () => {
    const project = requireProject();
    const frontId = project.pattern.panels[0].id;
    const topPoints = project.pattern.points.filter((p) => p.panelId === frontId && p.y > 0.3);
    if (topPoints.length === 0) {
      statusMessage = "demo edit: no top-edge points found";
      return;
    }
    let edited: PatternDocument = project.pattern;
    for (const pt of topPoints) {
      edited = movePoint(edited, frontId, pt.id, [pt.x, pt.y + 0.03]);
    }
    const plan = rebuildWithPlan(project, edited);
    applyProject(plan.project, "rebuild");
    statusMessage = `2D edit applied (${plan.classification.level}) — rebuilt`;
  },
  unpinAll: () => {
    pins.clear();
    statusMessage = "all pins removed";
  },
  dropSelectedPin: (pinId: string) => {
    pins.remove(pinId);
    refreshOutliner();
  },
  frameSelection: () => {
    if (!workspace.frameSelection(canvasAspect())) workspace.frameGarment(canvasAspect());
  },
};

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

const elInfo = document.getElementById("info")!;
const elTools = document.getElementById("view-tools")!;
const elOutliner = document.getElementById("outliner")!;
const elSim = document.getElementById("sim-panel")!;
const elStatus = document.getElementById("status-bar")!;
const elHelp = document.getElementById("help")!;

function button(label: string, onClick: () => void, cls = ""): HTMLButtonElement {
  const b = document.createElement("button");
  b.textContent = label;
  b.className = cls;
  b.addEventListener("click", () => guarded(onClick));
  return b;
}

function checkbox(label: string, checked: boolean, onChange: (value: boolean) => void): HTMLLabelElement {
  const l = document.createElement("label");
  l.className = "chk";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = checked;
  input.addEventListener("change", () => onChange(input.checked));
  l.appendChild(input);
  l.appendChild(document.createTextNode(label));
  return l;
}

function row(...children: HTMLElement[]): HTMLDivElement {
  const div = document.createElement("div");
  div.className = "row";
  for (const c of children) div.appendChild(c);
  return div;
}

function kv(key: string, value: string): HTMLDivElement {
  const div = document.createElement("div");
  div.className = "kv";
  const k = document.createElement("span");
  k.className = "k";
  k.textContent = key;
  const v = document.createElement("span");
  v.textContent = value;
  div.append(k, v);
  return div;
}

function refreshInfo(): void {
  elInfo.replaceChildren();
  const project = workspace.project;
  const title = document.createElement("h3");
  title.textContent = "CLOTHSIM — 3D Workspace";
  elInfo.appendChild(title);
  if (!project) {
    elInfo.appendChild(document.createTextNode("no garment loaded"));
    return;
  }
  elInfo.append(
    kv("project", `${project.metadata.name} (r${project.metadata.revision})`),
    kv("panels", String(project.pattern.panels.length)),
    kv("seams", String(project.seams.length)),
    kv("vertices / tris", `${workspace.vertexCount} / ${workspace.assembled ? workspace.assembled.indices.length / 3 : 0}`),
    kv("garment epoch", String(workspace.garmentEpoch)),
    kv("sim steps", String(session.stepCount)),
    kv("pins", String(pins.list().length)),
    kv("avatar", project.avatar ? project.avatar.id : "none"),
  );
}

function refreshViewTools(): void {
  elTools.replaceChildren();
  const h = document.createElement("h3");
  h.textContent = "View";
  elTools.appendChild(h);
  elTools.appendChild(row(
    button("Frame sel (F)", actions.frameSelection),
    button("Garment (G)", () => workspace.frameGarment(canvasAspect())),
  ));
  elTools.appendChild(row(
    button("Avatar (A)", () => { if (currentProject?.avatar) workspace.frameAvatar(currentProject.avatar, canvasAspect()); }),
    button("Reset (R)", () => workspace.camera.reset()),
  ));
  const modeBtn = button(workspace.camera.state.mode === "perspective" ? "Persp (O)" : "Ortho (O)", () =>
    workspace.camera.setMode(workspace.camera.state.mode === "perspective" ? "orthographic" : "perspective"));
  elTools.appendChild(row(modeBtn));
  elTools.appendChild(row(
    button("Front 1", () => workspace.camera.setView("front")),
    button("Back 2", () => workspace.camera.setView("back")),
    button("Left 3", () => workspace.camera.setView("left")),
    button("Right 4", () => workspace.camera.setView("right")),
    button("Top 5", () => workspace.camera.setView("top")),
    button("Bot 6", () => workspace.camera.setView("bottom")),
  ));
  const vh = document.createElement("h3");
  vh.textContent = "Visibility";
  elTools.appendChild(vh);
  const v = workspace.visibility;
  elTools.appendChild(row(
    checkbox("garment", v.garment, (x) => { v.garment = x; }),
    checkbox("avatar", v.avatar, (x) => { v.avatar = x; }),
  ));
  elTools.appendChild(row(
    checkbox("seams", v.seams, (x) => { v.seams = x; }),
    checkbox("boundaries", v.panelBoundaries, (x) => { v.panelBoundaries = x; }),
  ));
  elTools.appendChild(row(
    checkbox("wireframe", v.wireframe, (x) => { v.wireframe = x; }),
    checkbox("normals", v.normals, (x) => { v.normals = x; renderer.refreshSelection(workspace); }),
  ));
  elTools.appendChild(row(
    checkbox("pins", v.pins, (x) => { v.pins = x; }),
    checkbox("penetration", v.penetrationHeat, (x) => { v.penetrationHeat = x; renderer.refreshSelection(workspace); }),
  ));
  const gh = document.createElement("h3");
  gh.textContent = "Garment placement";
  elTools.appendChild(gh);
  elTools.appendChild(row(
    button("◀ X", () => actions.garmentTranslate(0, -1)),
    button("X ▶", () => actions.garmentTranslate(0, 1)),
    button("▲ Y", () => actions.garmentTranslate(1, 1)),
    button("Y ▼", () => actions.garmentTranslate(1, -1)),
    button("⟲", () => actions.garmentYaw(1)),
    button("⟳", () => actions.garmentYaw(-1)),
  ));
  elTools.appendChild(row(
    button("Recenter on avatar", actions.recenterOnAvatar),
    button("Reset placement", actions.resetPlacement, "warn"),
  ));
  const ph = document.createElement("h3");
  ph.textContent = "Selected panel";
  elTools.appendChild(ph);
  elTools.appendChild(row(
    button("◀ X", () => actions.panelTranslate(0, -1)),
    button("X ▶", () => actions.panelTranslate(0, 1)),
    button("▲ Y", () => actions.panelTranslate(1, 1)),
    button("Y ▼", () => actions.panelTranslate(1, -1)),
    button("Z+", () => actions.panelTranslate(2, 1)),
    button("Z-", () => actions.panelTranslate(2, -1)),
  ));
  elTools.appendChild(row(
    button("Rotate +15°", () => actions.panelYaw(1)),
    button("Rotate −15°", () => actions.panelYaw(-1)),
  ));
}

let outlinerSelectionRef = "";

function refreshOutliner(): void {
  const key = JSON.stringify({
    panels: workspace.selection.panelIds,
    seams: workspace.selection.seamIds,
    pins: pins.list().map((p) => p.id),
    epoch: workspace.garmentEpoch,
  });
  if (key === outlinerSelectionRef) return;
  outlinerSelectionRef = key;
  elOutliner.replaceChildren();
  const h = document.createElement("h3");
  h.textContent = "Outliner";
  elOutliner.appendChild(h);
  const panels = currentProject?.pattern.panels ?? [];
  const ph = document.createElement("div");
  ph.className = "k";
  ph.textContent = "Panels";
  ph.style.marginTop = "4px";
  elOutliner.appendChild(ph);
  for (const panel of panels) {
    const item = document.createElement("div");
    item.className = "list-item" + (workspace.selection.panelIds.includes(panel.id) ? " selected" : "");
    item.textContent = panel.name;
    const dim = document.createElement("span");
    dim.className = "dim";
    dim.textContent = panel.id.split("/").pop() ?? panel.id;
    item.appendChild(dim);
    item.addEventListener("click", () => {
      togglePanelSelection(panel.id);
      workspace.frameSelection(canvasAspect());
      renderer.refreshSelection(workspace);
      refreshOutliner();
    });
    elOutliner.appendChild(item);
  }
  const sh = document.createElement("div");
  sh.className = "k";
  sh.textContent = "Seams";
  sh.style.marginTop = "4px";
  elOutliner.appendChild(sh);
  for (const seam of currentSeams) {
    const item = document.createElement("div");
    item.className = "list-item" + (workspace.selection.seamIds.includes(seam.id) ? " selected" : "");
    item.textContent = seam.id;
    item.addEventListener("click", () => {
      if (workspace.selection.seamIds.includes(seam.id)) {
        workspace.selection.seamIds = workspace.selection.seamIds.filter((id) => id !== seam.id);
      } else {
        workspace.selection.seamIds.push(seam.id);
      }
      renderer.refreshSelection(workspace);
      refreshOutliner();
    });
    elOutliner.appendChild(item);
  }
  const pinh = document.createElement("div");
  pinh.className = "k";
  pinh.textContent = "Pins";
  pinh.style.marginTop = "4px";
  elOutliner.appendChild(pinh);
  for (const pin of pins.list()) {
    const item = document.createElement("div");
    item.className = "list-item";
    item.textContent = `${pin.id} @v${pin.vertexId}`;
    const remove = button("✕", () => actions.dropSelectedPin(pin.id), "warn");
    remove.style.padding = "0 5px";
    item.appendChild(remove);
    elOutliner.appendChild(item);
  }
  if (pins.list().length > 0) {
    elOutliner.appendChild(row(button("Remove all pins", actions.unpinAll, "warn")));
  }
}

function refreshSimPanel(): void {
  elSim.replaceChildren();
  const h = document.createElement("h3");
  h.textContent = `Simulation — ${session.status}`;
  elSim.appendChild(h);
  elSim.appendChild(row(
    button(session.isPlaying ? "Pause (Space)" : "Simulate (Space)", actions.playPause, "primary"),
    button("Step ×1", actions.step1),
    button("Step ×10", actions.step10),
  ));
  elSim.appendChild(row(
    button("Restore stable", actions.restoreStable, "warn"),
    button("Reset sim", actions.resetSim),
    button("Demo: 2D edit +3cm", actions.lengthenTopEdge),
  ));
}

function refreshStatusBar(): void {
  const stats = session.lastStats;
  const parts = [
    statusMessage,
    `pick: ${lastPickLabel}`,
    drag.isActive ? `DRAG ${drag.state?.pin.id}` : "",
    session.status !== "detached" ? `sim: ${session.status} step=${session.stepCount}` : "",
    stats ? `E=${stats.energy.toFixed(4)} strain=${stats.maxStrain.toFixed(3)} iters=${stats.newtonIters}/${stats.pcgIters} contact=${stats.contact ? stats.contact.activePairs : 0}` : "",
    `epoch=${workspace.garmentEpoch} simEpoch=${workspace.simEpoch}`,
  ].filter((s) => s.length > 0);
  elStatus.replaceChildren(document.createTextNode(parts.join("   |   ")));
}

function refreshHelp(): void {
  elHelp.replaceChildren();
  const h = document.createElement("h3");
  h.textContent = "Input";
  elHelp.appendChild(h);
  const lines = [
    "LMB click: select panel / avatar",
    "LMB drag on vertex: grab & drag (temporary pin)",
    "Alt+LMB drag: orbit · Shift+Alt+LMB: pan · MMB: orbit · RMB: orbit",
    "Wheel: zoom",
    "Space: simulate/pause · F/G/A frame · R reset · O persp/ortho",
    "1–6: front/back/left/right/top/bottom · Esc: cancel/clear",
  ];
  for (const line of lines) {
    const div = document.createElement("div");
    div.textContent = line;
    elHelp.appendChild(div);
  }
}

// ---------------------------------------------------------------------------
// Bootstrap + frame loop
// ---------------------------------------------------------------------------

function resize(): void {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.resize(width, height);
}

window.addEventListener("resize", resize);
resize();

applyProject(buildTshirtProject().project, "initial-load");
refreshViewTools();
refreshSimPanel();
refreshHelp();
refreshStatusBar();

let lastFrameTime = performance.now();
let hudAccumulator = 0;

function frame(now: number): void {
  const dt = Math.min((now - lastFrameTime) / 1000, 0.1);
  lastFrameTime = now;

  session.tick(dt);
  renderer.sync(workspace);
  renderer.updatePins(pins);
  renderer.render(workspace);

  hudAccumulator += dt;
  if (hudAccumulator > 0.25) {
    hudAccumulator = 0;
    refreshInfo();
    refreshSimPanel();
    refreshOutliner();
    refreshStatusBar();
  }

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
