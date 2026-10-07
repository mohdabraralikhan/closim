import { describe, expect, it } from "vitest";
import {
  movePanel,
  movePoint,
  rotatePanel,
  duplicatePanel,
} from "../../src/pattern/cad.js";
import {
  classifyPatternChange,
  interactionStateSurvives,
  panelVertexRanges,
  rebuildWithPlan,
  seamWeldMap,
  vertexPanelId,
} from "../../src/view/sync.js";
import { WorkspaceError } from "../../src/view/types.js";
import { buildTshirtFixture } from "./fixtures.js";

describe("rebuild classification", () => {
  const fixture = buildTshirtFixture();
  const frontId = fixture.project.pattern.panels[0].id;

  it("classifies an identical document as none", () => {
    const result = classifyPatternChange(fixture.project.pattern, fixture.project.pattern);
    expect(result.level).toBe("none");
    expect(result.panelIds).toEqual([]);
    expect(interactionStateSurvives(result)).toBe(true);
  });

  it("classifies a point move as a panel remesh", () => {
    const point = fixture.project.pattern.points.find((p) => p.panelId === frontId)!;
    const edited = movePoint(fixture.project.pattern, frontId, point.id, [point.x + 0.01, point.y]);
    const result = classifyPatternChange(fixture.project.pattern, edited);
    expect(result.level).toBe("panel-remesh");
    expect(result.panelIds).toEqual([frontId]);
    expect(interactionStateSurvives(result)).toBe(false);
  });

  it("classifies a panel transform change as a geometry refresh", () => {
    const panel = fixture.project.pattern.panels[0];
    const edited = movePanel(fixture.project.pattern, panel.id, [0.05, 0]);
    const result = classifyPatternChange(fixture.project.pattern, edited);
    expect(result.level).toBe("geometry-refresh");
    expect(result.panelIds).toEqual([panel.id]);
  });

  it("classifies panel duplication and deletion as topology rebuilds", () => {
    const panel = fixture.project.pattern.panels[0];
    const { document: duplicated } = duplicatePanel(fixture.project.pattern, panel.id, [2, 2]);
    const added = classifyPatternChange(fixture.project.pattern, duplicated);
    expect(added.level).toBe("topology-rebuild");
    expect(interactionStateSurvives(added)).toBe(false);

    const { document: withExtra } = duplicatePanel(fixture.project.pattern, panel.id, [2, 2]);
    const removed = classifyPatternChange(withExtra, fixture.project.pattern);
    expect(removed.level).toBe("topology-rebuild");
  });

  it("classifies rotation as a geometry refresh", () => {
    const panel = fixture.project.pattern.panels[1];
    const edited = rotatePanel(fixture.project.pattern, panel.id, 0.2, [0.2, 0.3]);
    const result = classifyPatternChange(fixture.project.pattern, edited);
    expect(result.level).toBe("geometry-refresh");
  });
});

describe("id mapping tables", () => {
  const fixture = buildTshirtFixture();

  it("maps every panel to a disjoint vertex range", () => {
    const ranges = panelVertexRanges(fixture.assembled);
    expect(ranges.size).toBe(fixture.project.pattern.panels.length);
    const covered = new Set<number>();
    for (const range of ranges.values()) {
      for (let v = range.vertexStart; v < range.vertexStart + range.vertexCount; v++) {
        expect(covered.has(v)).toBe(false);
        covered.add(v);
      }
    }
    expect(covered.size).toBe(fixture.assembled.positions.length / 3);
  });

  it("resolves vertices back to their panel", () => {
    const ranges = panelVertexRanges(fixture.assembled);
    for (const [panelId, range] of ranges) {
      expect(vertexPanelId(fixture.assembled, range.vertexStart)).toBe(panelId);
      expect(vertexPanelId(fixture.assembled, range.vertexStart + range.vertexCount - 1)).toBe(panelId);
    }
    expect(vertexPanelId(fixture.assembled, -1)).toBeNull();
    expect(vertexPanelId(fixture.assembled, 10 ** 6)).toBeNull();
  });

  it("groups weld pairs by seam id", () => {
    const seams = seamWeldMap(fixture.assembled);
    expect(seams.size).toBe(fixture.project.seams.length);
    let total = 0;
    for (const entry of seams.values()) total += entry.weldIndices.length;
    expect(total).toBe(fixture.assembled.weldPairs.length);
  });
});

describe("rebuild with plan", () => {
  it("runs the classify -> rebuild workflow and reports the plan", () => {
    const fixture = buildTshirtFixture();
    const frontId = fixture.project.pattern.panels[0].id;
    const point = fixture.project.pattern.points.find((p) => p.panelId === frontId)!;
    const edited = movePoint(fixture.project.pattern, frontId, point.id, [point.x, point.y + 0.02]);
    const plan = rebuildWithPlan(fixture.project, edited);
    expect(plan.classification.level).toBe("panel-remesh");
    expect(plan.project.pattern).toBe(edited);
    expect(plan.project.metadata.revision).toBe(fixture.project.metadata.revision + 1);
    expect(plan.result.assembled.positions.length).toBe(fixture.assembled.positions.length);
    expect(plan.result.fitting.scene.positions.length).toBe(fixture.assembled.positions.length);
  });

  it("propagates invalid pattern edits as errors (caller keeps old state)", () => {
    const fixture = buildTshirtFixture();
    const broken = JSON.parse(JSON.stringify(fixture.project.pattern));
    broken.panels[0].boundaryLoops[0].segmentIds.pop();
    expect(() => rebuildWithPlan(fixture.project, broken)).toThrowError();
  });
});
