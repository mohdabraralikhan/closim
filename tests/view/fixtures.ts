import { rebuildGarment } from "../../src/garment/project.js";
import { buildTshirtProject } from "../../src/garment/tshirt.js";
import type { AssembledGarment, FittingScene } from "../../src/garment/assembly.js";
import type { AvatarSpec } from "../../src/garment/avatar.js";
import type { GarmentProject } from "../../src/garment/project.js";

export interface TshirtFixture {
  project: GarmentProject;
  avatar: AvatarSpec;
  assembled: AssembledGarment;
  fitting: FittingScene;
}

export function buildTshirtFixture(): TshirtFixture {
  const { project, avatar } = buildTshirtProject();
  const { assembled, fitting } = rebuildGarment(project);
  return { project, avatar, assembled, fitting };
}
