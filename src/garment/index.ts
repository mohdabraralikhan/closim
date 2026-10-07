export * from "./types.js";
export {
  canonicalGarment,
  deserializeGarment,
  deterministicStitchId,
  GarmentValidationError,
  mapSeamParameter,
  serializeGarment,
  validateGarment,
} from "./garment.js";
export { buildGarmentSimulationMesh } from "./bridge.js";
export type { ArealDensityResolver } from "./bridge.js";
