// G7A material abstraction + calibration.
//
// Backend-neutral wrapper around the verified orthotropic StVK implementation
// (membrane.ts / membrane-hvp.ts / bending.ts, assembled exactly as fem.ts).
// The default model delegates to those exact functions, so its numerics are
// identical to the legacy path by construction; the parity tests in
// tests/material/ pin that down bitwise.
//
// Deliberately OUT of scope (G7A owns only the model layer + tests):
// plasticity/hysteresis MODELS (metadata representation only — constructing a
// model with a non-"none" dissipative model throws), ContactSystem/CCD,
// Newton/PCG/preconditioners, topology, rendering. Consumers (scene, solvers,
// GPU upload) keep reading ClothMaterial; fromLegacy/toLegacy is the bridge.

import type { ClothMeshData } from "../mesh/mesh.js";
import type { ClothMaterial } from "./types.js";
import { evalInternal, internalEnergyOnly } from "./fem.js";
import { evalMembraneHvp } from "./membrane-hvp.js";

// ---------------------------------------------------------------------------
// Physical parameters (SI, validated). Calibrated values live here; artistic
// controls do not exist on this type. Coulomb friction mu is owned by the
// contact system (ContactParams), NOT the material model.
// ---------------------------------------------------------------------------

export interface InertiaParams {
  /** Lumped-mass areal density. m_i = arealDensityKgM2 * (sum A_t / 3). */
  arealDensityKgM2: number;
}

export interface MembraneStiffnessPa {
  /** Orthotropic StVK moduli C00/C11/C01/G in Pascal (= N/m^2 = J/m^3). */
  warpPa: number;
  weftPa: number;
  couplingPa: number;
  shearPa: number;
}

export interface BendingStiffnessNm {
  /** Hinge stiffness k in Newton-meters (= Joules). */
  warpNm: number;
  weftNm: number;
}

export interface PhysicalMaterialParams {
  inertia: InertiaParams;
  /** Shell thickness in meters (elastic energy W = thickness * area * psi). */
  thicknessM: number;
  membrane: MembraneStiffnessPa;
  bending: BendingStiffnessNm;
  /** Velocity damping ratio, dimensionless, valid range [0, 1]. The engine's
   *  normal operating band is 0..0.1; this is the one semi-artistic knob and
   *  it is range-validated rather than hidden. */
  dampingRatio: number;
}

/** Machine-readable SI unit table (reviewed by the SI-validation test). */
export interface ParameterUnit {
  path: string;
  unit: string;
}

export function describeParameters(): ParameterUnit[] {
  return [
    { path: "inertia.arealDensityKgM2", unit: "kg/m^2" },
    { path: "thicknessM", unit: "m" },
    { path: "membrane.warpPa", unit: "Pa" },
    { path: "membrane.weftPa", unit: "Pa" },
    { path: "membrane.couplingPa", unit: "Pa" },
    { path: "membrane.shearPa", unit: "Pa" },
    { path: "bending.warpNm", unit: "N*m" },
    { path: "bending.weftNm", unit: "N*m" },
    { path: "dampingRatio", unit: "dimensionless" },
  ];
}

/** Coulomb friction is contact-owned; recorded here so the split is explicit. */
export const FRICTION_OWNERSHIP_NOTE =
  "Coulomb friction mu is owned by the contact system (ContactParams, dimensionless), not the material model.";

// ---------------------------------------------------------------------------
// Calibration representation (Fabric-101-motivated modularity).
// Stores elastic measurements plus RESERVED hysteresis/plasticity metadata.
// No dissipative model is implemented: any non-"none" model tag throws.
// ---------------------------------------------------------------------------

export interface ElasticCalibration {
  /** Source dataset, e.g. "Fabric-101". */
  dataset: string;
  specimenId?: string;
  /** Test protocol, e.g. "uniaxial-warp/weft + 45deg-shear". */
  protocol?: string;
  measuredAt?: string; // ISO date
  /** Fitted moduli in Pa (may differ from live model params after tuning). */
  moduliPa?: { warp: number; weft: number; coupling: number; shear: number };
  notes?: string;
}

export interface HysteresisMetadata {
  /** Only "none" is supported: recoverable-hysteresis MODELS are not implemented. */
  model: "none";
  notes?: string;
}

export interface PlasticityMetadata {
  /** Only "none" is supported: plasticity MODELS are not implemented. */
  model: "none";
  notes?: string;
}

export interface MaterialCalibration {
  elastic: ElasticCalibration;
  hysteresis?: HysteresisMetadata;
  plasticity?: PlasticityMetadata;
  provenanceNotes?: string;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export class MaterialValidationError extends Error {
  readonly code: string;
  constructor(code: string, message: string) {
    super(`invalid material (${code}): ${message}`);
    this.name = "MaterialValidationError";
    this.code = code;
  }
}

function num(v: unknown, path: string): number {
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new MaterialValidationError("non-finite", `${path} must be a finite number, got ${String(v)}`);
  }
  return v;
}

/** Structural validation of physical parameters (SI + positive-definiteness). */
export function validatePhysicalParams(p: PhysicalMaterialParams): void {
  if (p === null || typeof p !== "object") throw new MaterialValidationError("shape", "params must be an object");
  const rho = num((p.inertia ?? {}).arealDensityKgM2, "inertia.arealDensityKgM2");
  if (rho <= 0) throw new MaterialValidationError("non-positive-density", `arealDensityKgM2 must be > 0, got ${rho}`);
  const h = num(p.thicknessM, "thicknessM");
  if (h <= 0) throw new MaterialValidationError("non-positive-thickness", `thicknessM must be > 0, got ${h}`);
  const c00 = num(p.membrane?.warpPa, "membrane.warpPa");
  const c11 = num(p.membrane?.weftPa, "membrane.weftPa");
  const c01 = num(p.membrane?.couplingPa, "membrane.couplingPa");
  const g = num(p.membrane?.shearPa, "membrane.shearPa");
  if (c00 < 0 || c11 < 0 || g < 0) {
    throw new MaterialValidationError("negative-stiffness", "membrane moduli must be >= 0");
  }
  if (c00 * c11 - c01 * c01 < 0) {
    throw new MaterialValidationError(
      "indefinite-coupling",
      `|couplingPa| must satisfy C00*C11 - C01^2 >= 0, got ${c00}*${c11} - ${c01}^2 < 0`,
    );
  }
  const bw = num(p.bending?.warpNm, "bending.warpNm");
  const be = num(p.bending?.weftNm, "bending.weftNm");
  if (bw < 0 || be < 0) throw new MaterialValidationError("negative-stiffness", "bending moduli must be >= 0");
  const d = num(p.dampingRatio, "dampingRatio");
  if (d < 0 || d > 1) throw new MaterialValidationError("damping-range", `dampingRatio must be in [0,1], got ${d}`);
}

/** Rejects any implemented dissipative model (none exist yet — metadata only). */
export function validateCalibration(cal: MaterialCalibration | undefined): void {
  if (cal === undefined) return;
  if (cal === null || typeof cal !== "object") throw new MaterialValidationError("shape", "calibration must be an object");
  if (!cal.elastic || typeof cal.elastic.dataset !== "string" || cal.elastic.dataset.length === 0) {
    throw new MaterialValidationError("calibration", "elastic.dataset (string) is required");
  }
  if (cal.hysteresis !== undefined && cal.hysteresis.model !== "none") {
    throw new MaterialValidationError("not-implemented", `hysteresis model "${String(cal.hysteresis.model)}" is not implemented (metadata only)`);
  }
  if (cal.plasticity !== undefined && cal.plasticity.model !== "none") {
    throw new MaterialValidationError("not-implemented", `plasticity model "${String(cal.plasticity.model)}" is not implemented (metadata only)`);
  }
}

// ---------------------------------------------------------------------------
// Backend-neutral model interface
// ---------------------------------------------------------------------------

export interface MaterialModelEval {
  energy: number;
  grad: Float64Array; // length 3n
  maxStrain: number;
}

export interface ClothMaterialModel {
  readonly kind: string;
  readonly params: PhysicalMaterialParams;
  readonly calibration: MaterialCalibration | undefined;
  /** Total internal energy (membrane + bending), Joules. */
  energy(x: ArrayLike<number>, mesh: ClothMeshData): number;
  /** Total internal energy + gradient (membrane + bending). */
  gradient(x: ArrayLike<number>, mesh: ClothMeshData, out?: Float64Array): MaterialModelEval;
  /** Membrane-only analytic HVP (inexact-Newton approximation: bending is
   *  kept in the gradient but excluded from the Hessian, exactly like the
   *  legacy evalMembraneHvp/FD-oracle pair). */
  hessianVector(x: ArrayLike<number>, p: ArrayLike<number>, mesh: ClothMeshData, out?: Float64Array): Float64Array;
  toLegacy(): ClothMaterial;
  clone(): ClothMaterialModel;
  toJSON(): MaterialModelJSON;
}

export interface MaterialModelJSON {
  kind: string;
  version: number;
  params: PhysicalMaterialParams;
  calibration?: MaterialCalibration;
}

// ---------------------------------------------------------------------------
// Default implementation: orthotropic StVK (delegates to verified code)
// ---------------------------------------------------------------------------

function deepCopy<T>(v: T): T {
  if (v === undefined) return v;
  return JSON.parse(JSON.stringify(v)) as T;
}

function freezeParams(p: PhysicalMaterialParams): PhysicalMaterialParams {
  const c = deepCopy(p);
  Object.freeze(c.inertia);
  Object.freeze(c.membrane);
  Object.freeze(c.bending);
  Object.freeze(c);
  return c;
}

export class OrthotropicStVKMaterial implements ClothMaterialModel {
  static readonly KIND = "orthotropic-stvk";
  static readonly VERSION = 1;

  readonly kind = OrthotropicStVKMaterial.KIND;
  readonly params: PhysicalMaterialParams;
  readonly calibration: MaterialCalibration | undefined;
  /** Legacy view (frozen); the single source the delegated kernels read. */
  private readonly legacy: ClothMaterial;

  constructor(params: PhysicalMaterialParams, calibration?: MaterialCalibration) {
    validatePhysicalParams(params);
    validateCalibration(calibration);
    this.params = freezeParams(params);
    this.calibration = calibration === undefined ? undefined : deepCopy(calibration);
    const p = this.params;
    this.legacy = Object.freeze({
      arealDensityKgM2: p.inertia.arealDensityKgM2,
      thickness: p.thicknessM,
      stretchWarp: p.membrane.warpPa,
      stretchWeft: p.membrane.weftPa,
      stretchCoupling: p.membrane.couplingPa,
      shear: p.membrane.shearPa,
      bendWarp: p.bending.warpNm,
      bendWeft: p.bending.weftNm,
      damping: p.dampingRatio,
    });
  }

  /** Bridge from the legacy struct (validates: unphysical legacy values throw). */
  static fromLegacy(mat: ClothMaterial, calibration?: MaterialCalibration): OrthotropicStVKMaterial {
    return new OrthotropicStVKMaterial(
      {
        inertia: { arealDensityKgM2: mat.arealDensityKgM2 },
        thicknessM: mat.thickness,
        membrane: {
          warpPa: mat.stretchWarp, weftPa: mat.stretchWeft,
          couplingPa: mat.stretchCoupling, shearPa: mat.shear,
        },
        bending: { warpNm: mat.bendWarp, weftNm: mat.bendWeft },
        dampingRatio: mat.damping,
      },
      calibration,
    );
  }

  toLegacy(): ClothMaterial {
    const l = this.legacy;
    return {
      arealDensityKgM2: l.arealDensityKgM2,
      thickness: l.thickness,
      stretchWarp: l.stretchWarp,
      stretchWeft: l.stretchWeft,
      stretchCoupling: l.stretchCoupling,
      shear: l.shear,
      bendWarp: l.bendWarp,
      bendWeft: l.bendWeft,
      damping: l.damping,
    };
  }

  energy(x: ArrayLike<number>, mesh: ClothMeshData): number {
    return internalEnergyOnly(x, mesh, this.legacy);
  }

  gradient(x: ArrayLike<number>, mesh: ClothMeshData, out?: Float64Array): MaterialModelEval {
    return evalInternal(x, mesh, this.legacy, out);
  }

  hessianVector(x: ArrayLike<number>, p: ArrayLike<number>, mesh: ClothMeshData, out?: Float64Array): Float64Array {
    return evalMembraneHvp(x, p, mesh, this.legacy, out);
  }

  clone(): OrthotropicStVKMaterial {
    return new OrthotropicStVKMaterial(deepCopy(this.params), deepCopy(this.calibration));
  }

  toJSON(): MaterialModelJSON {
    return {
      kind: OrthotropicStVKMaterial.KIND,
      version: OrthotropicStVKMaterial.VERSION,
      params: deepCopy(this.params),
      calibration: deepCopy(this.calibration),
    };
  }

  static fromJSON(json: MaterialModelJSON | string): OrthotropicStVKMaterial {
    const o: MaterialModelJSON = typeof json === "string" ? JSON.parse(json) as MaterialModelJSON : json;
    if (o === null || typeof o !== "object") throw new MaterialValidationError("shape", "model JSON must be an object");
    if (o.kind !== OrthotropicStVKMaterial.KIND) {
      throw new MaterialValidationError("unknown-kind", `expected kind "${OrthotropicStVKMaterial.KIND}", got ${String(o.kind)}`);
    }
    if (o.version !== OrthotropicStVKMaterial.VERSION) {
      throw new MaterialValidationError("version", `expected version ${OrthotropicStVKMaterial.VERSION}, got ${String(o.version)}`);
    }
    return new OrthotropicStVKMaterial(o.params, o.calibration);
  }
}
