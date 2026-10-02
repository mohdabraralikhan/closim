export interface ClothMaterial {
  /** Areal density in kg/m^2. Lumped mass: m_i = arealDensityKgM2 * (sum A_t / 3).
   *  NOTE: this is NOT volumetric density; thickness enters only the elastic
   *  energy (W = thickness * area * psi), never inertia. */
  arealDensityKgM2: number;
  thickness: number; // m
  stretchWarp: number; // C00
  stretchWeft: number; // C11
  stretchCoupling: number; // C01
  shear: number; // G
  bendWarp: number;
  bendWeft: number;
  damping: number; // 0..0.1 velocity damping
}

export const DEFAULT_MATERIAL: ClothMaterial = {
  arealDensityKgM2: 0.15,
  thickness: 0.001,
  stretchWarp: 20000,
  stretchWeft: 20000,
  stretchCoupling: 0,
  shear: 5000,
  bendWarp: 1e-5,
  bendWeft: 1e-5,
  damping: 0.001,
};

export interface Hinge {
  v0: number; // edge a
  v1: number; // edge b
  v2: number; // opposite in tri A
  v3: number; // opposite in tri B
  restAngle: number;
  edgeLen: number;
  areaSum: number;
}
