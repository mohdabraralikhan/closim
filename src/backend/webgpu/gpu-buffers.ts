// GPU-resident buffer model (Phase 2 §5-§6).
// All sizes are pure functions of (nVerts, nTris, nHinges, nContacts) so unit
// tests can validate layout without a GPU device.
//
// Alignment rules applied here (NOT left to WGSL guesswork):
// - positions / velocities are vec4f arrays (xyz + padding) -> stride 16 B
// - scalar f32 arrays stay 4 B, u32 arrays 4 B
// - SimParams uniform is padded to a multiple of 16 B (explicit layout below)
// - SolverStatus is 16 x 4 B = 64 B, read back as ONE compact staging copy

export interface GpuBufferSizes {
  vertexCount: number;
  triangleCount: number;
  hingeCount: number;
  contactCapacity: number;
  /** G1 fixed candidate-pair buffer capacity (default 16384 pairs). */
  pairCapacity?: number;
  /**
   * G2 expanded-primitive scratch capacity (default pairCapacity * 15:
   * 6 VT + 9 EE worst case per triangle pair). Pass a smaller explicit
   * value for validation scenes; the default is production-sized.
   */
  primCapacity?: number;
  /** G1/G2 exclusion-pair buffer capacity in (lo,hi) records (default 6×tris). */
  exclusionCount?: number;
  /** Static collider vertex count: position-class buffers grow by +ns (D3). */
  staticCount?: number;
  /**
   * G5C exact Schwarz domain count (computed from mesh topology before
   * allocation). When absent, sizes fall back to the connected-mesh bound
   * ceil(n/8)+2 — pass the exact count for unusual topologies.
   */
  schwarzDomainCount?: number;
  /**
   * G5.5 exact coarse block-CSR nonzero count (computed from mesh topology
   * before allocation). When absent, sizes fall back to schwarzDoms * 16.
   */
  coarseNnz?: number;
}

export interface GpuLayoutBytes {
  position: number;      // n * 16 (vec4f)
  position0: number;     // n * 16
  velocity: number;      // n * 16
  mass: number;          // n * 4
  inverseMass: number;   // n * 4
  pinMask: number;       // n * 4
  triangles: number;     // m * 3 * 4
  hinges: number;        // h * 4 * 4 (v0,v1,v2,v3 packed u32)
  hingeMeta: number;     // h * 4 * 4 (restAngle, edgeLen, areaSum, kScale f32)
  dmInv: number;         // m * 4 * 4 (a,b,c,d f32)
  restArea: number;      // m * 4
  elementGradient: number; // m * 9 * 4 (triangle-local 3x3, f32)
  elementEnergy: number;   // m * 4 (per-triangle energy, diagnostics)
  hingeGradient: number;   // h * 12 * 4 (hinge-local 4x3, f32)
  hingeEnergyOut: number;  // h * 4 (per-hinge energy, diagnostics)
  gradient: number;      // n * 3 * 4 (assembled vertex gradient)
  rhs: number;           // n * 3 * 4
  searchDirection: number;// n * 3 * 4
  diag: number;          // n * 3 * 4
  pcgResidual: number;   // n * 3 * 4
  pcgSearch: number;     // n * 3 * 4
  pcgAp: number;         // n * 3 * 4
  pcgZ: number;          // n * 3 * 4
  xTrial: number;        // n * 3 * 4
  xReference: number;    // n * 3 * 4
  contactData: number;   // cap * 16 * 4 (frozen contact record, 64 B each)
  simParams: number;     // 64 B uniform (see SIM_PARAMS_BYTES)
  solverStatus: number;  // 64 B status (see SOLVER_STATUS_BYTES)
  reduceScratch: number; // workgroupCount * 4 (partial dot sums)
  // ---- G1 broad-phase (static topology vs dynamic bounds split) ----
  triAabb: number;       // m * 6 * 4 (minXYZ/maxXYZ f32, DYNAMIC: rebuilt per Newton iter)
  triCentroid: number;   // m * 4 * 4 (xyz + 0 vec4f, DYNAMIC)
  mortonKeys: number;    // P(m) * 4 (u32, DYNAMIC; P = next pow2, tail = INF)
  mortonPayload: number; // P(m) * 4 (sorted tri ids u32, DYNAMIC)
  // ---- G6C.1 indexed bitonic sort (STATIC params, DYNAMIC cursor) ----
  sortParams: number;      // T * 16 (vec4u (P, stage, sub, 0) per pass)
  sortCursor: number;      // 16 B pass cursor (u32, reset per rebuild)
  lbvhNodes: number;     // LEGACY blob (superseded by the 5 split buffers below)
  lbvhMin: number;      // nodes * 16 (vec4f minXYZ)
  lbvhMax: number;      // nodes * 16 (vec4f maxXYZ)
  lbvhChild: number;    // nodes * 16 (vec4u left/right/leafTri/isLeaf)
  lbvhRange: number;    // nodes * 16 (vec4u lo/hi, internal only)
  lbvhRoot: number;     // 16 B (single root id)
  candidatePairs: number;// cap * 2 * 4 (CandidatePair {a,b} u32, DYNAMIC)
  pairCount: number;      // 16 B padded atomic counter (DYNAMIC)
  overflowFlag: number;   // 16 B padded (DYNAMIC diagnostic)
  exclusionKeys: number;  // exclCap * 8 (sorted (lo,hi) vec2u, STATIC)
  // ---- G4B incidence maps (STATIC topology, uploaded once) ----
  vertexElementOffsets: number; // (n+1) * 4 (CSR row pointers)
  vertexElementIds: number;     // 3m * 4 (triangle id per record)
  vertexElementCorners: number; // 3m * 4 (local corner 0..2 per record)
  vertexHingeOffsets: number;   // (n+1) * 4
  vertexHingeIds: number;       // 4h * 4 (hinge id per record)
  vertexHingeCorners: number;   // 4h * 4 (local corner 0..3, first-match-wins)
  // ---- G5B block-Jacobi (rebuilt per Newton iter, DYNAMIC) ----
  blockInv: number;         // n * 9 * 4 (row-major 3x3 inverse per vertex)
  blockFlag: number;        // n * 4 (u32 1 = Cholesky ok, 0 = Jacobi fallback)
  // ---- G5C Schwarz (STATIC topology, DYNAMIC factors) ----
  schwarzDomain: number;    // n * 4 (u32 vertex -> domain)
  schwarzLocal: number;     // n * 4 (u32 vertex -> local index)
  schwarzVerts: number;     // nDoms * 8 * 4 (u32 domain members, 0xFFFFFFFF pad)
  schwarzMat: number;       // nDoms * 576 * 4 (assembled local matrices)
  schwarzInv: number;       // nDoms * 576 * 4 (explicit local inverses)
  schwarzFlag: number;      // nDoms * 4 (u32 1 = Cholesky ok, 0 = Jacobi fallback)
  // ---- G5.5 assembled coarse block-CSR (STATIC pattern, DYNAMIC values) ----
  coarseRowOffsets: number; // (nDoms+1) * 4 (block counts per row)
  coarseColIndices: number; // nnz * 4 (block column per nonzero, sorted per row)
  coarseBlockRows: number;  // nnz * 4 (block row per nonzero, for transpose lookup)
  coarseBlockValues: number;// nnz * 9 * 4 (row-major 3x3 per block, symmetrized)
  // ---- G6B batched Armijo (DYNAMIC per batch) ----
  armijoAlphas: number;     // 8 * 4 (candidate alphas; trust applied via trustScaleStore)
  armijoCandidates: number; // 8 * 8 * 4 (per-candidate verdict rows)
  armijoStatus: number;     // 16 * 4 (compact batch status, one readback)
  armijoCur: number;        // 16 B current candidate index (u32)
  // ---- G6C.2 GPU Newton control (DYNAMIC per round/step) ----
  e0Store: number;          // 16 B accepted-energy mirror (f32, GPU-maintained)
  gtdxStore: number;        // 16 B descent-dot mirror (f32, GPU-maintained)
  trustScaleStore: number;  // 16 B trust-region scale mirror (f32)
  descentDir: number;       // n*3 * 4 (Jacobi descent direction, GPU-selected)
  newtonStatus: number;     // 20 * 4 (compact round status, one read per round)
  newtonCtl: number;        // 16 * 4 (control flags: done/fallback/accepted/commit/alpha)
  // ---- G5.5 inner coarse-PCG vectors/scalars (DYNAMIC per solve) ----
  coarseX: number;          // nDoms*3 * 4 (inner solution)
  coarseR: number;          // nDoms*3 * 4 (inner residual)
  coarseP: number;          // nDoms*3 * 4 (inner search dir)
  coarseAp: number;         // nDoms*3 * 4 (inner A*p)
  coarseZ: number;          // nDoms*3 * 4 (inner preconditioned residual)
  coarseProd: number;       // nDoms*3 * 4 (inner dot products)
  coarseAlpha: number;      // 16 B inner alpha (separate from outer slots)
  coarseBeta: number;       // 16 B inner beta
  coarseRzPrev: number;     // 16 B inner previous rz
  coarseBreak: number;      // 16 B inner breakdown latch
  masContactSpan: number;   // 16 B cross-aggregate contact counter (atomic)
  // ---- G5D MAS coarse vectors (DYNAMIC per solve) ----
  masCoarseR: number;       // nDoms * 3 * 4 (restricted residual)
  masCoarseZ: number;       // nDoms * 3 * 4 (scaled coarse correction)
  masCoarseDiag: number;    // nDoms * 3 * 4 (R diag(H) P diagonal, per solve)
  // ---- G2 contact pipeline (STATIC topology vs DYNAMIC primitive scratch) ----
  primIdsVT: number;      // primCap * 16 (vec4u (p,a,b,c) extended ids, DYNAMIC)
  primIdsEE: number;      // primCap * 16 (vec4u (a,b,c,d) extended ids, DYNAMIC)
  vtSTD: number;          // primCap * 16 (s,t,dist,0 vec4f)
  vtR: number;            // primCap * 16 (rx,ry,rz,degen)
  eeSTD: number;          // primCap * 16
  eeR: number;            // primCap * 16
  primTOIVT: number;       // primCap * 4 (f32 TOI, DYNAMIC)
  primFlagVT: number;      // primCap * 4 (u32 status, DYNAMIC)
  primTOIEE: number;       // primCap * 4 (DYNAMIC)
  primFlagEE: number;      // primCap * 4 (DYNAMIC)
  primCountVT: number;    // 16 B padded atomic (DYNAMIC)
  primCountEE: number;    // 16 B padded atomic (DYNAMIC)
  contactTOI: number;     // contactCap * 4 (f32 per compact record, DYNAMIC)
  contactDist: number;    // contactCap * 4 (frozen distance sidecar)
  contactW: number;       // cap * 16 (frozen w vec4f)
  contactN: number;       // cap * 16 (nx,ny,nz,kind)
  contactId: number;      // cap * 16 (vec4u ids, extended space)
  contactPrm: number;     // cap * 16 (dHat,kappa,mu/floorY,eps)
  pairScanned: number;    // 16 B traversal scanned counter
  gradientAlt: number;    // n * 3 * 4 (FD assemble scratch)
  negRhs: number;         // n * 3 * 4 (PCG right-hand side b = -g)
  contactCount: number;   // 16 B padded atomic (DYNAMIC)
  contactOverflow: number;// 16 B padded (DYNAMIC diagnostic)
  contactScanned: number; // 16 B padded (DYNAMIC diagnostic)
  contactFail: number;    // 16 B padded (DYNAMIC diagnostic)
  // ---- G3 device path (FEM/HVP/PCG/contact-force scratch) ----
  elementGradientB: number; // m * 9 * 4 (FD scratch)
  elementHVP: number;     // m * 9 * 4 (G4A analytic membrane dgrad per triangle)
  hpPlus: number;         // n * 3 * 4 (FD + side)
  hpMinus: number;        // n * 3 * 4 (FD - side)
  hpMembrane: number;     // n * 3 * 4 (FD-combined membrane HVP)
  hpBarrier: number;      // n * 3 * 4 (frozen barrier HVP)
  hvpXMinus: number;      // (n+ns) * 16 (x - h*p vec4f)
  contactForce: number;   // n * 3 * 4 (assembled barrier+friction residual)
  contactForceZero: number; // n * 3 * 4 (pre-zeroed, bound during FD passes)
  contactScratch: number; // cap * 8 * 4 (barrier per-contact scratch)
  contactEnergy: number;  // cap * 4
  frictionScratch: number;// cap * 16 (physical force vec4f per contact)
  slip: number;           // (n+ns) * 16 (x - xStep vec4f)
  laggedN: number;        // cap * 16 (nx,ny,nz,lambdaN vec4f)
  jvOut: number;          // cap * 4 (J*v per contact)
  coeffOut: number;       // cap * 4 (kappa*b''(d) per contact)
  contactDiag: number;    // n * 3 * 4 (Jacobi contact curvature)
  pcgProd: number;        // n * 3 * 4 (per-element products for reductions)
  execMarker: number;     // 16 B [magic, epoch, stageMask, dispatchCount]
  pinPos: number;         // n * 16 (STATIC pinned targets vec4f)
  alphaSlot: number;      // 16 B PCG alpha storage scalar
  betaSlot: number;       // 16 B PCG beta storage scalar
  rzPrevSlot: number;     // 16 B PCG previous rz scalar
  breakFlag: number;      // 16 B PCG breakdown latch (f32 0/1)
  diagScratch: number;    // groups8 * 8 * 4 (diagnostics partials)
  uniformBank: number;    // 64 * 16 B scalar/vector uniform slots (executor-managed)
}

/** G3 uniform-bank slot indices (16 B each; see gpu-executor.ts UniformBank). */
export const GpuUniformSlot = {
  BlasParams: 0,
  StageBit: 1,
  SortN: 2,
  SortStage: 3,
  SortSub: 4,
  Pad: 5,
  DHat: 6,
  Kappa: 7,
  Mu: 8,
  FricEps: 9,
  FloorY: 10,
  FloorOn: 11,
  DMin: 12,
  VtPrimTotal: 13,
  EePrimTotal: 14,
  PairCountIn: 15,
  ContactCapacity: 16,
  ExclusionCount: 17,
  PairCapacity: 18,
  Thickness: 19,
  PrimCount: 20,
  SceneMin: 21, // vec4f (16 B slot)
  SceneMax: 22, // vec4f
  JacobiBeta: 23,
  MatC00: 24,
  MatC11: 25,
  MatC01: 26,
  MatG: 27,
  MatThickness: 28,
  OutSel: 29,
  ReduceCount: 30,
  ReduceGroups: 31,
  SchwarzDomains: 32,
  MasOmega: 33,
  MasCount: 34,
  CoarseCount: 35,
  CoarseGroups: 36,
  ArmijoE0: 37,
  ArmijoGtdx: 38,
  ArmijoK: 39,
  ArmijoPcgBd: 40,
  ArmijoNewtonConv: 41,
  NewtonTol: 42,
  NewtonTrust: 43,
} as const;

/** G6C.2 Newton status lane indices (20 f32; see newton-control.wgsl). */
export const NewtonStatusLane = {
  Iteration: 0,
  Converged: 1,
  Failure: 2,
  DirectionValid: 3,
  PcgBreakdown: 4,
  ArmijoAccepted: 5,
  ArmijoBatchIndex: 6,
  SelectedAlpha: 7,
  SelectedTrialIndex: 8,
  GradNorm: 9,
  StepNorm: 10,
  Merit: 11,
  Energy: 12,
  MinDistance: 13,
  MinToi: 14,
  ContactOverflow: 15,
  CcdFailure: 16,
  BarrierFailure: 17,
  Residual: 18,
  DescentDot: 19,
} as const;

/** G6C.2 Newton control lane indices (16 f32 scratch). */
export const NewtonCtlLane = {
  Done: 0,
  Fallback: 1,
  Accepted: 2,
  DoCommit: 3,
  Alpha: 4,
} as const;
export type GpuUniformSlot = (typeof GpuUniformSlot)[keyof typeof GpuUniformSlot];
/** Uniform bank capacity in 16 B slots. */
export const UNIFORM_BANK_SLOTS = 64;
/**
 * Uniform-bank stride in bytes. WebGPU requires uniform-buffer binding
 * offsets to be multiples of minUniformBufferOffsetAlignment (256 on every
 * real backend), so each 16 B slot rides in its own 256 B stride. The bank
 * costs 16 KiB — negligible next to the state buffers.
 */
export const UNIFORM_SLOT_STRIDE = 256;

/** G2 worst-case primitives per triangle pair: 6 VT + 9 EE. */
export const G2_PRIM_PER_PAIR = 15;

/** Next power of two >= v (v >= 1). Bitonic sort lane count. */
export function nextPow2(v: number): number {
  let p = 1;
  while (p < v) p *= 2;
  return p;
}

/** G1 LBVH node: bounds minXYZ/maxXYZ (24 B) + left/right/leafTri/leafCount (16 B) + 24 B pad = 64 B. */
export const LBVH_NODE_BYTES = 64;
/** G1 candidate pair record: {a: u32, b: u32} = 8 B. */
export const CANDIDATE_PAIR_BYTES = 8;
/**
 * G2 exclusion encoding: sorted (lo,hi) u32 pairs (vec2u), binary-searched
 * per component. No triCount ceiling (supersedes the G1 packed-u32 path).
 */
export const EXCLUSION_PAIR_BYTES = 8;

export const SIM_PARAMS_BYTES = 64;
export const SOLVER_STATUS_BYTES = 64;
/** Frozen contact record: [w0,w1,w2,w3, nx,ny,nz,lambdaN, a,b,c,d, kind,pad,pad,pad, d, dHat, kappa, mu] as f32/u32 mix, padded to 64 B. */
export const CONTACT_RECORD_BYTES = 64;

/** WGSL SimParams layout mirror — keep field order identical to params.wgsl. */
export interface SimParams {
  dt: number; invDt2: number;
  gravityX: number; gravityY: number; gravityZ: number;
  vertexCount: number; triangleCount: number; hingeCount: number; contactCount: number;
  newtonIteration: number; pcgIteration: number;
  lineSearchAlpha: number; trustRegion: number;
  barrierActivation: number; barrierEpsilon: number; frictionMu: number;
  // pad to 64 B (4 spare f32)
}

/** Compact per-Newton-iteration status — the ONLY hot-loop readback (§19). */
export interface SolverStatus {
  energy: number; barrierEnergy: number; gradNorm: number; directionDotGradient: number;
  minDistance: number; minToi: number;
  finite: number; ccdSafe: number; barrierSafe: number;
  pcgBreakdown: number; converged: number;
  // + 5 spare f32 pad to 64 B
}

/** G5C Schwarz sizing: fixed pad width 8 verts/domain, 24x24 dense blocks. */
export const SCHWARZ_PAD_VERTS = 8;
export const SCHWARZ_DOF = 24;
export const SCHWARZ_MAT_ENTRIES = 576;
/** G5.5 coarse assembly: max neighbor blocks per row the GPU kernel holds. */
export const COARSE_MAX_DEGREE = 32;

export function gpuLayoutBytes(s: GpuBufferSizes): GpuLayoutBytes {
  const { vertexCount: n, triangleCount: m, hingeCount: h, contactCapacity: c } = s;
  const schwarzDoms = s.schwarzDomainCount ?? Math.ceil(n / SCHWARZ_PAD_VERTS) + 2;
  const coarseNnz = s.coarseNnz ?? schwarzDoms * 16;
  const pairs = s.pairCapacity ?? 16384;
  const prims = s.primCapacity ?? pairs * G2_PRIM_PER_PAIR;
  // G6C.1: T = S(S+1)/2 sub-passes for P = nextPow2(m) lanes.
  const sortP = nextPow2(Math.max(m, 1));
  const sortStages = Math.log2(sortP);
  const sortT = Math.max(1, (sortStages * (sortStages + 1)) / 2);
  const lbvhNodeCount = Math.max(1, 2 * m - 1);
  // D3: cloth ++ static extended position-class buffers (static tail static).
  const ns = s.staticCount ?? 0;
  const nx = n + ns;
  return {
    position: nx * 16,
    position0: nx * 16,
    velocity: n * 16,
    mass: n * 4,
    inverseMass: n * 4,
    pinMask: n * 4,
    triangles: m * 3 * 4,
    hinges: h * 4 * 4,
    hingeMeta: h * 4 * 4,
    dmInv: m * 4 * 4,
    restArea: m * 4,
    elementGradient: m * 9 * 4,
    elementHVP: m * 9 * 4,
    elementEnergy: Math.max(m, 1) * 4,
    hingeGradient: h * 12 * 4,
    hingeEnergyOut: Math.max(h, 1) * 4,
    gradient: n * 3 * 4,
    rhs: n * 3 * 4,
    searchDirection: n * 3 * 4,
    diag: n * 3 * 4,
    pcgResidual: n * 3 * 4,
    pcgSearch: n * 3 * 4,
    pcgAp: n * 3 * 4,
    pcgZ: n * 3 * 4,
    xTrial: nx * 16, // vec4f (extended): Newton trial / FD + side
    xReference: n * 16, // vec4f: y_hat snapshot
    contactData: c * CONTACT_RECORD_BYTES,
    simParams: SIM_PARAMS_BYTES,
    solverStatus: SOLVER_STATUS_BYTES,
    reduceScratch: Math.max(1, Math.ceil((n * 3) / 64)) * 4,
    triAabb: m * 6 * 4,
    triCentroid: m * 4 * 4,
    mortonKeys: Math.max(nextPow2(m), 1) * 4,
    mortonPayload: Math.max(nextPow2(m), 1) * 4,
    sortParams: sortT * 16,
    sortCursor: 16,
    lbvhNodes: lbvhNodeCount * LBVH_NODE_BYTES,
    lbvhMin: lbvhNodeCount * 16,
    lbvhMax: lbvhNodeCount * 16,
    lbvhChild: lbvhNodeCount * 16,
    lbvhRange: lbvhNodeCount * 16,
    lbvhRoot: 16,
    candidatePairs: Math.max(pairs, 1) * CANDIDATE_PAIR_BYTES,
    pairCount: 16,
    overflowFlag: 16,
    exclusionKeys: Math.max(s.exclusionCount ?? m * 6, 1) * EXCLUSION_PAIR_BYTES,
    vertexElementOffsets: (n + 1) * 4,
    vertexElementIds: Math.max(3 * m, 1) * 4,
    vertexElementCorners: Math.max(3 * m, 1) * 4,
    vertexHingeOffsets: (n + 1) * 4,
    vertexHingeIds: Math.max(4 * h, 1) * 4,
    vertexHingeCorners: Math.max(4 * h, 1) * 4,
    blockInv: Math.max(n, 1) * 9 * 4,
    blockFlag: Math.max(n, 1) * 4,
    schwarzDomain: Math.max(n, 1) * 4,
    schwarzLocal: Math.max(n, 1) * 4,
    schwarzVerts: Math.max(schwarzDoms, 1) * SCHWARZ_PAD_VERTS * 4,
    schwarzMat: Math.max(schwarzDoms, 1) * SCHWARZ_MAT_ENTRIES * 4,
    schwarzInv: Math.max(schwarzDoms, 1) * SCHWARZ_MAT_ENTRIES * 4,
    schwarzFlag: Math.max(schwarzDoms, 1) * 4,
    masCoarseR: Math.max(schwarzDoms, 1) * 3 * 4,
    masCoarseZ: Math.max(schwarzDoms, 1) * 3 * 4,
    masCoarseDiag: Math.max(schwarzDoms, 1) * 3 * 4,
    coarseRowOffsets: (Math.max(schwarzDoms, 1) + 1) * 4,
    coarseColIndices: Math.max(coarseNnz, 1) * 4,
    coarseBlockRows: Math.max(coarseNnz, 1) * 4,
    coarseBlockValues: Math.max(coarseNnz, 1) * 9 * 4,
    coarseX: Math.max(schwarzDoms, 1) * 3 * 4,
    coarseR: Math.max(schwarzDoms, 1) * 3 * 4,
    coarseP: Math.max(schwarzDoms, 1) * 3 * 4,
    coarseAp: Math.max(schwarzDoms, 1) * 3 * 4,
    coarseZ: Math.max(schwarzDoms, 1) * 3 * 4,
    coarseProd: Math.max(schwarzDoms, 1) * 3 * 4,
    coarseAlpha: 16,
    coarseBeta: 16,
    coarseRzPrev: 16,
    coarseBreak: 16,
    masContactSpan: 16,
    armijoAlphas: 8 * 4,
    armijoCandidates: 8 * 8 * 4,
    armijoStatus: 16 * 4,
    armijoCur: 16,
    e0Store: 16,
    gtdxStore: 16,
    trustScaleStore: 16,
    descentDir: n * 3 * 4,
    newtonStatus: 20 * 4,
    newtonCtl: 16 * 4,
    primIdsVT: Math.max(prims, 1) * 16,
    primIdsEE: Math.max(prims, 1) * 16,
    vtSTD: Math.max(prims, 1) * 16,
    vtR: Math.max(prims, 1) * 16,
    eeSTD: Math.max(prims, 1) * 16,
    eeR: Math.max(prims, 1) * 16,
    primTOIVT: Math.max(prims, 1) * 4,
    primFlagVT: Math.max(prims, 1) * 4,
    primTOIEE: Math.max(prims, 1) * 4,
    primFlagEE: Math.max(prims, 1) * 4,
    primCountVT: 16,
    primCountEE: 16,
    contactTOI: Math.max(c, 1) * 4,
    contactDist: Math.max(c, 1) * 4,
    contactW: Math.max(c, 1) * 16,
    contactN: Math.max(c, 1) * 16,
    contactId: Math.max(c, 1) * 16,
    contactPrm: Math.max(c, 1) * 16,
    pairScanned: 16,
    gradientAlt: n * 3 * 4,
    negRhs: n * 3 * 4,
    contactCount: 16,
    contactOverflow: 16,
    contactScanned: 16,
    contactFail: 16,
    elementGradientB: m * 9 * 4,
    hpPlus: n * 3 * 4,
    hpMinus: n * 3 * 4,
    hpMembrane: n * 3 * 4,
    hpBarrier: n * 3 * 4,
    hvpXMinus: (n + ns) * 16,
    contactForce: n * 3 * 4,
    contactForceZero: n * 3 * 4,
    contactScratch: Math.max(c, 1) * 8 * 4,
    contactEnergy: Math.max(c, 1) * 4,
    frictionScratch: Math.max(c, 1) * 16,
    slip: (n + ns) * 16,
    laggedN: Math.max(c, 1) * 16,
    jvOut: Math.max(c, 1) * 4,
    coeffOut: Math.max(c, 1) * 4,
    contactDiag: n * 3 * 4,
    pcgProd: n * 3 * 4,
    execMarker: 16,
    pinPos: Math.max(n, 1) * 16,
    alphaSlot: 16,
    betaSlot: 16,
    rzPrevSlot: 16,
    breakFlag: 16,
    diagScratch: Math.max(1, Math.ceil(Math.max(n * 3, m, h, c) / 64)) * 8 * 4,
    uniformBank: UNIFORM_BANK_SLOTS * UNIFORM_SLOT_STRIDE,
  };
}

/** Total device bytes for a scene (excludes staging/readback copies). */
export function gpuTotalBytes(s: GpuBufferSizes): number {
  const l = gpuLayoutBytes(s);
  return Object.values(l).reduce((a, b) => a + b, 0);
}

/** Pack xyz Float64 positions into vec4f-padded Float32 (w = 0). */
export function packVec4Positions(xyz: ArrayLike<number>, n: number): Float32Array {
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = Math.fround(xyz[i * 3]);
    out[i * 4 + 1] = Math.fround(xyz[i * 3 + 1]);
    out[i * 4 + 2] = Math.fround(xyz[i * 3 + 2]);
    out[i * 4 + 3] = 0;
  }
  return out;
}

/** Unpack vec4f-padded Float32 back to interleaved xyz Float64Array. */
export function unpackVec4Positions(packed: ArrayLike<number>, n: number): Float64Array {
  const out = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    out[i * 3] = packed[i * 4];
    out[i * 3 + 1] = packed[i * 4 + 1];
    out[i * 3 + 2] = packed[i * 4 + 2];
  }
  return out;
}

/** Serialize SimParams to a 64 B uniform payload (little-endian f32/u32 mix). */
export function encodeSimParams(p: SimParams): ArrayBuffer {
  const buf = new ArrayBuffer(SIM_PARAMS_BYTES);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  f[0] = p.dt; f[1] = p.invDt2;
  f[2] = p.gravityX; f[3] = p.gravityY; f[4] = p.gravityZ;
  u[5] = 0; // pad
  u[6] = p.vertexCount; u[7] = p.triangleCount; u[8] = p.hingeCount; u[9] = p.contactCount;
  u[10] = p.newtonIteration; u[11] = p.pcgIteration;
  f[12] = p.lineSearchAlpha; f[13] = p.trustRegion;
  f[14] = p.barrierActivation; f[15] = p.frictionMu;
  void p.barrierEpsilon;
  return buf;
}

/** Decode the 64 B solver-status payload produced by diagnostics.wgsl. */
export function decodeSolverStatus(buf: ArrayBuffer): SolverStatus {
  const f = new Float32Array(buf);
  // Flag lanes are f32 1.0/0.0 on device (never bit-cast integers): threshold.
  // minDistance uses the 1e30 empty-set sentinel; minToi the 2.0 NO_HIT
  // sentinel. Both normalize to Infinity (mirror/CPU convention).
  const flag = (i: number): number => (f[i] > 0.5 ? 1 : 0);
  return {
    energy: f[0], barrierEnergy: f[1], gradNorm: f[2], directionDotGradient: f[3],
    minDistance: f[4] >= 1e29 ? Infinity : f[4],
    minToi: f[5] >= 2.0 - 1e-6 ? Infinity : f[5],
    finite: flag(6), ccdSafe: flag(7), barrierSafe: flag(8),
    pcgBreakdown: flag(9), converged: flag(10),
  };
}
