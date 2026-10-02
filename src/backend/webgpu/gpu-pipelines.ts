// G0 pipeline registry: WGSL source strings + pipeline descriptors.
// WGSL files under shaders/*.wgsl are the human-readable source of truth;
// the string constants below MUST match them (validated by tests/webgpu
// shader-consistency test). Embedding strings keeps the Node/CI build working
// without a raw-text loader, while the .wgsl files serve editors/tooling.
//
// Workgroup default: @workgroup_size(64) for all 1D vector kernels (§23).
// Benchmark 32/64/128/256 where device limits permit — do not assume one
// vendor's optimum transfers to another.

export const WORKGROUP_DEFAULT = 64;

export const SHADER_NAMES = [
  "predictor",
  "membrane-gradient",
  "membrane-hvp",
  "assemble-gather",
  "block-jacobi",
  "schwarz",
  "mas",
  "coarse",
  "armijo",
  "newton-control",
  "bending-gradient",
  "assemble-gradient",
  "barrier-gradient",
  "friction",
  "newton-rhs",
  "barrier-hvp",
  "hessian-vector",
  "jacobi",
  "pcg-update",
  "pcg-reduce",
  "apply-step",
  "diagnostics",
  "marker",
  "blas",
  "contact-force",
  "broadphase-aabb",
  "broadphase-morton",
  "broadphase-sort",
  "broadphase-lbvh",
  "broadphase-traverse",
  "closest-vt",
  "closest-ee",
  "ccd-vt",
  "ccd-ee",
  "contact-compact",
] as const;

export type ShaderName = (typeof SHADER_NAMES)[number];

/** Entry points per shader module (must match `fn` names in the .wgsl). */
export const SHADER_ENTRY_POINTS: Record<ShaderName, string[]> = {
  "predictor": ["main"],
  "membrane-gradient": ["main"],
  "membrane-hvp": ["membrane_hvp", "assemble_hvp"],
  "assemble-gather": ["gather_membrane_grad", "gather_membrane_hvp", "gather_hinge_grad"],
  "block-jacobi": ["bj_build_factor"],
  "schwarz": ["schwarz_assemble", "schwarz_factor", "schwarz_apply"],
  "mas": ["mas_restrict", "mas_coarse_scale", "mas_prolongate_add", "mas_coarse_diag", "mas_contact_span"],
  "newton-control": ["pcg_report", "trust_compute", "descent_check", "select_fallback", "commit_arm", "commit_apply", "commit_copy_if", "commit_lagged_if", "newton_check", "round_report"],
  "armijo": ["apply_0", "apply_1", "apply_2", "apply_3", "apply_4", "apply_5", "apply_6", "apply_7", "armijo_clear_records", "armijo_clear_counts", "armijo_record", "armijo_select"],
  "coarse": ["coarse_assemble_values", "coarse_spmv", "c_init", "c_update_xr", "c_update_z", "c_update_p2", "c_mul", "c_break", "coarse_reduce_stage1", "coarse_reduce_stage2"],
  "pcg-update": ["pcg_init", "pcg_update_xr", "pcg_update_p", "pcg_update_z", "pcg_update_p2", "pcg_init_bj", "pcg_update_z_bj"],
  "bending-gradient": ["main"],
  "assemble-gradient": ["main"],
  "barrier-gradient": ["main"],
  "friction": ["main"],
  "newton-rhs": ["main"],
  "hessian-vector": ["main"],
  "jacobi": ["main"],
  "pcg-reduce": ["reduce_stage1", "reduce_stage2", "reduce_max_stage1", "reduce_max_stage2"],
  "apply-step": ["main", "apply_step_vec4", "enforce_pins"],
  "diagnostics": ["diag_stage1", "diag_stage2"],
  "marker": ["mark_stage"],
  "blas": ["axpy", "copy", "scale", "zero", "add_into", "fd_combine", "mul", "absv", "neg_div", "axpy_vec4", "axpy_v4", "copy_v4", "sub_v4", "scale_v4", "sdiv", "scopy", "sflag_le"],
  "contact-force": ["assemble_contact_force", "commit_lagged", "velocity_filter"],
  "barrier-hvp": ["barrier_hvp_project", "assemble_barrier_hvp", "contact_diag"],
  "broadphase-aabb": ["main"],
  "broadphase-morton": ["main"],
  "broadphase-sort": ["bitonic_sort_step", "sort_next", "sort_step_indexed", "sort_reset", "sort_step_derived"],
  "broadphase-lbvh": ["lbvh_seed_leaves", "lbvh_build", "lbvh_find_root", "lbvh_refit"],
  "broadphase-traverse": ["traverse"],
  "closest-vt": ["closest_vt"],
  "closest-ee": ["closest_ee"],
  "ccd-vt": ["ccd_vt"],
  "ccd-ee": ["ccd_ee"],
  "contact-compact": ["expand_pairs", "compact_pairs", "compact_pairs_ee", "compact_floor"],
};

export function dispatchWorkgroups(count: number, workgroupSize = WORKGROUP_DEFAULT): number {
  return Math.max(1, Math.ceil(count / workgroupSize));
}

/**
 * Validate a compiled shader module at init time (§31: validate modules and
 * pipelines at initialization). On a real device this compiles the module and
 * surfaces WGSL errors eagerly instead of at first dispatch.
 */
export async function validateShaderModule(
  device: any,
  label: string,
  wgslSource: string,
): Promise<{ ok: boolean; messages: string }> {
  try {
    const module = device.createShaderModule({ label, code: wgslSource });
    const info = await module.getCompilationInfo?.();
    if (!info) return { ok: true, messages: "" };
    const errors = (info.messages ?? []).filter((m: any) => m.type === "error");
    const text = (info.messages ?? []).map((m: any) => `${m.type}: ${m.message}`).join("\n");
    return { ok: errors.length === 0, messages: text };
  } catch (err) {
    return { ok: false, messages: err instanceof Error ? err.message : String(err) };
  }
}
