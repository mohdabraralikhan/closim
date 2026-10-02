// G3 tiny vector kernels (BLAS-level): fully GPU-side PCG/HVP bookkeeping.
// These keep scalar-heavy glue (axpy, combine, zero) on device so PCG never
// round-trips vectors through JS. Only 4-byte dot scalars and the 64 B status
// cross to CPU (documented sync points in gpu-executor.ts).
//
// One shared bind layout for every entry:
//   @binding(0) BlasParams uniform { count, alpha, beta, _pad } (16 B)
//   @binding(1) x    : read-only  f32 vector
//   @binding(2) y    : read-write f32 vector (in-place target where noted)
//   @binding(3) out  : write      f32 vector (where noted)

struct BlasParams {
  count : u32,
  alpha : f32,
  beta : f32,
  _pad : f32,
};

@group(0) @binding(0) var<uniform> bp : BlasParams;
@group(0) @binding(1) var<storage, read> x : array<f32>;
@group(0) @binding(2) var<storage, read> y : array<f32>;
@group(0) @binding(3) var<storage, read_write> out : array<f32>;
// In-place target (axpy/add_into only). Split from y because WebGPU forbids
// the same buffer as read-only AND writable in one pass (mul(x,x) pattern).
@group(0) @binding(7) var<storage, read_write> yw : array<f32>;
// vec4-path bindings (own entries below; pruned from f32 entries' layouts)
@group(0) @binding(4) var<storage, read> x4 : array<vec4f>;
@group(0) @binding(5) var<storage, read_write> y4 : array<vec4f>;
@group(0) @binding(6) var<storage, read_write> out4 : array<vec4f>;

// y[i] = alpha * x[i] + y[i] (in-place via yw)
@compute @workgroup_size(64)
fn axpy(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= bp.count) { return; }
  yw[i] = bp.alpha * x[i] + yw[i];
}

// out[i] = x[i] (copy; y binding unused but present for layout sharing)
@compute @workgroup_size(64)
fn copy(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= bp.count) { return; }
  out[i] = x[i];
}

// out[i] = alpha * x[i]
@compute @workgroup_size(64)
fn scale(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= bp.count) { return; }
  out[i] = bp.alpha * x[i];
}

// out[i] = 0
@compute @workgroup_size(64)
fn zero(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= bp.count) { return; }
  out[i] = 0.0;
}

// y[i] += x[i] (in-place via yw)
@compute @workgroup_size(64)
fn add_into(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= bp.count) { return; }
  yw[i] = yw[i] + x[i];
}

// out[i] = (x[i] - y[i]) * alpha — central-difference HVP combine, where
// x = g(x+h*p), y = g(x-h*p), alpha = 1/(2h).
@compute @workgroup_size(64)
fn fd_combine(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= bp.count) { return; }
  out[i] = (x[i] - y[i]) * bp.alpha;
}

// out[i] = x[i] * y[i] (dot-product prep for pcg-reduce)
@compute @workgroup_size(64)
fn mul(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= bp.count) { return; }
  out[i] = x[i] * y[i];
}

// out[i] = abs(x[i]) (trust-region max-norm prep for reduce_max)
@compute @workgroup_size(64)
fn absv(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= bp.count) { return; }
  out[i] = abs(x[i]);
}

// out[i] = -x[i] / y[i] (Jacobi-scaled steepest descent; y = diagSafe)
@compute @workgroup_size(64)
fn neg_div(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= bp.count) { return; }
  out[i] = -x[i] / y[i];
}

// y4[v].xyz += alpha * x[3v..] (vec4 state stepped by packed-f32 direction;
// used for FD perturbations x +/- h*p without a packed position copy).
// bp.count is a VERTEX count here (dispatch width is exact regardless).
@compute @workgroup_size(64)
fn axpy_vec4(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= bp.count) { return; }
  y4[v] = y4[v] + vec4f(bp.alpha * vec3f(x[v * 3u], x[v * 3u + 1u], x[v * 3u + 2u]), 0.0);
}

// y4[v] += alpha * x4[v] (vec4 axpy: trial lerp, commit blends)
@compute @workgroup_size(64)
fn axpy_v4(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= bp.count) { return; }
  y4[v] = y4[v] + bp.alpha * x4[v];
}

// out4 = x4 (vec4 copy: accept trial, snapshot state)
@compute @workgroup_size(64)
fn copy_v4(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= bp.count) { return; } // bp.count = vertexCount here
  out4[v] = x4[v];
}

// out4 = (x4 - y4) (vec4 subtract: velocity numerator x - x0)
@compute @workgroup_size(64)
fn sub_v4(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= bp.count) { return; }
  out4[v] = x4[v] - y4[v];
}

// y4 = alpha * y4 (vec4 scale: velocity damping / 1/h factor)
@compute @workgroup_size(64)
fn scale_v4(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= bp.count) { return; }
  y4[v] = bp.alpha * y4[v];
}

// scalar ops (single-lane; bp.count >= 1): out[0] = x[0]/y[0], guarded.
// Used for PCG alpha/beta entirely on device (no scalar readback mid-solve).
@compute @workgroup_size(64)
fn sdiv(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  let den = y[0];
  out[0] = select(0.0, x[0] / den, abs(den) > 1e-30);
}

// out[0] = x[0]
@compute @workgroup_size(64)
fn scopy(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  out[0] = x[0];
}

// flag[0] = 1.0 when x[0] <= threshold (bp.alpha), else untouched.
// Breakdown latch as plain f32 (no atomics needed: single lane writes once;
// CPU polls it once per PCG solve alongside the residual scalar).
@compute @workgroup_size(64)
fn sflag_le(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  if (x[0] <= bp.alpha) {
    out[0] = 1.0;
  }
}
