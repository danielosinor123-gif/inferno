//! inferno-gpu — the ggml compute bridge (G1, docs/G1-GPU.md).
//!
//! Principle: borrow the KERNELS, keep OUR architecture. This crate exposes
//! just enough of ggml's classic C API for inferno's `Engine` trait to run
//! individual ops on it — the batcher, paged KV, registry, and every gate
//! stay in inferno-core/inferno-model untouched.
//!
//! First milestone (this file): prove the bridge numerically — a mat-vec
//! (`y = W x`) computed through ggml must match the scalar reference in
//! inferno-model's `gemv` within the same tolerance the reference tests use.

use std::os::raw::c_void;

// ---- FFI (classic ggml API; stable across the pinned range) ----

#[allow(non_camel_case_types)]
pub type ggml_type = u32;

pub const GGML_TYPE_F32: ggml_type = 0;

#[repr(C)]
struct ggml_init_params {
    mem_size: usize,
    mem_buffer: *mut c_void,
    no_alloc: bool,
}

#[repr(C)]
pub struct ggml_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_tensor {
    _private: [u8; 0],
}

#[link(name = "ggml")]
extern "C" {
    fn ggml_init(params: ggml_init_params) -> *mut ggml_context;
    fn ggml_free(ctx: *mut ggml_context);
    fn ggml_new_tensor_2d(
        ctx: *mut ggml_context,
        type_: ggml_type,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;
    fn ggml_mul_mat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    fn ggml_new_f32(ctx: *mut ggml_context, value: f32) -> *mut ggml_tensor;
    fn ggml_set_f32(tensor: *mut ggml_tensor, value: f32);
    fn ggml_nelements(tensor: *const ggml_tensor) -> usize;
    fn ggml_nbytes(tensor: *const ggml_tensor) -> usize;
    fn ggml_get_data_f32(tensor: *const ggml_tensor) -> *const f32;
    fn ggml_graph_compute_with_ctx(ctx: *mut ggml_context, graph: *mut ggml_cgraph, cgraph_plan: *mut c_void) -> isize;
    fn ggml_new_graph(ctx: *mut ggml_context) -> *mut ggml_cgraph;
    fn ggml_build_forward_expand(graph: *mut ggml_cgraph, tensor: *mut ggml_tensor);
    fn ggml_graph_nbytes(graph: *const ggml_cgraph) -> usize;
    fn ggml_graph_get_tensor(graph: *const ggml_cgraph, name: *const std::os::raw::c_char) -> *mut ggml_tensor;
}

#[repr(C)]
struct ggml_cgraph {
    _private: [u8; 0],
}

// ---- Safe surface ----

/// One ggml computation arena. Contexts are cheap; one per op keeps this
/// boring and obviously correct.
pub struct GgmlCtx {
    raw: *mut ggml_context,
}

// The raw context is only used behind &mut self; ggml's classic API is
// single-threaded per context.
unsafe impl Send for GgmlCtx {}

impl GgmlCtx {
    /// A context sized for this many bytes of tensors + graph work area.
    pub fn new(bytes: usize) -> Result<Self, String> {
        let params = ggml_init_params {
            mem_size: bytes,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        };
        let raw = unsafe { ggml_init(params) };
        if raw.is_null() {
            return Err("ggml_init failed".into());
        }
        Ok(Self { raw })
    }

    /// Create an f32 tensor with ne = {cols, rows} (ggml convention:
    /// ne0 fastest). For a row-major [rows, cols] matrix this is the same
    /// bytes reinterpreted — element (i, j) sits at i + j*cols.
    fn tensor_f32_2d(&self, cols: i64, rows: i64) -> Result<*mut ggml_tensor, String> {
        let t = unsafe { ggml_new_tensor_2d(self.raw, GGML_TYPE_F32, cols, rows) };
        if t.is_null() {
            return Err("ggml_new_tensor_2d failed (arena too small?)".into());
        }
        Ok(t)
    }
}

impl Drop for GgmlCtx {
    fn drop(&mut self) {
        unsafe { ggml_free(self.raw) }
    }
}

/// Copy `src` (row-major [rows, cols]) into a ggml f32 tensor.
fn load_matrix(ctx: &GgmlCtx, src: &[f32], rows: usize, cols: usize) -> Result<*mut ggml_tensor, String> {
    debug_assert_eq!(src.len(), rows * cols);
    let t = ctx.tensor_f32_2d(cols as i64, rows as i64)?;
    let n = unsafe { ggml_nelements(t) };
    debug_assert_eq!(n, rows * cols);
    // ggml's classic API gives us the data pointer via ggml_get_data_f32;
    // the tensor allocates its own memory in the arena.
    let dst = unsafe { ggml_get_data_f32(t) as *mut f32 };
    if dst.is_null() {
        return Err("ggml tensor has no data pointer".into());
    }
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, rows * cols);
    }
    Ok(t)
}

/// The bridge core: `y = W x` through ggml's mul_mat, where `w` is
/// row-major [rows, cols] (our native HF layout — no transposed copies)
/// and `x` has length cols. `out` receives [rows].
///
/// Returned values are bit-identical to the scalar loop ONLY if the CPU
/// backend's accumulation matches; the gate compares within the reference
/// tolerance, not exact bits — see docs/G1-GPU.md gate 1.
pub fn gemm_naive(ctx: &GgmlCtx, w: &[f32], x: &[f32], rows: usize, out: &mut [f32]) -> Result<(), String> {
    debug_assert_eq!(w.len(), rows * x.len());
    debug_assert_eq!(out.len(), rows);
    let cols = x.len();
    let wt = load_matrix(ctx, w, rows, cols)?;
    let xt = {
        // x as a [cols, 1] tensor
        let t = ctx.tensor_f32_2d(cols as i64, 1)?;
        let dst = unsafe { ggml_get_data_f32(t) as *mut f32 };
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), dst, cols) };
        t
    };
    let result = unsafe { ggml_mul_mat(ctx.raw, wt, xt) };
    if result.is_null() {
        return Err("ggml_mul_mat returned null".into());
    }
    // Build + run the graph
    let graph = unsafe { ggml_new_graph(ctx.raw) };
    if graph.is_null() {
        return Err("ggml_new_graph failed".into());
    }
    unsafe { ggml_build_forward_expand(graph, result) };
    let rc = unsafe { ggml_graph_compute_with_ctx(ctx.raw, graph, std::ptr::null_mut()) };
    if rc != 0 {
        return Err(format!("ggml_graph_compute_with_ctx failed: {rc}"));
    }
    let src = unsafe { ggml_get_data_f32(result) };
    if src.is_null() {
        return Err("result tensor has no data".into());
    }
    unsafe {
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), rows);
    }
    let _ = ggml_new_f32; // referenced for API completeness; used by later slices
    Ok(())
}

/// Bytes an arena needs for a [rows, cols] x [cols] gemm (tensors + graph
/// overhead; ggml asks for ~10x the tensor bytes in practice).
pub fn arena_bytes_for(rows: usize, cols: usize) -> usize {
    let tensors = rows * cols * 4 + cols * 4 + rows * 4;
    tensors * 4 + 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference: the scalar row-dot from inferno-model's gemv (kept
    /// local so this crate's gate has no dependency arrow back).
    fn scalar_gemv(w: &[f32], x: &[f32], rows: usize, out: &mut [f32]) {
        let k = x.len();
        for j in 0..rows {
            let row = &w[j * k..j * k + k];
            let mut s = 0.0f32;
            for i in 0..k {
                s += row[i] * x[i];
            }
            out[j] = s;
        }
    }

    fn prng(state: &mut u64) -> f32 {
        let mut x = *state | 1;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        ((x >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }

    #[test]
    fn ggml_mul_mat_matches_scalar_gemv() {
        // Small enough to be hermetic, big enough that a wiring bug cannot
        // accidentally pass (the same philosophy as the reference test).
        let (rows, cols) = (11usize, 7usize);
        let mut st = 0xdead_beefu64;
        let w: Vec<f32> = (0..rows * cols).map(|_| prng(&mut st)).collect();
        let x: Vec<f32> = (0..cols).map(|_| prng(&mut st)).collect();

        let ctx = GgmlCtx::new(arena_bytes_for(rows, cols)).expect("ctx");
        let mut got = vec![0.0f32; rows];
        gemm_naive(&ctx, &w, &x, rows, &mut got).expect("ggml gemm");
        let mut want = vec![0.0f32; rows];
        scalar_gemv(&w, &x, rows, &mut want);

        // Same tolerance family as the engine's reference test (1e-4
        // relative): the CPU backend's FMA ordering differs from the
        // scalar loop; bit-identity is NOT this gate's claim (docs/G1-GPU.md).
        for (i, (g, wv)) in got.iter().zip(&want).enumerate() {
            let rel = (g - wv).abs() / (1.0 + wv.abs());
            assert!(rel < 1e-4, "row {i}: ggml {g} vs scalar {wv} (rel {rel})");
        }
    }
}
