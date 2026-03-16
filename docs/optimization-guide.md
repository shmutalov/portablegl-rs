# Performance Optimization Guide for PortableGL-rs

This document is a deep analysis of performance gaps between the C PortableGL and its
Rust port, with concrete optimization strategies ranked by expected impact.

## Current Benchmark Status

Tested on Intel i5-12400F, 32 GB DDR4-3200, Windows 10 LTSC.
C compiled with g++ 15.2 `-O2`, Rust with rustc 1.94.0 `--release`.

| Benchmark | C (FPS) | Rust (FPS) | Ratio | Bottleneck |
|-----------|--------:|----------:|------:|------------|
| `points_perf` | 1085 | 647 | 0.60x | Per-pixel allocation + fn pointer |
| `pointsize_perf` | 1737 | 901 | 0.52x | Same, amplified by point size logic |
| `lines_perf` | 274 | 224 | 0.81x | Per-fragment allocation |
| `lines8_perf` | 43 | 39 | 0.92x | Fill-rate bound (width 8) |
| `lines16_perf` | 22 | 20 | 0.94x | Fill-rate bound (width 16) |
| `triangles_perf` | 24 | 29 | **1.21x** | Fill-rate bound, Rust wins |
| `tri_interp_perf` | 42 | 18 | 0.43x | Interpolation + per-pixel alloc |
| `tri_clipxy_perf` | 691 | 551 | 0.80x | Clipping overhead + vertex cloning |
| `tri_clipz_perf` | 162 | 165 | **1.02x** | Parity |
| `tri_clipxyz_perf` | 270 | 292 | **1.08x** | Rust wins |
| `blend_perf` | 328 | 245 | 0.75x | Blend pixel unpack/repack |

Key observation: tests where Rust loses badly all share the same root causes —
**per-pixel heap allocation** and **function pointer call overhead**.

---

## Critical Finding: Per-Pixel Vec Allocation

The single biggest performance problem in the Rust port. In every rasterization path
(points, lines, triangles), the fragment shader input is allocated on the heap **for
every single pixel**:

```rust
// gl_internal.rs — appears in draw_triangle_fill, draw_point, draw_line_*
let mut fs_input_copy: Vec<f32> = c.fs_input[..vs_output_size].to_vec();
unsafe { (fs)(fs_input_copy.as_mut_ptr(), &mut c.builtins, uniform); }
```

This calls `malloc` and `free` potentially millions of times per frame. The C version
simply passes `c->fs_input` directly — no copy, no allocation.

### Why it exists

The Rust port copies `fs_input` because the fragment shader receives a mutable pointer.
If the shader modifies the input, the shared `c.fs_input` buffer would be corrupted for
subsequent use. The C version doesn't have this concern because it either doesn't care
about the mutation or re-computes `fs_input` from scratch for each pixel anyway.

### Fix (Priority 1 — highest impact)

**Option A: Pass `c.fs_input` directly (zero-copy).** Since `fs_input` is recomputed
via `setup_fs_input_*()` before every fragment shader call, the shader is free to
mutate it — we'll overwrite it on the next pixel anyway.

```rust
// Replace:
let mut fs_input_copy: Vec<f32> = c.fs_input[..vs_output_size].to_vec();
unsafe { (fs)(fs_input_copy.as_mut_ptr(), ...) };

// With:
unsafe { (fs)(c.fs_input.as_mut_ptr(), ...) };
```

**Option B: Pre-allocated scratch buffer.** If there's a reason `c.fs_input` can't be
passed directly, add a `fs_input_scratch: Vec<f32>` field to `GlContext` that's
allocated once during `init()` and reused:

```rust
// In init():
self.fs_input_scratch = vec![0.0; GL_MAX_VERTEX_OUTPUT_COMPONENTS];

// In draw loop:
c.fs_input_scratch[..vs_output_size].copy_from_slice(&c.fs_input[..vs_output_size]);
unsafe { (fs)(c.fs_input_scratch.as_mut_ptr(), ...) };
```

**Expected impact:** Eliminates millions of malloc/free per frame. Should bring
`points_perf`, `tri_interp_perf`, and all line tests to near-parity or better
immediately. This single fix likely accounts for 30-60% of the total performance gap.

---

## Optimization 1: Eliminate Vertex Cloning in Rasterization

### Problem

In `run_pipeline()` (gl_internal.rs), vertices are cloned for every primitive:

```rust
// Lines 826, 834-835, 844-845, 861-862
let vert = c.glverts[i].clone();
```

`GlVertex` contains `vs_out: Vec<f32>`, so every clone heap-allocates. For 1000 lines
(2000 vertices), that's 2000 unnecessary allocations per frame.

### Fix

Pass vertices by reference or index instead of cloning:

```rust
// Instead of cloning, pass indices into c.glverts[]
draw_triangle(c, v0_idx, v1_idx, v2_idx, provoke);
```

Or use a fixed-size array instead of Vec for `vs_out`:

```rust
pub struct GlVertex {
    pub clip_space: Vec4,
    pub screen_space: Vec4,
    pub clip_code: i32,
    pub edge_flag: i32,
    pub vs_out: [f32; GL_MAX_VERTEX_OUTPUT_COMPONENTS], // Stack-allocated, Copy
}
```

This makes `GlVertex` implement `Copy`, so clones become trivial memcpy with no heap.

**Expected impact:** Significant for clipping-heavy tests and high-vertex-count draws.

---

## Optimization 2: Eliminate Allocations in Clipping

### Problem

`draw_triangle_clip()` and `draw_line_clip()` allocate new `Vec<f32>` for interpolated
vertex outputs on every clip operation:

```rust
// draw_triangle_clip, interpolate_vertex
let mut vs_out = vec![0.0f32; vs_size];  // heap allocation per clip vertex
```

Sutherland-Hodgman clipping can produce up to ~10 output vertices per triangle, each
requiring this allocation.

### Fix

Use stack arrays (the max output size is bounded by `GL_MAX_VERTEX_OUTPUT_COMPONENTS`):

```rust
let mut vs_out = [0.0f32; GL_MAX_VERTEX_OUTPUT_COMPONENTS];
```

Or use `arrayvec::ArrayVec` if dynamic sizing is needed within a fixed max.

**Expected impact:** Noticeable on clip-heavy tests (`tri_clipxy_perf`, `tri_clipxyz_perf`).

---

## Optimization 3: VAO Attribute Array Clone

### Problem

```rust
// vertex_stage(), gl_internal.rs line 683
let v = c.vertex_arrays[vao_idx].vertex_attribs.clone();
```

Clones the entire 8-element vertex attribute descriptor array on every draw call.

### Fix

Use a reference:

```rust
let v = &c.vertex_arrays[vao_idx].vertex_attribs;
```

**Expected impact:** Small but free improvement on every draw call.

---

## Optimization 4: Build Configuration

These are zero-effort changes that affect all benchmarks:

### Cargo.toml release profile

```toml
[profile.release]
lto = "fat"          # Full link-time optimization across all crates
codegen-units = 1    # Single codegen unit for maximum inlining
opt-level = 3        # Maximum optimization (default for release, but explicit)
```

### Target CPU tuning (.cargo/config.toml)

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

Unlocks AVX2, FMA, BMI2 on the i5-12400F, enabling wider auto-vectorization
(256-bit instead of 128-bit).

### Profile-Guided Optimization (PGO)

```sh
cargo install cargo-pgo
cargo pgo build
# Run the perf_tests benchmark as training workload
./target/release/examples/perf_tests
cargo pgo optimize build
```

PGO is especially effective for code with many branches (clipping, state dispatch).

**Expected combined impact:** 10-30% improvement across all benchmarks from build
config alone. LTO + codegen-units=1 is typically 15-20% for cross-module code.

---

## Optimization 5: Bounds Check Elimination in Inner Loops

### Problem

Every pixel read/write in the framebuffer and depth buffer has a runtime bounds check:

```rust
fn read_zbuf(c: &GlContext, idx: usize) -> u32 {
    let data = &c.zbuf.buf;
    let off = idx * 4;
    if off + 4 > data.len() { return 0; }  // per-pixel bounds check
    u32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]])
}
```

The rasterization loop already clamps `x` and `y` to the viewport, so these checks
are always true but the compiler can't prove it.

### Fix

**Option A: Assert before the loop, use unchecked access inside:**

```rust
let buf_len = c.back_buffer.buf.len();
assert!(max_y * width + max_x < buf_len / 4);

for y in min_y..max_y {
    for x in min_x..max_x {
        let idx = y * width + x;
        // SAFETY: bounds verified by assert above
        unsafe {
            let off = idx * 4;
            let ptr = buf.as_ptr().add(off);
            u32::from_le_bytes(*(ptr as *const [u8; 4]))
        }
    }
}
```

**Option B: Pre-slice the row buffer:**

```rust
for y in min_y..max_y {
    let row = &mut buf[y * width * 4 .. (y + 1) * width * 4];
    for x in min_x..max_x {
        let off = x * 4;
        // LLVM can prove this is in bounds from the slice length
        let pixel = u32::from_le_bytes([row[off], row[off+1], row[off+2], row[off+3]]);
    }
}
```

**Option C: Use `u32` slice instead of `u8`.**

Store the framebuffer as `Vec<u32>` instead of `Vec<u8>`. This eliminates the `*4`
offset arithmetic and the `from_le_bytes` assembly entirely:

```rust
// Current: buf is Vec<u8>, read 4 bytes, assemble u32
let pixel = u32::from_le_bytes([buf[off], buf[off+1], buf[off+2], buf[off+3]]);

// Proposed: buf is Vec<u32>, single read
let pixel = buf[idx];
```

This also makes bounds checks cheaper (one comparison instead of `off+4 > len`).

**Expected impact:** Meaningful for fill-rate-bound tests. Bounds checks themselves
are cheap (compare+branch), but they also block LLVM's auto-vectorizer.

---

## Optimization 6: Perspective-Correct Interpolation

### Problem

The C version pre-divides all varyings by `w` once before the rasterization loop:

```c
// C version: done ONCE before the pixel loop
for (i = 0; i < vs_output_size; i++) {
    perspective[i] = v0_out[i] / p0.w;
    perspective[n+i] = v1_out[i] / p1.w;
    perspective[2*n+i] = v2_out[i] / p2.w;
}
```

Then in the per-pixel loop, it only does multiply-accumulate (no division per varying):

```c
// C version: per-pixel (only multiplies)
tmp = alpha * perspective[i] + beta * perspective[n+i] + gamma * perspective[2*n+i];
fs_input[i] = tmp / w_interpolated;  // ONE division for all varyings
```

The Rust version currently performs the division **per varying per pixel**:

```rust
// Rust version: per-pixel, per-varying (division inside the loop)
let val = alpha * v0_out[j] / w0 + beta * v1_out[j] / w1 + gamma * v2_out[j] / w2;
c.fs_input[j] = val / w_interp;
```

### Fix

Pre-compute `v_out[j] / w` for all three vertices before entering the pixel loop,
matching the C version's approach:

```rust
// Before the pixel loop:
let mut persp = [0.0f32; MAX_VS_OUTPUT * 3];
for j in 0..vs_output_size {
    persp[j] = v0_out[j] * inv_w0;
    persp[vs_output_size + j] = v1_out[j] * inv_w1;
    persp[2 * vs_output_size + j] = v2_out[j] * inv_w2;
}

// In the pixel loop:
for j in 0..vs_output_size {
    let val = alpha * persp[j] + beta * persp[vs_output_size + j]
            + gamma * persp[2 * vs_output_size + j];
    c.fs_input[j] = val / w_interp;
}
```

This replaces `3 * vs_output_size` divisions per pixel with `3 * vs_output_size`
multiplications (which are ~5x faster than divisions on modern CPUs).

**Expected impact:** Major improvement on `tri_interp_perf` (currently 0.43x). This
test exists specifically to measure interpolation throughput. Should roughly double
its performance.

---

## Optimization 7: Match C Compile-Time Elisions

### PGL_UNSAFE equivalent

The C benchmarks define `PGL_UNSAFE`, which eliminates all GL error checking
(`PGL_SET_ERR`, `PGL_ERR`, `PGL_ERR_RET_VAL` become no-ops). The Rust port should
have an equivalent feature:

```toml
# Cargo.toml
[features]
unsafe_mode = []  # Already exists but may not elide all checks
```

Audit every `Err(GL_INVALID_*)` return in `gl_impl.rs` and wrap them:

```rust
#[cfg(not(feature = "unsafe_mode"))]
if !self.validate_something() {
    return Err(GL_INVALID_OPERATION);
}
```

### PGL_DISABLE_COLOR_MASK equivalent

The C benchmarks define `PGL_DISABLE_COLOR_MASK`, which eliminates the per-pixel
color mask operation:

```c
// Without PGL_DISABLE_COLOR_MASK:
src = (src & c->color_mask) | (dst & ~c->color_mask);

// With PGL_DISABLE_COLOR_MASK:
// (line is simply not present)
```

Add a Cargo feature:

```toml
[features]
disable_color_mask = []
```

And in `draw_pixel()`:

```rust
#[cfg(not(feature = "disable_color_mask"))]
{
    let dst = read_backbuf_pixel(c, idx);
    pixel_val = (pixel_val & c.color_mask) | (dst & !c.color_mask);
}
```

**Expected impact:** 5-15% on fill-rate-bound tests. The color mask operation
requires an extra framebuffer read (cache miss) even when the mask is `0xFFFFFFFF`.

---

## Optimization 8: gl_clear Optimization

### Problem

The current `gl_clear` evaluates three boolean conditions per pixel:

```rust
for i in 0..total {
    if do_color { /* write color */ }
    if do_depth { /* write depth */ }
    if do_stencil { /* write stencil */ }
}
```

For 640x640 = 409,600 pixels, this is 1.2M unnecessary branch evaluations.

### Fix

Specialize the loop for the 8 combinations:

```rust
match (do_color, do_depth, do_stencil) {
    (true, false, false) => {
        // Fast path: just fill color buffer
        // Can use memset / slice::fill for uniform color
        for i in 0..total { write_pixel(&mut color_buf, i, color_val); }
    }
    (true, true, false) => {
        for i in 0..total {
            write_pixel(&mut color_buf, i, color_val);
            write_pixel(&mut depth_buf, i, depth_val);
        }
    }
    // ... etc
}
```

For the common `GL_COLOR_BUFFER_BIT`-only case, this can further optimize to a
`memset` / `slice::fill` if the color mask is `0xFFFFFFFF`.

**Expected impact:** Small overall, but affects every frame. The benchmarks call
`gl_clear(GL_COLOR_BUFFER_BIT)` every frame.

---

## Optimization 9: Monomorphized Shader Dispatch (Long-term)

### Problem

Fragment shaders are called through `unsafe extern "C" fn` pointers. LLVM cannot
inline through indirect calls, which means:

1. Every pixel pays the cost of an indirect branch
2. No interprocedural optimization between the shader and the rasterizer
3. The auto-vectorizer cannot combine shader logic with the rasterization loop

For a 640x640 framebuffer with 50 triangles covering ~50% of pixels, that's roughly
10 million indirect calls per frame.

### Fix (non-breaking addition)

Add a generic draw path alongside the existing function-pointer path:

```rust
pub fn draw_triangles_mono<VS, FS>(
    &mut self,
    vs: VS, fs: FS,
    mode: GLenum, first: i32, count: i32,
)
where
    VS: Fn(*mut f32, *mut Vec4, *mut ShaderBuiltins, *mut c_void),
    FS: Fn(*mut f32, *mut ShaderBuiltins, *mut c_void),
{
    // Same rasterization logic but with concrete shader types
    // LLVM inlines the closures into the pixel loop
}
```

This is a larger refactor but provides the maximum possible speedup. The C version
doesn't have this advantage — function pointers are indirect calls in C too. This
would let the Rust port **surpass** the C version on shader-heavy workloads.

**Expected impact:** 2-5x improvement on shader-bound tests like `tri_interp_perf`.
When the shader is trivial (identity), the indirect call overhead dominates.

---

## Optimization 10: SIMD Scanline Processing (Long-term)

### Opportunity

The innermost triangle rasterization loop processes pixels one at a time. Modern CPUs
can process 4 (SSE) or 8 (AVX2) pixels simultaneously:

```rust
// Current: scalar, one pixel at a time
for x in min_x..max_x {
    let alpha = l12.func(px, py) * inv_alpha;
    // ... test, interpolate, shade, write ONE pixel
}

// Proposed: SIMD, 4 pixels at a time
for x in (min_x..max_x).step_by(4) {
    let px4 = f32x4::new(x, x+1, x+2, x+3);
    let alpha4 = l12.func_x4(px4, py) * inv_alpha;
    // ... test 4 pixels, interpolate 4, shade 4, write 4
}
```

### Implementation approach

1. Use the `wide` crate (stable Rust) or `std::simd` (nightly) for portable SIMD
2. Compute edge functions for 4 pixels at once (`f32x4`)
3. Use SIMD comparison to generate a pixel mask
4. Interpolate varyings for 4 pixels in parallel
5. Call the fragment shader 4 times (or batch if monomorphized)
6. Write 4 pixels to the framebuffer with a single 128-bit store

### Where SIMD helps most

| Operation | Scalar ops/pixel | SIMD ops/4 pixels | Speedup |
|-----------|----------------:|------------------:|--------:|
| Edge function (3x) | 6 mul + 6 add | 6 mul + 6 add | 4x |
| Depth test | 1 cmp + 1 branch | 1 cmp (masked) | 4x |
| Varying interpolation | 3N mul + 2N add | 3N mul + 2N add | 4x |
| Framebuffer write | 1 store | 1 store (128-bit) | 4x |

**Expected impact:** 2-3x on fill-rate-bound tests (triangles, blend). Less on tests
dominated by setup (clipping, points).

---

## Priority Order for Implementation

Ranked by effort vs impact:

| Priority | Optimization | Effort | Expected Gain | Tests Affected |
|---------:|-------------|--------|--------------|----------------|
| **1** | Remove per-pixel Vec allocation | Small | **30-60%** | All |
| **2** | Build config (LTO, codegen-units, target-cpu) | Trivial | **10-30%** | All |
| **3** | Pre-compute perspective division | Small | **20-40%** | tri_interp |
| **4** | GlVertex vs_out as fixed array | Medium | **10-20%** | All with clipping |
| **5** | Remove vertex cloning in run_pipeline | Medium | **5-15%** | All |
| **6** | Bounds check elimination in pixel I/O | Small | **5-10%** | Fill-rate bound |
| **7** | PGL_UNSAFE / PGL_DISABLE_COLOR_MASK features | Small | **5-15%** | All |
| **8** | gl_clear specialization | Small | **2-5%** | All (every frame) |
| **9** | PGO | Small | **10-20%** | All |
| **10** | Monomorphized shader dispatch | Large | **50-200%** | Shader-heavy |
| **11** | SIMD scanline processing | Large | **100-300%** | Fill-rate bound |

**Optimizations 1-4 alone should bring every benchmark to parity or better.**
Optimizations 10-11 are long-term investments that could make the Rust port
significantly faster than the C original.

---

## Appendix A: C Version Compile-Time Flags

The C benchmark binary is compiled with these defines:

```cpp
#define PGL_UNSAFE           // Skip all GL error checking
#define PGL_PREFIX_TYPES     // Prefix types to avoid conflicts
#define PGL_DISABLE_COLOR_MASK  // Skip color masking
#define PGL_ARGB32           // ARGB pixel format
```

Other relevant C flags not used in benchmarks but available:

| Flag | Effect |
|------|--------|
| `PGL_UNSAFE` | Eliminates all `PGL_SET_ERR()` / `PGL_ERR()` calls |
| `PGL_DISABLE_COLOR_MASK` | Removes per-pixel mask-combine operation |
| `PGL_BETTER_THICK_LINES` | More correct but 15-17% slower thick lines |
| `PGL_ENABLE_CLAMP_TO_BORDER` | Adds conditionals to texture sampling |
| `PGL_DONT_CONVERT_TEXTURES` | Skips auto-conversion of texture formats |
| `PGL_NO_DEPTH_NO_STENCIL` | Eliminates depth/stencil buffers entirely |
| `PGL_D16` | 16-bit depth (half the memory of D24S8) |
| `PGL_NO_STENCIL` | Removes stencil buffer (with D16) |

## Appendix B: Key Source Locations

| File | Lines | What |
|------|-------|------|
| `gl_internal.rs` | ~2438-2607 | Triangle rasterization inner loop |
| `gl_internal.rs` | ~2073-2141 | Point drawing inner loop |
| `gl_internal.rs` | ~1502-1697 | Line drawing inner loop (thick) |
| `gl_internal.rs` | ~1326-1375 | `setup_fs_input_triangle` (interpolation) |
| `gl_internal.rs` | ~1278-1324 | `setup_fs_input_line` (interpolation) |
| `gl_internal.rs` | ~297-413 | Depth/color buffer read/write |
| `gl_internal.rs` | ~2279-2421 | Sutherland-Hodgman clipping |
| `gl_internal.rs` | ~674-778 | `vertex_stage` (attribute fetching) |
| `gl_internal.rs` | ~826-870 | `run_pipeline` (primitive dispatch, vertex cloning) |
| `gl_impl.rs` | ~1507-1562 | `gl_clear` implementation |
| `gl_impl.rs` | ~79-103 | Color packing/unpacking |
| `gl_types.rs` | ~710-730 | `GlVertex` struct (contains `vs_out: Vec<f32>`) |
| `pgl_shaders.rs` | ~89-101 | Default identity shader |

## Appendix C: Rust Build Configuration Reference

Recommended `.cargo/config.toml` for maximum performance:

```toml
[target.x86_64-pc-windows-gnu]
rustflags = ["-L", "lib", "-C", "target-cpu=native"]

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native"]
```

Recommended `Cargo.toml` release profile:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
opt-level = 3

[profile.release-with-pgo]
inherits = "release"
# Use after collecting PGO data
```
