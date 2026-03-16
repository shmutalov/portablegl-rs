PortableGL-rs
=============

[![crates.io](https://img.shields.io/crates/v/portablegl.svg)](https://crates.io/crates/portablegl)
[![CI](https://github.com/shmutalov/portablegl-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/shmutalov/portablegl-rs/actions/workflows/ci.yml)
[![license](https://img.shields.io/crates/l/portablegl.svg)](LICENSE)

***"Because of the nature of Moore's law, anything that an extremely clever graphics programmer can do at one point can be replicated by a merely competent programmer some number of years later."*** -John Carmack

PortableGL-rs is a Rust port of [PortableGL](https://github.com/rswinkle/PortableGL), an implementation of OpenGL 3.x core
(mostly; see [GL Version](#gl-version)) as a pure software renderer. No GPU required.

The original is a ~13,000 line single-header C99 library. This port rewrites it in idiomatic Rust while maintaining
functional equivalence, and can also serve as a **drop-in C library replacement** via an optional FFI layer.

It can be used with anything that takes a 32 or 16 bit framebuffer/texture as input in any format
(including just writing images to disk or blitting raw pixels to the screen via SDL2, win32, X11, etc.).

It supports arbitrary 32- and 16-bit color buffer formats (selected at compile time via Cargo features)
with several common ones ready to use out of the box.

Goals (inherited from the original project, roughly in order of priority):

1. Portability
2. Matching the OpenGL API within reason
3. Ease of Use
4. Straightforward code
5. Speed

Like the original, shaders are native function pointers (Rust `fn` / C function pointers) rather than GLSL strings.
Uniforms are passed as a raw `*mut c_void` pointer to a user-defined struct.

Features
========

- Pure software OpenGL 3.x core profile renderer (no GPU, no platform dependencies)
- Vertex processing with configurable attributes and instancing
- Triangle rasterization with barycentric coordinates
- Sutherland-Hodgman triangle clipping against 6 frustum planes
- Cohen-Sutherland parametric line clipping
- Perspective-correct, noperspective, and flat interpolation
- Depth testing, stencil testing, scissor testing
- Alpha blending with separate RGB/Alpha equations
- Face culling, polygon modes (fill/line/point)
- 1D/2D/3D textures, cubemaps, texture arrays, rectangle textures
- Nearest and linear (bilinear/trilinear) texture filtering
- Texture wrap modes: repeat, clamp to edge, clamp to border, mirrored repeat
- Bresenham and Wu's anti-aliased line drawing
- Point sprites with gl_PointCoord
- Polygon offset (depth bias)
- Logic operations
- Monomorphized shader dispatch via `FragmentShader` trait (LLVM inlines shaders into pixel loops)
- `no_std` support (with `alloc`)
- C FFI layer for drop-in replacement of the original C library

Differences from the C Version
==============================

| Aspect | C (PortableGL) | Rust (PortableGL-rs) |
|--------|---------------|---------------------|
| Language | C99, single-header library | Rust 2021 edition, multi-module crate |
| API style | Free functions with global `glContext*` | Methods on `GlContext` struct |
| Shaders | C function pointers | `unsafe extern "C" fn` pointers (compatible with both Rust and C callers) |
| Memory | Manual (`malloc`/`free`, `cvector`) | `Vec<T>`, RAII, ownership |
| Pixel format | Preprocessor macros | Cargo feature flags |
| Build system | Makefile / single header include | `cargo build` |
| `no_std` | Overridable alloc via macros; only requires `math.h`/`stdint.h` | Supported via `no_std` feature + `alloc` |
| C FFI | Native | Optional via `ffi` feature flag |
| Thread safety | Global mutable state | Owned `GlContext` per thread (FFI uses global for C compat) |

Getting Started
===============

### As a Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
portablegl = "0.8"
```

Basic usage:

```rust
use portablegl::gl_context::GlContext;
use portablegl::gl_types::*;
use portablegl::math::*;
use core::ffi::c_void;

// Define a vertex shader
unsafe extern "C" fn my_vs(
    vs_output: *mut f32,
    vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins,
    uniforms: *mut c_void,
) {
    (*builtins).gl_Position = *vertex_attribs;
}

// Define a fragment shader
unsafe extern "C" fn my_fs(
    fs_input: *mut f32,
    builtins: *mut ShaderBuiltins,
    uniforms: *mut c_void,
) {
    (*builtins).gl_FragColor = Vec4::new(1.0, 0.0, 0.0, 1.0);
}

fn main() {
    let mut ctx = GlContext::new();
    let pixels = ctx.init(640, 480);

    // Create and use a shader program
    let program = ctx.pgl_create_program(my_vs, my_fs, 0, &[], false);
    ctx.gl_use_program(program);

    // Set up geometry, draw, etc.
    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // pixels Vec<u32> contains the rendered framebuffer
}
```

### Monomorphized Shaders (Advanced)

For maximum performance, implement the `FragmentShader` trait on a concrete type.
LLVM inlines the shader body directly into the rasterization loop, eliminating
per-pixel indirect call overhead:

```rust
use portablegl::gl_types::*;
use portablegl::math::*;

struct RedShader;

impl FragmentShader for RedShader {
    #[inline(always)]
    unsafe fn shade(&self, _fs_input: *mut f32, builtins: *mut ShaderBuiltins) {
        (*builtins).gl_FragColor = Vec4::new(1.0, 0.0, 0.0, 1.0);
    }
}

// Use instead of gl_draw_arrays for triangle modes:
ctx.gl_draw_arrays_with_fs(GL_TRIANGLES, 0, count, &RedShader);
```

This is a non-breaking opt-in — the standard `gl_draw_arrays` API works unchanged.

### As a C Drop-in Replacement

Build with the `ffi` feature to produce a shared/static library:

```sh
cargo build --features ffi --release
```

This produces:
- `portablegl.dll` / `libportablegl.so` (shared library)
- `libportablegl.a` (static library)

Link against it from C/C++ and use the same API as the original PortableGL:

```c
#include <stdint.h>
// Use the same function names as PortableGL
extern GLboolean init_glContext(glContext* c, uint32_t** back, int w, int h);
extern void set_glContext(glContext* c);
extern void glClearColor(float r, float g, float b, float a);
extern void glClear(unsigned int mask);
// ... etc.
```

### For `no_std` Environments

```toml
[dependencies]
portablegl = { version = "0.8", features = ["no_std"] }
```

Requires an allocator (`alloc` crate). Uses `libm` for float math. Works on embedded targets,
custom OS kernels, WASM without `std`, etc.

### Running Examples

Three interactive examples are included (using SDL2 for windowing):

```sh
cargo run --example hello_triangle --features examples      # Solid red triangle
cargo run --example colored_triangle --features examples    # RGB interpolated triangle
cargo run --example spinning_triangle --features examples   # Rotating 3D triangle
```

Cargo Features
==============

| Feature | Description | Default |
|---------|-------------|---------|
| `abgr32` | ABGR 32-bit pixel format | Yes |
| `rgba32` | RGBA 32-bit pixel format | |
| `argb32` | ARGB 32-bit pixel format | |
| `bgra32` | BGRA 32-bit pixel format | |
| `rgb565` | RGB 16-bit pixel format | |
| `bgr565` | BGR 16-bit pixel format | |
| `d24s8` | 24-bit depth + 8-bit stencil | Yes |
| `d16` | 16-bit depth, no stencil | |
| `no_depth` | No depth buffer | |
| `no_std` | `no_std` + `alloc` support (uses `libm`) | |
| `no_stencil` | Disable stencil buffer | |
| `ffi` | Enable C FFI wrappers | |
| `disable_color_mask` | Skip per-pixel color mask (matches C `PGL_DISABLE_COLOR_MASK`) | |
| `hermite_smoothing` | Use Hermite smoothing for interpolation | |
| `better_thick_lines` | Improved thick line rendering | |
| `examples` | Build interactive examples and benchmarks (uses SDL2) | |

Project Structure
=================

```
src/
  lib.rs          - Crate root, module declarations
  math.rs         - Vec2/3/4, Mat3/4, Color, Line types and operations
  gl_types.rs     - GL type aliases, 313 constants, core structs
  gl_context.rs   - GlContext state machine struct
  gl_impl.rs      - GL API implementation (buffers, textures, VAOs, draw calls, state)
  gl_internal.rs  - Core pipeline (clipping, rasterization, fragment processing)
  gl_glsl.rs      - Texture sampling functions (1D/2D/3D, cubemap, array, fetch)
  pgl_ext.rs      - PGL extensions (clear screen, draw frame, format conversion)
  float_math.rs   - Float math compatibility layer for no_std
  ffi.rs          - C FFI wrappers (behind "ffi" feature)
```

GL Version
==========

Same as the original PortableGL: mostly OpenGL 3.x core profile, with some 4.x DSA functions
and compatibility profile features (like a default VAO) included where they come for free in
a software renderer. See the [original README](https://github.com/rswinkle/PortableGL#gl-version)
for details.

Documentation
=============

The best way to learn is to look at the original PortableGL
[examples](https://github.com/rswinkle/PortableGL/tree/master/examples/README.md)
and [demos](https://github.com/rswinkle/PortableGL/tree/master/demos/README.md),
as well as the [LearnPortableGL](https://github.com/rswinkle/LearnPortableGL) tutorials.

The official OpenGL [reference pages](https://www.khronos.org/registry/OpenGL-Refpages/gl4/)
cover 90-95% of the API usage.

Real-World Example: Craft-rs
=============================

[Craft-rs](https://github.com/shmutalov/craft-rs) is a Rust port of [Craft](https://github.com/fogleman/Craft)
(a Minecraft clone by Michael Fogleman) that uses PortableGL-rs for all rendering — no GPU required.
It demonstrates the library in a real-world application with textured voxel terrain, ambient occlusion,
day/night cycle, procedural world generation, and multiple shader programs (block, sky, text, line).

Performance Benchmarks
======================

Comparative benchmarks between the original C PortableGL and this Rust port, using identical
test scenes and SDL2 for display/timing. All 11 benchmarks are ported 1:1 from the upstream
[testing/performance_tests.cpp](https://github.com/rswinkle/PortableGL/blob/master/testing/performance_tests.cpp).

### Test System

- **CPU:** Intel Core i5-12400F (6 cores / 12 threads)
- **RAM:** 32 GB DDR4-3200
- **OS:** Windows 10 LTSC (Build 19044)
- **C compiler:** g++ 15.2.0 (MinGW-w64, `-O2 -ffp-contract=off`)
- **Rust compiler:** rustc 1.94.0 (`--release`)

### Results

| Benchmark | C (FPS) | Rust (FPS) | Ratio | Description |
|-----------|--------:|----------:|---------:|-------------|
| `points_perf` | 1085.1 | 744.4 | 0.69x | 12,000 points, size 1 |
| `pointsize_perf` | 1737.3 | 1271.1 | 0.73x | ~857 points, size 4 |
| `lines_perf` | 274.3 | 266.7 | 0.97x | 1,000 lines, width 1 |
| `lines8_perf` | 42.6 | 47.2 | **1.11x** | 1,000 lines, width 8 |
| `lines16_perf` | 21.6 | 23.9 | **1.11x** | 1,000 lines, width 16 |
| `triangles_perf` | 24.3 | 29.1 | **1.20x** | 50 random triangles |
| `tri_interp_perf` | 42.2 | 39.8 | 0.94x | 30 smooth-shaded triangles |
| `tri_clipxy_perf` | 691.3 | 701.4 | **1.01x** | 20 triangles, XY clipping |
| `tri_clipz_perf` | 162.1 | 199.3 | **1.23x** | 15 triangles, Z clipping |
| `tri_clipxyz_perf` | 270.2 | 329.9 | **1.22x** | 50 triangles, XYZ clipping |
| `blend_perf` | 328.3 | 290.2 | 0.88x | Alpha blending (9 quads) |

Values are the average of two consecutive runs. **Bold** ratios indicate Rust is faster.

### Running the Benchmarks

```sh
# Rust benchmarks (all tests)
cargo run --example perf_tests --features examples --release

# Rust benchmarks (specific test)
cargo run --example perf_tests --features examples --release -- triangles_perf

# C benchmarks (from the PortableGL repo)
cd PortableGL/testing
make -f perf_tests.make config=release
./perf_tests
```

### Analysis

The Rust port matches or exceeds C performance in 7 of 11 benchmarks, with no unsafe code in
the hot pixel paths. The remaining gaps are in point rendering and blending, where the C version
benefits from `PGL_UNSAFE` and `PGL_DISABLE_COLOR_MASK` compile-time elisions. The Rust
`disable_color_mask` feature provides an equivalent elision for the color mask operation.

Similar/Related Projects
========================

- [PortableGL](https://github.com/rswinkle/PortableGL) - The original C99 implementation this is ported from.
- [Craft-rs](https://github.com/shmutalov/craft-rs) - Minecraft clone running on PortableGL-rs (Rust port of Craft).
- [pgl](https://github.com/TotallyGamerJet/pgl) - A Go port of PortableGL.
- [TinyGL](https://bellard.org/TinyGL/) - Fabrice Bellard's OpenGL 1.x subset implementation.
- [Mesa3D](https://mesa3d.org/) - Full open source OpenGL/Vulkan implementation with software renderers.
- [SoftGLRender](https://github.com/keith2018/SoftGLRender) - OpenGL software renderer in modern C++.

Acknowledgments
================

This project was ported from C to Rust with the assistance of **Claude Opus 4.6** (Anthropic).
Claude helped translate the ~13,000-line C99 codebase into idiomatic Rust, implement the `no_std`
support layer, C FFI bindings, and port the regression test suite — achieving full functional
equivalence with the original PortableGL.

LICENSE
=======

PortableGL-rs is licensed under the MIT License (MIT).

The code used for clipping is copyright (c) Fabrice Bellard from TinyGL, also under the MIT License.
See [LICENSE](LICENSE).
