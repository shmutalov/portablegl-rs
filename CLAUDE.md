# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rust port of [PortableGL](https://github.com/rswinkle/PortableGL) — a pure software OpenGL 3.x core profile renderer. No GPU required. Shaders are native `unsafe extern "C" fn` pointers (not GLSL strings); uniforms are passed as `*mut c_void` to user-defined structs.

## Build & Test Commands

```sh
cargo build                                    # Build with default features (abgr32, d24s8)
cargo build --features ffi                     # Build with C FFI layer (produces .dll/.so/.a)
cargo rustc --features no_std --crate-type lib # Check no_std build
cargo test --release                           # Run all tests (release mode required for regression tests)
cargo test --release hello_triangle            # Run a single test by name
cargo test --release -- --nocapture            # Run tests with stdout visible
cargo run --example hello_triangle --features examples  # Run an interactive example
```

Tests are in `tests/regression.rs` (109 regression tests) with unit tests in `src/math.rs` and `src/gl_glsl.rs`. Regression tests compare rendered output pixel-for-pixel against reference PNGs in `tests/expected/`.

## Architecture

All GL state lives in `GlContext` (defined in `gl_context.rs`). The public API is methods on this struct — there is no global state in the Rust API (the FFI layer uses a global for C compatibility).

**Rendering pipeline flow:**
1. `gl_impl.rs` — Public API entry points: `gl_draw_arrays`, `gl_draw_elements`, buffer/texture/VAO management, state setting
2. `gl_internal.rs` — Internal pipeline: vertex processing → clipping (Sutherland-Hodgman for triangles, Cohen-Sutherland for lines) → rasterization → fragment processing → framebuffer write
3. `gl_glsl.rs` — Texture sampling functions called from user shaders (`texture2D`, `texelFetch`, cubemap, etc.)
4. `pgl_ext.rs` — PortableGL extensions: `pgl_clear_screen`, direct pixel/line/triangle drawing, format conversion helpers

**Pixel/depth format selection** is compile-time via Cargo features. Only one pixel format and one depth format should be active. The `gl_types.rs` file uses `#[cfg(feature = ...)]` to define `ColorType` as either `u32` or `u16` and sets up channel packing macros accordingly.

**Shader interface:** Vertex and fragment shaders receive raw pointers — `*mut f32` for interpolated varyings, `*mut Vec4` for vertex attributes, `*mut ShaderBuiltins` for builtins like `gl_Position`/`gl_FragColor`, and `*mut c_void` for the user's uniform struct. See `tests/regression.rs` for shader examples.

## Key Conventions

- The crate suppresses `non_snake_case` and `non_upper_case_globals` warnings to match OpenGL naming (e.g., `GL_TRIANGLES`, `gl_clear_color`)
- CI runs on Linux, macOS, and Windows with `cargo test --release`
- Regression test names match the original C test names for traceability
- The `ffi.rs` module mirrors the original C API function names exactly (`glClearColor`, `glDrawArrays`, etc.)
