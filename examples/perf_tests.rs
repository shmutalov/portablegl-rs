//! Performance benchmarks ported 1:1 from PortableGL's testing/performance_tests.cpp
//!
//! Run with: cargo run --example perf_tests --features examples --release
//! Run specific test: cargo run --example perf_tests --features examples --release -- points_perf
//!
//! Original C version uses PGL_ARGB32 + SDL_PIXELFORMAT_ARGB8888.
//! This Rust version uses whichever pixel format is compiled in (default: abgr32).

#![allow(non_snake_case, non_upper_case_globals, unused_assignments)]

use std::ffi::c_void;
use std::mem;

use sdl2::event::Event;
use sdl2::keyboard::Scancode;
use sdl2::pixels::PixelFormatEnum;

use portablegl::gl_types::*;
use portablegl::math::*;
use portablegl::pgl_shaders::*;
use portablegl::gl_context::GlContext;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 640;

// ---------------------------------------------------------------------------
// Simple seeded PRNG (matching C's srand/rand behavior for benchmarks)
// ---------------------------------------------------------------------------

struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn rand(&mut self) -> u32 {
        // MSVC-style LCG (the C tests use srand/rand)
        self.state = self.state.wrapping_mul(214013).wrapping_add(2531011);
        (self.state >> 16) & 0x7FFF
    }

    fn randf(&mut self) -> f32 {
        self.rand() as f32 / (0x7FFF as f32 + 1.0)
    }

    fn randf_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.randf()
    }

    fn random_in_unit_sphere(&mut self) -> Vec3 {
        loop {
            let p = Vec3::new(
                self.randf_range(-1.0, 1.0),
                self.randf_range(-1.0, 1.0),
                self.randf_range(-1.0, 1.0),
            );
            if p.x * p.x + p.y * p.y + p.z * p.z < 1.0 {
                return p;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: convert &[f32] to &[u8] for gl_buffer_data
// ---------------------------------------------------------------------------

fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, mem::size_of_val(data)) }
}

// ---------------------------------------------------------------------------
// SDL pixel format matching the compiled portablegl pixel format
// ---------------------------------------------------------------------------

fn sdl_pixel_format() -> PixelFormatEnum {
    #[cfg(feature = "abgr32")]
    { PixelFormatEnum::ABGR8888 }
    #[cfg(feature = "argb32")]
    { PixelFormatEnum::ARGB8888 }
    #[cfg(feature = "rgba32")]
    { PixelFormatEnum::RGBA8888 }
    #[cfg(feature = "bgra32")]
    { PixelFormatEnum::BGRA8888 }
    #[cfg(any(feature = "rgb565", feature = "bgr565"))]
    { PixelFormatEnum::RGB565 }
}

// ---------------------------------------------------------------------------
// Bytes per pixel for stride calculation
// ---------------------------------------------------------------------------

fn bytes_per_pixel() -> usize {
    #[cfg(any(feature = "rgb565", feature = "bgr565"))]
    { 2 }
    #[cfg(not(any(feature = "rgb565", feature = "bgr565")))]
    { 4 }
}

// ---------------------------------------------------------------------------
// Test definitions
// ---------------------------------------------------------------------------

struct PerfTest {
    name: &'static str,
    test_func: for<'a> fn(&mut TestEnv<'a>, i32, i32) -> f32,
    frames: i32,
    num: i32,
}

const TEST_SUITE: &[PerfTest] = &[
    PerfTest { name: "points_perf",     test_func: points_perf,      frames: 5000, num: 1  },
    PerfTest { name: "pointsize_perf",  test_func: points_perf,      frames: 5000, num: 4  },
    PerfTest { name: "lines_perf",      test_func: lines_perf,       frames: 2000, num: 1  },
    PerfTest { name: "lines8_perf",     test_func: lines_perf,       frames: 1000, num: 8  },
    PerfTest { name: "lines16_perf",    test_func: lines_perf,       frames: 250,  num: 16 },
    PerfTest { name: "triangles_perf",  test_func: tris_perf,        frames: 300,  num: 0  },
    PerfTest { name: "tri_interp_perf", test_func: tris_interp_perf, frames: 300,  num: 0  },
    PerfTest { name: "tri_clipxy_perf", test_func: tri_clipxy_perf,  frames: 4000, num: 0  },
    PerfTest { name: "tri_clipz_perf",  test_func: tri_clipz_perf,   frames: 4000, num: 0  },
    PerfTest { name: "tri_clipxyz_perf",test_func: tri_clipxyz_perf, frames: 4000, num: 0  },
    PerfTest { name: "blend_perf",      test_func: blend_test,       frames: 2000, num: 0  },
];

// ---------------------------------------------------------------------------
// Test environment — holds SDL + GL state shared across tests
// ---------------------------------------------------------------------------

struct TestEnv<'a> {
    canvas: sdl2::render::Canvas<sdl2::video::Window>,
    texture: sdl2::render::Texture<'a>,
    event_pump: sdl2::EventPump,
    timer: sdl2::TimerSubsystem,
    ctx: GlContext,
}

impl<'a> TestEnv<'a> {
    fn handle_events(&mut self) -> bool {
        for event in self.event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => std::process::exit(0),
                Event::KeyDown { scancode: Some(Scancode::Escape), .. } => {
                    std::process::exit(0);
                }
                _ => {}
            }
        }
        false
    }

    fn present_frame(&mut self) {
        let fb = self.ctx.pgl_get_back_buffer();
        let stride = fb.w as usize * bytes_per_pixel();
        self.texture.update(None, &fb.buf, stride).unwrap();
        self.canvas.copy(&self.texture, None, None).unwrap();
        self.canvas.present();
    }

    fn reinit_context(&mut self) {
        self.ctx = GlContext::new();
        self.ctx.init(WIDTH as i32, HEIGHT as i32);
    }
}

// ---------------------------------------------------------------------------
// points_perf — 12000 random points, variable point size via `num`
// ---------------------------------------------------------------------------

fn points_perf(env: &mut TestEnv, frames: i32, num: i32) -> f32 {
    let mut rng = Rng::new(10);

    const NUM_POINTS: usize = 12000;
    let mut points = Vec::with_capacity(NUM_POINTS * 3);
    for _ in 0..NUM_POINTS {
        points.push(rng.randf_range(-1.1, 1.1));
        points.push(rng.randf_range(-1.1, 1.1));
        points.push(-1.0f32);
    }

    let ctx = &mut env.ctx;

    let bufs = ctx.gl_gen_buffers(1);
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, bufs[0]).unwrap();
    ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&points), GL_STATIC_DRAW).unwrap();
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    // Using default shader 0

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);

    let mut draw_count = NUM_POINTS as i32;
    if num >= 2 {
        draw_count = draw_count / (num * num - 2);
    }
    ctx.gl_point_size(num as f32);

    let start = env.timer.ticks();
    let mut j = 0;
    while j < frames {
        if env.handle_events() {
            break;
        }
        env.ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        env.ctx.gl_draw_arrays(GL_POINTS, 0, draw_count);
        env.present_frame();
        j += 1;
    }
    let end = env.timer.ticks();

    j as f32 / ((end - start) as f32 / 1000.0)
}

// ---------------------------------------------------------------------------
// lines_perf — 1000 random line endpoints, variable line width via `num`
// ---------------------------------------------------------------------------

fn lines_perf(env: &mut TestEnv, frames: i32, num: i32) -> f32 {
    let mut rng = Rng::new(10);

    const NUM_LINE_VERTS: usize = 1000;
    let mut lines = Vec::with_capacity(NUM_LINE_VERTS * 3);
    for _ in 0..NUM_LINE_VERTS {
        lines.push(rng.randf_range(-1.0, 1.0));
        lines.push(rng.randf_range(-1.0, 1.0));
        lines.push(0.0f32);
    }

    let ctx = &mut env.ctx;

    let bufs = ctx.gl_gen_buffers(1);
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, bufs[0]).unwrap();
    ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&lines), GL_STATIC_DRAW).unwrap();
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    // Using default shader 0

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_line_width(num as f32);

    let start = env.timer.ticks();
    let mut i = 0;
    while i < frames {
        if env.handle_events() {
            break;
        }
        env.ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        env.ctx.gl_draw_arrays(GL_LINES, 0, NUM_LINE_VERTS as i32);
        env.present_frame();
        i += 1;
    }
    let end = env.timer.ticks();

    i as f32 / ((end - start) as f32 / 1000.0)
}

// ---------------------------------------------------------------------------
// tris_perf — 50 random triangles
// ---------------------------------------------------------------------------

fn tris_perf(env: &mut TestEnv, frames: i32, _num: i32) -> f32 {
    let mut rng = Rng::new(10);

    const NUM_TRIS: usize = 50;
    let mut tris = Vec::with_capacity(NUM_TRIS * 3 * 3);
    for _ in 0..NUM_TRIS * 3 {
        tris.push(rng.randf_range(-1.0, 1.0));
        tris.push(rng.randf_range(-1.0, 1.0));
        tris.push(-1.0f32);
    }

    let ctx = &mut env.ctx;

    let bufs = ctx.gl_gen_buffers(1);
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, bufs[0]).unwrap();
    ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&tris), GL_STATIC_DRAW).unwrap();
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    // Using default shader 0

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);

    let num_verts = (NUM_TRIS * 3) as i32;
    let start = env.timer.ticks();
    let mut j = 0;
    while j < frames {
        if env.handle_events() {
            break;
        }
        env.ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        env.ctx.gl_draw_arrays(GL_TRIANGLES, 0, num_verts);
        env.present_frame();
        j += 1;
    }
    let end = env.timer.ticks();

    j as f32 / ((end - start) as f32 / 1000.0)
}

// ---------------------------------------------------------------------------
// tris_interp_perf — 30 triangles with smooth-interpolated per-vertex color
// ---------------------------------------------------------------------------

unsafe extern "C" fn ti_smooth_vs(
    vs_output: *mut f32,
    vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins,
    _uniforms: *mut c_void,
) {
    *(vs_output as *mut Vec4) = *vertex_attribs.add(1); // ATTR_COLOR
    (*builtins).gl_Position = *vertex_attribs.add(0);   // ATTR_VERTEX
}

unsafe extern "C" fn ti_smooth_fs(
    fs_input: *mut f32,
    builtins: *mut ShaderBuiltins,
    _uniforms: *mut c_void,
) {
    (*builtins).gl_FragColor.x = *fs_input;
    (*builtins).gl_FragColor.y = *fs_input.add(1);
    (*builtins).gl_FragColor.z = *fs_input.add(2);
    (*builtins).gl_FragColor.w = 1.0;
}

#[repr(C)]
struct TiUniforms {
    color: Vec4,
}

fn tris_interp_perf(env: &mut TestEnv, frames: i32, _num: i32) -> f32 {
    let smooth: [GLenum; 3] = [PGL_SMOOTH; 3];

    let mut rng = Rng::new(10);

    const NUM_TRIS_INTERP: usize = 30;

    // Interleaved: pos(3) + color(3) per vertex
    let mut tris: Vec<f32> = Vec::with_capacity(NUM_TRIS_INTERP * 3 * 6);
    for _i in 0..NUM_TRIS_INTERP {
        // Vertex 0 — red
        tris.push(rng.randf_range(-1.0, 1.0));
        tris.push(rng.randf_range(-1.0, 1.0));
        tris.push(-1.0);
        tris.push(1.0); tris.push(0.0); tris.push(0.0);
        // Vertex 1 — green
        tris.push(rng.randf_range(-1.0, 1.0));
        tris.push(rng.randf_range(-1.0, 1.0));
        tris.push(-1.0);
        tris.push(0.0); tris.push(1.0); tris.push(0.0);
        // Vertex 2 — blue
        tris.push(rng.randf_range(-1.0, 1.0));
        tris.push(rng.randf_range(-1.0, 1.0));
        tris.push(-1.0);
        tris.push(0.0); tris.push(0.0); tris.push(1.0);
    }

    let ctx = &mut env.ctx;

    let bufs = ctx.gl_gen_buffers(1);
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, bufs[0]).unwrap();
    ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&tris), GL_STATIC_DRAW).unwrap();

    let stride = (6 * mem::size_of::<f32>()) as i32;
    let color_offset = (3 * mem::size_of::<f32>()) as isize;

    ctx.gl_enable_vertex_attrib_array(0); // ATTR_VERTEX
    ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, stride, 0);
    ctx.gl_enable_vertex_attrib_array(1); // ATTR_COLOR
    ctx.gl_vertex_attrib_pointer(1, 3, GL_FLOAT, false, stride, color_offset);

    let shader = ctx.pgl_create_program(ti_smooth_vs, ti_smooth_fs, 3, &smooth, false);
    ctx.gl_use_program(shader);

    let mut the_uniforms = TiUniforms {
        color: Vec4::new(0.0, 0.0, 0.0, 0.0),
    };
    ctx.pgl_set_uniform(&mut the_uniforms as *mut TiUniforms as *mut c_void);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);

    let num_verts = (NUM_TRIS_INTERP * 3) as i32;
    let start = env.timer.ticks();
    let mut j = 0;
    while j < frames {
        if env.handle_events() {
            break;
        }
        env.ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        env.ctx.gl_draw_arrays(GL_TRIANGLES, 0, num_verts);
        env.present_frame();
        j += 1;
    }
    let end = env.timer.ticks();

    j as f32 / ((end - start) as f32 / 1000.0)
}

// ---------------------------------------------------------------------------
// tri_clipxy_perf — 20 triangles crossing XY boundaries
// ---------------------------------------------------------------------------

fn tri_clipxy_perf(env: &mut TestEnv, frames: i32, _num: i32) -> f32 {
    let mut tris: Vec<f32> = Vec::new();

    let mut s: f32 = -1.0;
    let b: f32 = 0.9;
    let tw: f32 = 0.1;

    // Top/bottom clipping triangles
    for _ in 0..10 {
        tris.extend_from_slice(&[s, b, 0.0]);
        tris.extend_from_slice(&[s + 2.0 * tw, b, 0.0]);
        tris.extend_from_slice(&[s + tw, 1.2, 0.0]);

        tris.extend_from_slice(&[s + 2.0 * tw, -b, 0.0]);
        tris.extend_from_slice(&[s, -b, 0.0]);
        tris.extend_from_slice(&[s + tw, -1.2, 0.0]);

        s += 3.0 * tw;
    }

    // Left/right clipping triangles
    s = -1.0;
    for _ in 0..10 {
        tris.extend_from_slice(&[b, s + 2.0 * tw, 0.0]);
        tris.extend_from_slice(&[b, s, 0.0]);
        tris.extend_from_slice(&[1.2, s + tw, 0.0]);

        tris.extend_from_slice(&[-b, s, 0.0]);
        tris.extend_from_slice(&[-b, s + 2.0 * tw, 0.0]);
        tris.extend_from_slice(&[-1.2, s + tw, 0.0]);

        s += 3.0 * tw;
    }

    let ctx = &mut env.ctx;

    let bufs = ctx.gl_gen_buffers(1);
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, bufs[0]).unwrap();
    ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&tris), GL_STATIC_DRAW).unwrap();
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);

    let num_verts = (tris.len() / 3) as i32;
    let start = env.timer.ticks();
    let mut j = 0;
    while j < frames {
        if env.handle_events() {
            break;
        }
        env.ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        env.ctx.gl_draw_arrays(GL_TRIANGLES, 0, num_verts);
        env.present_frame();
        j += 1;
    }
    let end = env.timer.ticks();

    j as f32 / ((end - start) as f32 / 1000.0)
}

// ---------------------------------------------------------------------------
// tri_clipz_perf — 15 triangles crossing near/far Z planes
// ---------------------------------------------------------------------------

fn tri_clipz_perf(env: &mut TestEnv, frames: i32, _num: i32) -> f32 {
    let mut tris: Vec<f32> = Vec::new();

    for _ in 0..15 {
        tris.extend_from_slice(&[-0.2, 0.1, 0.0]);
        tris.extend_from_slice(&[0.2, 0.1, 0.0]);
        tris.extend_from_slice(&[0.0, 0.6, 1.2]);

        tris.extend_from_slice(&[0.2, -0.1, -1.2]);
        tris.extend_from_slice(&[-0.2, -0.1, -1.2]);
        tris.extend_from_slice(&[0.0, -0.6, 0.0]);
    }

    let ctx = &mut env.ctx;

    let bufs = ctx.gl_gen_buffers(1);
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, bufs[0]).unwrap();
    ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&tris), GL_STATIC_DRAW).unwrap();
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);

    let num_verts = (tris.len() / 3) as i32;
    let start = env.timer.ticks();
    let mut j = 0;
    while j < frames {
        if env.handle_events() {
            break;
        }
        env.ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        env.ctx.gl_draw_arrays(GL_TRIANGLES, 0, num_verts);
        env.present_frame();
        j += 1;
    }
    let end = env.timer.ticks();

    j as f32 / ((end - start) as f32 / 1000.0)
}

// ---------------------------------------------------------------------------
// tri_clipxyz_perf — 50 random triangles in [-1.5, 1.5] range (all axes)
// ---------------------------------------------------------------------------

fn tri_clipxyz_perf(env: &mut TestEnv, frames: i32, _num: i32) -> f32 {
    let mut rng = Rng::new(10);

    const NUM_TRIS: usize = 50;
    let mut tris: Vec<f32> = Vec::with_capacity(NUM_TRIS * 3 * 3 * 3);

    for _ in 0..NUM_TRIS * 3 {
        let p1x = rng.randf_range(-1.5, 1.5);
        let p1y = rng.randf_range(-1.5, 1.5);
        let p1z = rng.randf_range(-1.5, 1.5);

        let sphere2 = rng.random_in_unit_sphere();
        let p2x = p1x + sphere2.x / 2.0;
        let p2y = p1y + sphere2.y / 2.0;
        let p2z = p1z + sphere2.z / 2.0;

        let sphere3 = rng.random_in_unit_sphere();
        let p3x = p1x + sphere3.x / 2.0;
        let p3y = p1y + sphere3.y / 2.0;
        let p3z = p1z + sphere3.z / 2.0;

        tris.extend_from_slice(&[p1x, p1y, p1z]);
        tris.extend_from_slice(&[p2x, p2y, p2z]);
        tris.extend_from_slice(&[p3x, p3y, p3z]);
    }

    let ctx = &mut env.ctx;

    let bufs = ctx.gl_gen_buffers(1);
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, bufs[0]).unwrap();
    ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&tris), GL_STATIC_DRAW).unwrap();
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);

    let num_verts = (tris.len() / 3) as i32;
    let start = env.timer.ticks();
    let mut j = 0;
    while j < frames {
        if env.handle_events() {
            break;
        }
        env.ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        env.ctx.gl_draw_arrays(GL_TRIANGLES, 0, num_verts);
        env.present_frame();
        j += 1;
    }
    let end = env.timer.ticks();

    j as f32 / ((end - start) as f32 / 1000.0)
}

// ---------------------------------------------------------------------------
// blend_test — opaque quads + semi-transparent overlays with alpha blending
// ---------------------------------------------------------------------------

fn blend_test(env: &mut TestEnv, frames: i32, _num: i32) -> f32 {
    #[rustfmt::skip]
    let points: [f32; 108] = [
        // 4 opaque quads (GL_TRIANGLE_STRIP, 4 verts each)
        -0.75,  0.75, 0.0,
        -0.75,  0.25, 0.0,
        -0.25,  0.75, 0.0,
        -0.25,  0.25, 0.0,

         0.25,  0.75, 0.0,
         0.25,  0.25, 0.0,
         0.75,  0.75, 0.0,
         0.75,  0.25, 0.0,

        -0.75, -0.25, 0.0,
        -0.75, -0.75, 0.0,
        -0.25, -0.25, 0.0,
        -0.25, -0.75, 0.0,

         0.25, -0.25, 0.0,
         0.25, -0.75, 0.0,
         0.75, -0.25, 0.0,
         0.75, -0.75, 0.0,

        // 5 semi-transparent quads
        // mix with white
        -0.15,  0.15, -0.1,
        -0.15, -0.15, -0.1,
         0.15,  0.15, -0.1,
         0.15, -0.15, -0.1,

        // mix with red
        -0.40,  0.65, -0.1,
        -0.40,  0.35, -0.1,
        -0.10,  0.65, -0.1,
        -0.10,  0.35, -0.1,

        // mix with green
         0.10,  0.65, -0.1,
         0.10,  0.35, -0.1,
         0.40,  0.65, -0.1,
         0.40,  0.35, -0.1,

        // mix with blue
        -0.40, -0.35, -0.1,
        -0.40, -0.65, -0.1,
        -0.10, -0.35, -0.1,
        -0.10, -0.65, -0.1,

        // mix with black
         0.10, -0.35, -0.1,
         0.10, -0.65, -0.1,
         0.40, -0.35, -0.1,
         0.40, -0.65, -0.1,
    ];

    let ctx = &mut env.ctx;

    let bufs = ctx.gl_gen_buffers(1);
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, bufs[0]).unwrap();
    ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&points), GL_STATIC_DRAW).unwrap();
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    let std_shaders = pgl_init_std_shaders(ctx);
    ctx.gl_use_program(std_shaders[PGL_SHADER_IDENTITY]);

    let mut the_uniforms = PglUniforms::default();
    ctx.pgl_set_uniform(&mut the_uniforms as *mut PglUniforms as *mut c_void);

    ctx.gl_clear_color(1.0, 1.0, 1.0, 1.0);

    let red   = Vec4::new(1.0, 0.0, 0.0, 1.0);
    let green = Vec4::new(0.0, 1.0, 0.0, 1.0);
    let blue  = Vec4::new(0.0, 0.0, 1.0, 1.0);
    let black = Vec4::new(0.0, 0.0, 0.0, 1.0);

    let start = env.timer.ticks();
    let mut i = 0;
    while i < frames {
        if env.handle_events() {
            break;
        }
        env.ctx.gl_clear(GL_COLOR_BUFFER_BIT);

        // 4 opaque quads
        the_uniforms.color = red;
        env.ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 0, 4);
        the_uniforms.color = green;
        env.ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 4, 4);
        the_uniforms.color = blue;
        env.ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 8, 4);
        the_uniforms.color = black;
        env.ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 12, 4);

        // 5 blended quads
        env.ctx.gl_enable(GL_BLEND);
        env.ctx.gl_blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        the_uniforms.color = Vec4::new(1.0, 0.0, 0.0, 0.5);
        env.ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 16, 4);
        env.ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 20, 4);
        env.ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 24, 4);
        env.ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 28, 4);
        env.ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 32, 4);
        env.ctx.gl_disable(GL_BLEND);

        env.present_frame();
        i += 1;
    }
    let end = env.timer.ticks();

    i as f32 / ((end - start) as f32 / 1000.0)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let sdl_context = sdl2::init().expect("Failed to init SDL2");
    let video = sdl_context.video().expect("Failed to init SDL2 video");
    let timer = sdl_context.timer().expect("Failed to get timer");

    let window = video
        .window("performance_tests", WIDTH, HEIGHT)
        .position_centered()
        .resizable()
        .build()
        .expect("Failed to create window");

    let canvas = window
        .into_canvas()
        .software()
        .build()
        .expect("Failed to create canvas");

    let texture_creator = canvas.texture_creator();
    let texture = texture_creator
        .create_texture_streaming(sdl_pixel_format(), WIDTH, HEIGHT)
        .expect("Failed to create texture");

    let event_pump = sdl_context.event_pump().expect("Failed to get event pump");

    let mut env = TestEnv {
        canvas,
        texture,
        event_pump,
        timer,
        ctx: GlContext::new(),
    };

    let args: Vec<String> = std::env::args().collect();

    if args.len() <= 1 {
        println!("Running {} tests...", TEST_SUITE.len());
        for test in TEST_SUITE {
            run_test(&mut env, test);
        }
    } else {
        let total = args.len() - 1;
        println!("Attempting to run {} tests...", total);
        for arg in &args[1..] {
            if let Some(test) = TEST_SUITE.iter().find(|t| t.name == arg.as_str()) {
                run_test(&mut env, test);
            } else {
                println!("Error: could not find test '{}', skipping", arg);
            }
        }
    }
}

fn run_test(env: &mut TestEnv, test: &PerfTest) {
    env.reinit_context();
    let fps = (test.test_func)(env, test.frames, test.num);
    println!("{}: {:.3} FPS", test.name, fps);
}
