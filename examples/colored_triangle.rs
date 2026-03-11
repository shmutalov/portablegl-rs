//! Example 2: Colored Triangle
//!
//! Renders a triangle with per-vertex RGB color interpolation (smooth shading).
//! Port of PortableGL's examples/original/ex2.c
//!
//! Run: cargo run --example colored_triangle --features examples

use core::ffi::c_void;
use minifb::{Key, Window, WindowOptions};
use portablegl::gl_context::GlContext;
use portablegl::gl_types::*;
use portablegl::math::*;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

unsafe extern "C" fn smooth_vs(
    vs_output: *mut f32,
    vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins,
    _uniforms: *mut c_void,
) {
    // Pass color (attribute 4) to fragment shader as output 0
    *(vs_output as *mut Vec4) = *vertex_attribs.add(4);
    (*builtins).gl_Position = *vertex_attribs;
}

unsafe extern "C" fn smooth_fs(
    fs_input: *mut f32,
    builtins: *mut ShaderBuiltins,
    _uniforms: *mut c_void,
) {
    (*builtins).gl_FragColor = *(fs_input as *const Vec4);
}

/// Read the PortableGL back buffer and convert ABGR32 → minifb (0x00RRGGBB),
/// flipping vertically (PortableGL is bottom-up, minifb is top-down).
fn read_framebuffer(ctx: &GlContext, display_buf: &mut [u32], w: usize, h: usize) {
    let fb = ctx.pgl_get_back_buffer();
    for y in 0..h {
        let src_row = h - 1 - y;
        for x in 0..w {
            let off = (src_row * w + x) * 4;
            let r = fb.buf[off] as u32;
            let g = fb.buf[off + 1] as u32;
            let b = fb.buf[off + 2] as u32;
            display_buf[y * w + x] = (r << 16) | (g << 8) | b;
        }
    }
}

fn main() {
    let mut window = Window::new(
        "PortableGL-rs: Colored Triangle (ESC to exit)",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .expect("Failed to create window");

    window.set_target_fps(60);

    let mut ctx = GlContext::new();
    let _pixels = ctx.init(WIDTH as i32, HEIGHT as i32);

    // Interleaved position (3) + color (3) data
    let points_n_colors: [f32; 18] = [
        -0.5, -0.5, 0.0,   1.0, 0.0, 0.0,  // Red vertex
         0.5, -0.5, 0.0,   0.0, 1.0, 0.0,  // Green vertex
         0.0,  0.5, 0.0,   0.0, 0.0, 1.0,  // Blue vertex
    ];

    // Interpolate 4 floats (vec4 color) smoothly across the triangle
    let smooth: [GLenum; 4] = [PGL_SMOOTH; 4];
    let program = ctx.pgl_create_program(smooth_vs, smooth_fs, 4, &smooth, false);
    ctx.gl_use_program(program);

    let buffers = ctx.gl_gen_buffers(1);
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, buffers[0]).unwrap();
    let data: &[u8] = unsafe {
        core::slice::from_raw_parts(
            points_n_colors.as_ptr() as *const u8,
            core::mem::size_of_val(&points_n_colors),
        )
    };
    ctx.gl_buffer_data(GL_ARRAY_BUFFER, data, GL_STATIC_DRAW).unwrap();

    // Attribute 0: position (3 floats, stride=24, offset=0)
    ctx.gl_enable_vertex_attrib_array(0);
    ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 6 * 4, 0);

    // Attribute 4: color (3 floats, stride=24, offset=12)
    ctx.gl_enable_vertex_attrib_array(4);
    ctx.gl_vertex_attrib_pointer(4, 3, GL_FLOAT, false, 6 * 4, 3 * 4);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);

    let mut display_buf = vec![0u32; WIDTH * HEIGHT];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

        read_framebuffer(&ctx, &mut display_buf, WIDTH, HEIGHT);
        window.update_with_buffer(&display_buf, WIDTH, HEIGHT).unwrap();
    }
}
