//! Example 1: Hello Triangle
//!
//! Renders a solid red triangle using a uniform color fragment shader.
//! Port of PortableGL's examples/original/ex1.c
//!
//! Run: cargo run --example hello_triangle --features examples

use core::ffi::c_void;
use minifb::{Key, Window, WindowOptions};
use portablegl::gl_context::GlContext;
use portablegl::gl_types::*;
use portablegl::math::*;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

#[repr(C)]
struct MyUniforms {
    v_color: Vec4,
}

unsafe extern "C" fn identity_vs(
    _vs_output: *mut f32,
    vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins,
    _uniforms: *mut c_void,
) {
    (*builtins).gl_Position = *vertex_attribs;
}

unsafe extern "C" fn uniform_color_fs(
    _fs_input: *mut f32,
    builtins: *mut ShaderBuiltins,
    uniforms: *mut c_void,
) {
    (*builtins).gl_FragColor = (*(uniforms as *const MyUniforms)).v_color;
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
        "PortableGL-rs: Hello Triangle (ESC to exit)",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .expect("Failed to create window");

    window.set_target_fps(60);

    let mut ctx = GlContext::new();
    let _pixels = ctx.init(WIDTH as i32, HEIGHT as i32);

    // Triangle vertices
    let points: [f32; 9] = [
        -0.5, -0.5, 0.0,
         0.5, -0.5, 0.0,
         0.0,  0.5, 0.0,
    ];

    let program = ctx.pgl_create_program(identity_vs, uniform_color_fs, 0, &[], false);
    ctx.gl_use_program(program);

    let mut the_uniforms = MyUniforms {
        v_color: Vec4::new(1.0, 0.0, 0.0, 1.0), // Red
    };
    ctx.pgl_set_uniform(&mut the_uniforms as *mut _ as *mut c_void);

    let vaos = ctx.gl_gen_vertex_arrays(1);
    ctx.gl_bind_vertex_array(vaos[0]).unwrap();

    let buffers = ctx.gl_gen_buffers(1);
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, buffers[0]).unwrap();
    let data: &[u8] = unsafe {
        core::slice::from_raw_parts(points.as_ptr() as *const u8, core::mem::size_of_val(&points))
    };
    ctx.gl_buffer_data(GL_ARRAY_BUFFER, data, GL_STATIC_DRAW).unwrap();
    ctx.gl_enable_vertex_attrib_array(0);
    ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, 0);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);

    let mut display_buf = vec![0u32; WIDTH * HEIGHT];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

        read_framebuffer(&ctx, &mut display_buf, WIDTH, HEIGHT);
        window.update_with_buffer(&display_buf, WIDTH, HEIGHT).unwrap();
    }
}
