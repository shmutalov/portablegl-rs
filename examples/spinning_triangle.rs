//! Example 3: Spinning Triangle
//!
//! Renders a color-interpolated triangle with 3D perspective projection
//! that rotates around the Y axis over time.
//! Port of PortableGL's examples/original/ex3.c
//!
//! Run: cargo run --example spinning_triangle --features examples

#![allow(unused_assignments)] // uniforms are read via raw pointer by the shader

use core::ffi::c_void;
use minifb::{Key, Window, WindowOptions};
use portablegl::gl_context::GlContext;
use portablegl::gl_types::*;
use portablegl::math::*;
use std::time::Instant;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

#[repr(C)]
struct MyUniforms {
    mvp_mat: Mat4,
}

unsafe extern "C" fn smooth_vs(
    vs_output: *mut f32,
    vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins,
    uniforms: *mut c_void,
) {
    // Pass color (attribute 4) to fragment shader
    *(vs_output as *mut Vec4) = *vertex_attribs.add(4);
    // Transform position by MVP matrix
    let u = &*(uniforms as *const MyUniforms);
    (*builtins).gl_Position = u.mvp_mat.mult_m4_v4(*vertex_attribs);
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
        "PortableGL-rs: Spinning Triangle (ESC to exit)",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .expect("Failed to create window");

    window.set_target_fps(60);

    let mut ctx = GlContext::new();
    let _pixels = ctx.init(WIDTH as i32, HEIGHT as i32);

    // Vertex positions
    let points: [f32; 9] = [
        -0.5, -0.5, 0.0,
         0.5, -0.5, 0.0,
         0.0,  0.5, 0.0,
    ];

    // Vertex colors (RGBA)
    let colors: [f32; 12] = [
        1.0, 0.0, 0.0, 1.0,  // Red
        0.0, 1.0, 0.0, 1.0,  // Green
        0.0, 0.0, 1.0, 1.0,  // Blue
    ];

    // Set up view-projection matrix
    let proj_mat = make_perspective_m4(radians(45.0), WIDTH as f32 / HEIGHT as f32, 1.0, 20.0);
    let trans_mat = translation_m4(0.0, 0.0, -5.0);
    let vp_mat = mult_m4_m4(proj_mat, trans_mat);

    // Interpolate 4 floats (vec4 color) smoothly
    let smooth: [GLenum; 4] = [PGL_SMOOTH; 4];
    let program = ctx.pgl_create_program(smooth_vs, smooth_fs, 4, &smooth, false);
    ctx.gl_use_program(program);

    let mut the_uniforms = MyUniforms {
        mvp_mat: Mat4::identity(),
    };
    ctx.pgl_set_uniform(&mut the_uniforms as *mut _ as *mut c_void);

    // Position buffer (attribute 0)
    let buffers = ctx.gl_gen_buffers(2);
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, buffers[0]).unwrap();
    let data: &[u8] = unsafe {
        core::slice::from_raw_parts(points.as_ptr() as *const u8, core::mem::size_of_val(&points))
    };
    ctx.gl_buffer_data(GL_ARRAY_BUFFER, data, GL_STATIC_DRAW).unwrap();
    ctx.gl_enable_vertex_attrib_array(0);
    ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, 0);

    // Color buffer (attribute 4)
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, buffers[1]).unwrap();
    let data: &[u8] = unsafe {
        core::slice::from_raw_parts(colors.as_ptr() as *const u8, core::mem::size_of_val(&colors))
    };
    ctx.gl_buffer_data(GL_ARRAY_BUFFER, data, GL_STATIC_DRAW).unwrap();
    ctx.gl_enable_vertex_attrib_array(4);
    ctx.gl_vertex_attrib_pointer(4, 4, GL_FLOAT, false, 0, 0);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);

    let mut save_rot = Mat4::identity();
    let mut last_frame = Instant::now();
    let mut display_buf = vec![0u32; WIDTH * HEIGHT];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let frame_time = now.duration_since(last_frame).as_secs_f32();
        last_frame = now;

        // Accumulate rotation around Y axis (30 degrees per second)
        let y_axis = Vec3::new(0.0, 1.0, 0.0);
        let rot_mat = load_rotation_m4(y_axis, radians(30.0) * frame_time);
        save_rot = mult_m4_m4(rot_mat, save_rot);
        the_uniforms.mvp_mat = mult_m4_m4(vp_mat, save_rot);

        ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

        read_framebuffer(&ctx, &mut display_buf, WIDTH, HEIGHT);
        window.update_with_buffer(&display_buf, WIDTH, HEIGHT).unwrap();
    }
}
