//! Example 1: Hello Triangle
//!
//! Renders a solid red triangle using a uniform color fragment shader.
//! Port of PortableGL's examples/original/ex1.c
//!
//! Run: cargo run --example hello_triangle --features examples

use core::ffi::c_void;
use sdl2::event::Event;
use sdl2::keyboard::Scancode;
use sdl2::pixels::PixelFormatEnum;
use portablegl::gl_context::GlContext;
use portablegl::gl_types::*;
use portablegl::math::*;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

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

fn main() {
    let sdl_context = sdl2::init().expect("Failed to init SDL2");
    let video = sdl_context.video().expect("Failed to init SDL2 video");

    let window = video
        .window("PortableGL-rs: Hello Triangle (ESC to exit)", WIDTH, HEIGHT)
        .position_centered()
        .build()
        .expect("Failed to create window");

    let mut canvas = window.into_canvas().software().build().expect("Failed to create canvas");
    let texture_creator = canvas.texture_creator();
    let mut texture = texture_creator
        .create_texture_streaming(PixelFormatEnum::ABGR8888, WIDTH, HEIGHT)
        .expect("Failed to create texture");

    let mut event_pump = sdl_context.event_pump().expect("Failed to get event pump");

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

    'main_loop: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown { scancode: Some(Scancode::Escape), .. } => break 'main_loop,
                _ => {}
            }
        }

        ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

        let fb = ctx.pgl_get_back_buffer();
        texture.update(None, &fb.buf, fb.w as usize * 4).unwrap();
        canvas.copy(&texture, None, None).unwrap();
        canvas.present();
    }
}
