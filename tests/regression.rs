#![allow(non_snake_case, non_upper_case_globals, unused, clippy::identity_op)]

use portablegl::gl_context::GlContext;
use portablegl::gl_types::*;
use portablegl::math::*;
use portablegl::pgl_shaders::*;
use core::ffi::c_void;
use std::path::Path;

const WIDTH: i32 = 640;
const HEIGHT: i32 = 640;

// ============================================================================
// Test harness: PNG I/O and comparison
// ============================================================================

fn abgr_to_rgba(pixel: u32) -> [u8; 4] {
    let r = (pixel & 0xFF) as u8;
    let g = ((pixel >> 8) & 0xFF) as u8;
    let b = ((pixel >> 16) & 0xFF) as u8;
    let a = ((pixel >> 24) & 0xFF) as u8;
    [r, g, b, a]
}

fn rgba_to_abgr(r: u8, g: u8, b: u8, a: u8) -> u32 {
    (a as u32) << 24 | (b as u32) << 16 | (g as u32) << 8 | (r as u32)
}

fn save_png(pixels: &[u32], w: u32, h: u32, path: &str) {
    let file = std::fs::File::create(path).expect("Failed to create PNG file");
    let ref mut bw = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(bw, w, h);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("Failed to write PNG header");
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for &px in pixels {
        let [r, g, b, a] = abgr_to_rgba(px);
        data.push(r);
        data.push(g);
        data.push(b);
        data.push(a);
    }
    writer.write_image_data(&data).expect("Failed to write PNG data");
}

fn load_png(path: &str) -> Option<(Vec<u32>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let w = info.width;
    let h = info.height;
    let bytes_per_pixel = info.color_type.samples() * if info.bit_depth == png::BitDepth::Sixteen { 2 } else { 1 };

    let mut pixels = Vec::with_capacity((w * h) as usize);
    let data = &buf[..info.buffer_size()];

    match (info.color_type, bytes_per_pixel) {
        (png::ColorType::Rgba, 4) => {
            for chunk in data.chunks_exact(4) {
                pixels.push(rgba_to_abgr(chunk[0], chunk[1], chunk[2], chunk[3]));
            }
        }
        (png::ColorType::Rgb, 3) => {
            for chunk in data.chunks_exact(3) {
                pixels.push(rgba_to_abgr(chunk[0], chunk[1], chunk[2], 255));
            }
        }
        _ => {
            eprintln!("Unsupported PNG format: {:?} {}bpp", info.color_type, bytes_per_pixel);
            return None;
        }
    }
    Some((pixels, w, h))
}

fn get_framebuffer_pixels(ctx: &GlContext) -> Vec<u32> {
    let fb = ctx.pgl_get_back_buffer();
    let len = (fb.w * fb.h) as usize;
    let mut pixels = Vec::with_capacity(len);
    for i in 0..len {
        let off = i * 4;
        let px = u32::from_le_bytes([fb.buf[off], fb.buf[off+1], fb.buf[off+2], fb.buf[off+3]]);
        pixels.push(px);
    }
    pixels
}

fn run_test(name: &str, test_fn: impl FnOnce(&mut GlContext)) {
    let mut ctx = GlContext::new();
    let _pixels = ctx.init(WIDTH, HEIGHT);

    test_fn(&mut ctx);

    let fb_pixels = get_framebuffer_pixels(&ctx);
    let out_dir = "testing/test_output";
    std::fs::create_dir_all(out_dir).ok();
    let out_path = format!("{}/{}.png", out_dir, name);
    save_png(&fb_pixels, WIDTH as u32, HEIGHT as u32, &out_path);

    let expected_path = format!("testing/expected_output/{}.png", name);
    if !Path::new(&expected_path).exists() {
        eprintln!("No expected output for '{}', skipping comparison", name);
        return;
    }

    let (expected, ew, eh) = load_png(&expected_path)
        .unwrap_or_else(|| panic!("Failed to load expected output: {}", expected_path));

    assert_eq!(ew, WIDTH as u32, "Width mismatch for {}", name);
    assert_eq!(eh, HEIGHT as u32, "Height mismatch for {}", name);

    // Allow per-channel tolerance of ±1 for floating-point rounding differences
    let mismatches: Vec<usize> = fb_pixels.iter().zip(expected.iter())
        .enumerate()
        .filter(|(_, (a, b))| {
            if a == b { return false; }
            let ar = (*a & 0xFF) as i32;
            let ag = ((*a >> 8) & 0xFF) as i32;
            let ab = ((*a >> 16) & 0xFF) as i32;
            let aa = ((*a >> 24) & 0xFF) as i32;
            let br = (*b & 0xFF) as i32;
            let bg = ((*b >> 8) & 0xFF) as i32;
            let bb = ((*b >> 16) & 0xFF) as i32;
            let ba = ((*b >> 24) & 0xFF) as i32;
            (ar - br).abs() > 1 || (ag - bg).abs() > 1 || (ab - bb).abs() > 1 || (aa - ba).abs() > 1
        })
        .map(|(i, _)| i)
        .collect();

    if !mismatches.is_empty() {
        // Print first 10 mismatches for debugging
        for &i in mismatches.iter().take(20) {
            let x = i % WIDTH as usize;
            let y = i / WIDTH as usize;
            let a = fb_pixels[i];
            let e = expected[i];
            eprintln!("  [{},{}] actual=0x{:08X} (r={},g={},b={},a={}) expected=0x{:08X} (r={},g={},b={},a={})",
                x, y,
                a, a&0xFF, (a>>8)&0xFF, (a>>16)&0xFF, (a>>24)&0xFF,
                e, e&0xFF, (e>>8)&0xFF, (e>>16)&0xFF, (e>>24)&0xFF,
            );
        }
        // Save diff image
        let mut diff = vec![0u32; fb_pixels.len()];
        for &i in &mismatches {
            diff[i] = 0xFFFFFFFF;
        }
        let diff_path = format!("{}/{}_diff.png", out_dir, name);
        save_png(&diff, WIDTH as u32, HEIGHT as u32, &diff_path);

        panic!(
            "Test '{}' FAILED: {} pixel mismatches out of {}. Diff saved to {}",
            name,
            mismatches.len(),
            fb_pixels.len(),
            diff_path
        );
    }
}

// ============================================================================
// Helper: cast any slice to &[u8]
// ============================================================================
fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    }
}

// ============================================================================
// Helper: set up a VBO from float data
// ============================================================================
fn setup_vbo(ctx: &mut GlContext, data: &[f32]) -> GLuint {
    let bufs = ctx.gl_gen_buffers(1);
    let buf = bufs[0];
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, buf);
    let _ = ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(data), GL_STATIC_DRAW);
    buf
}

fn setup_ebo(ctx: &mut GlContext, data: &[GLuint]) -> GLuint {
    let bufs = ctx.gl_gen_buffers(1);
    let buf = bufs[0];
    ctx.gl_bind_buffer(GL_ELEMENT_ARRAY_BUFFER, buf);
    let _ = ctx.gl_buffer_data(GL_ELEMENT_ARRAY_BUFFER, as_bytes(data), GL_STATIC_DRAW);
    buf
}

// ============================================================================
// Test: hello_triangle (argc=0)
// ============================================================================
#[test]
fn hello_triangle() {
    run_test("hello_triangle", |ctx| {
        let points: [f32; 9] = [
            -0.5, -0.5, 0.0,
             0.5, -0.5, 0.0,
             0.0,  0.5, 0.0,
        ];

        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
        setup_vbo(ctx, &points);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

        ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);
    });
}

// ============================================================================
// Test: client_arrays1 (hello_triangle argc=1)
// ============================================================================
#[test]
fn client_arrays1() {
    run_test("client_arrays1", |ctx| {
        let points: [f32; 9] = [
            -0.5, -0.5, 0.0,
             0.5, -0.5, 0.0,
             0.0,  0.5, 0.0,
        ];

        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
        // Client array: no VBO, pass pointer as offset
        ctx.gl_bind_buffer(GL_ARRAY_BUFFER, 0);
        ctx.gl_vertex_attrib_pointer(
            PGL_ATTR_VERT, 3, GL_FLOAT, false, 0,
            points.as_ptr() as GLsizeiptr,
        );

        ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);
    });
}

// ============================================================================
// Test: hello_indexing (argc=0..7)
// ============================================================================
fn hello_indexing_impl(ctx: &mut GlContext, argc: i32) {
    let points: [f32; 12] = [
        -0.5,  0.5, 0.0,
        -0.5, -0.5, 0.0,
         0.5,  0.5, 0.0,
         0.5, -0.5, 0.0,
    ];
    let indices: [GLuint; 6] = [0, 1, 2, 2, 1, 3];

    if argc < 2 || argc == 4 || argc == 5 {
        setup_ebo(ctx, &indices);
    }

    if argc < 4 {
        setup_vbo(ctx, &points);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);
    } else {
        ctx.gl_bind_buffer(GL_ARRAY_BUFFER, 0);
        ctx.gl_vertex_attrib_pointer(
            PGL_ATTR_VERT, 3, GL_FLOAT, false, 0,
            points.as_ptr() as GLsizeiptr,
        );
    }

    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);

    match argc {
        0 | 4 => ctx.gl_draw_elements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0),
        1 | 5 => ctx.gl_draw_elements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 3 * std::mem::size_of::<GLuint>()),
        2 | 6 => {
            ctx.gl_bind_buffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            ctx.gl_draw_elements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indices.as_ptr() as usize);
        }
        3 | 7 => {
            ctx.gl_bind_buffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            ctx.gl_draw_elements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, indices[3..].as_ptr() as usize);
        }
        _ => {}
    }
}

#[test] fn hello_indexing0() { run_test("hello_indexing0", |ctx| hello_indexing_impl(ctx, 0)); }
#[test] fn hello_indexing1() { run_test("hello_indexing1", |ctx| hello_indexing_impl(ctx, 1)); }
#[test] fn hello_indexing2() { run_test("hello_indexing2", |ctx| hello_indexing_impl(ctx, 2)); }
#[test] fn hello_indexing3() { run_test("hello_indexing3", |ctx| hello_indexing_impl(ctx, 3)); }
#[test] fn client_arrays3() { run_test("client_arrays3", |ctx| hello_indexing_impl(ctx, 4)); }
#[test] fn client_arrays4() { run_test("client_arrays4", |ctx| hello_indexing_impl(ctx, 5)); }
#[test] fn client_arrays5() { run_test("client_arrays5", |ctx| hello_indexing_impl(ctx, 6)); }
#[test] fn client_arrays6() { run_test("client_arrays6", |ctx| hello_indexing_impl(ctx, 7)); }

// ============================================================================
// Test: hello_interpolation (argc=0,1)
// ============================================================================
fn hello_interpolation_impl(ctx: &mut GlContext, argc: i32) {
    let points_n_colors: [f32; 18] = [
        -0.5, -0.5, 0.0,  1.0, 0.0, 0.0,
         0.5, -0.5, 0.0,  0.0, 1.0, 0.0,
         0.0,  0.5, 0.0,  0.0, 0.0, 1.0,
    ];

    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_COLOR);

    let stride = (6 * std::mem::size_of::<f32>()) as GLsizei;

    if argc == 0 {
        setup_vbo(ctx, &points_n_colors);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, stride, 0);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_COLOR, 3, GL_FLOAT, false, stride, (3 * std::mem::size_of::<f32>()) as GLsizeiptr);
    } else {
        ctx.gl_bind_buffer(GL_ARRAY_BUFFER, 0);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, stride, points_n_colors.as_ptr() as GLsizeiptr);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_COLOR, 3, GL_FLOAT, false, stride, points_n_colors[3..].as_ptr() as GLsizeiptr);
    }

    let std_shaders = pgl_init_std_shaders(ctx);
    ctx.gl_use_program(std_shaders[PGL_SHADER_SHADED]);

    let mut the_uniforms = PglUniforms::default();
    the_uniforms.mvp_mat = Mat4::identity();
    ctx.pgl_set_uniform(&mut the_uniforms as *mut PglUniforms as *mut c_void);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);
}

#[test] fn hello_interpolation() { run_test("hello_interpolation", |ctx| hello_interpolation_impl(ctx, 0)); }
#[test] fn client_arrays2() { run_test("client_arrays2", |ctx| hello_interpolation_impl(ctx, 1)); }

// ============================================================================
// Test: line_interpolation
// ============================================================================
#[test]
fn line_interpolation() {
    run_test("line_interpolation", |ctx| {
        #[rustfmt::skip]
        let points_n_colors: [f32; 96] = [
            -0.8, 0.9, 0.0,  1.0, 0.0, 0.0,
             0.4, 0.9, 0.0,  0.0, 0.0, 1.0,

            -5.0, 0.7, 0.0,  1.0, 0.0, 0.0,
             5.0, 0.7, 0.0,  0.0, 0.0, 1.0,

            -0.8, 0.4, 0.0,  1.0, 0.0, 0.0,
             0.4, 0.4, 0.0,  0.0, 0.0, 1.0,

            -5.0, 0.2, 0.0,  1.0, 0.0, 0.0,
             5.0, 0.2, 0.0,  0.0, 0.0, 1.0,

            -5.0, -0.2, 0.0,  1.0, 0.0, 0.0,
             5.0, -0.2, 0.0,  0.0, 0.0, 1.0,

            -0.8, -0.4, 0.0,  1.0, 0.0, 0.0,
             0.4, -0.4, 0.0,  0.0, 0.0, 1.0,

            -0.8, -0.9, 0.0,  1.0, 0.0, 0.0,
             0.4, -0.9, 0.0,  0.0, 0.0, 1.0,

            -5.0, -0.7, 0.0,  1.0, 0.0, 0.0,
             5.0, -0.7, 0.0,  0.0, 0.0, 1.0,
        ];

        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_COLOR);
        setup_vbo(ctx, &points_n_colors);
        let stride = (6 * std::mem::size_of::<f32>()) as GLsizei;
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, stride, 0);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_COLOR, 3, GL_FLOAT, false, stride, (3 * std::mem::size_of::<f32>()) as GLsizeiptr);

        let std_shaders = pgl_init_std_shaders(ctx);
        ctx.gl_use_program(std_shaders[PGL_SHADER_SHADED]);

        let mut the_uniforms = PglUniforms::default();
        the_uniforms.mvp_mat = Mat4::identity();
        ctx.pgl_set_uniform(&mut the_uniforms as *mut PglUniforms as *mut c_void);

        ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);

        ctx.gl_draw_arrays(GL_LINES, 0, 4);

        ctx.gl_line_width(8.0);
        ctx.gl_draw_arrays(GL_LINES, 4, 4);

        ctx.gl_enable(GL_LINE_SMOOTH);
        ctx.gl_line_width(1.0);
        ctx.gl_draw_arrays(GL_LINES, 8, 4);

        ctx.gl_line_width(8.0);
        ctx.gl_draw_arrays(GL_LINES, 12, 4);
    });
}

// ============================================================================
// Test: polygon_modes (argc=1, argc=8)
// ============================================================================
fn polygon_modes_impl(ctx: &mut GlContext, argc: i32) {
    #[rustfmt::skip]
    let points: [f32; 27] = [
        -0.8, -0.8, 0.0,
        -0.2, -0.8, 0.0,
        -0.5, -0.3, 0.0,

         0.2, -0.8, 0.0,
         0.8, -0.8, 0.0,
         0.5, -0.3, 0.0,

        -0.8, 0.3, 0.0,
        -0.2, 0.3, 0.0,
        -0.5, 0.8, 0.0,
    ];

    ctx.gl_line_width(argc as f32);
    ctx.gl_point_size(argc as f32);

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

    ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_POINT);
    ctx.gl_draw_arrays(GL_TRIANGLES, 3, 3);

    ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_LINE);
    ctx.gl_draw_arrays(GL_TRIANGLES, 6, 3);
}

#[test] fn polygon_modes() { run_test("polygon_modes", |ctx| polygon_modes_impl(ctx, 1)); }
#[test] fn polygon_modes_lw_ps() { run_test("polygon_modes_lw_ps", |ctx| polygon_modes_impl(ctx, 8)); }

// ============================================================================
// Test: front_back_culling (argc=0..11)
// ============================================================================
fn front_back_culling_impl(ctx: &mut GlContext, argc: i32) {
    #[rustfmt::skip]
    let points: [f32; 36] = [
        // bottom two are CCW
        -0.8, -0.8, 0.0,  -0.2, -0.8, 0.0,  -0.5, -0.3, 0.0,
         0.2, -0.8, 0.0,   0.8, -0.8, 0.0,   0.5, -0.3, 0.0,
        // top two are CW
        -0.2,  0.3, 0.0,  -0.8,  0.3, 0.0,  -0.5,  0.8, 0.0,
         0.8,  0.3, 0.0,   0.2,  0.3, 0.0,   0.5,  0.8, 0.0,
    ];

    match argc {
        1 => { ctx.gl_enable(GL_CULL_FACE); }
        2 => { ctx.gl_enable(GL_CULL_FACE); ctx.gl_front_face(GL_CW); }
        3 => { ctx.gl_enable(GL_CULL_FACE); ctx.gl_cull_face(GL_FRONT); }
        4 => { ctx.gl_enable(GL_CULL_FACE); ctx.gl_cull_face(GL_FRONT); ctx.gl_front_face(GL_CW); }
        5 => { ctx.gl_enable(GL_CULL_FACE); ctx.gl_cull_face(GL_FRONT_AND_BACK); }
        6 => { ctx.gl_polygon_mode(GL_FRONT, GL_POINT); }
        7 => { ctx.gl_polygon_mode(GL_BACK, GL_POINT); }
        8 => { ctx.gl_polygon_mode(GL_FRONT, GL_LINE); }
        9 => { ctx.gl_polygon_mode(GL_BACK, GL_LINE); }
        10 => { ctx.gl_polygon_mode(GL_FRONT, GL_LINE); ctx.gl_polygon_mode(GL_BACK, GL_POINT); }
        11 => { ctx.gl_front_face(GL_CW); ctx.gl_polygon_mode(GL_FRONT, GL_LINE); ctx.gl_polygon_mode(GL_BACK, GL_POINT); }
        _ => {}
    }

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 12);
}

#[test] fn cull_off() { run_test("cull_off", |ctx| front_back_culling_impl(ctx, 0)); }
#[test] fn cull_on() { run_test("cull_on", |ctx| front_back_culling_impl(ctx, 1)); }
#[test] fn cull_on_CW_front() { run_test("cull_on_CW_front", |ctx| front_back_culling_impl(ctx, 2)); }
#[test] fn cull_front_on() { run_test("cull_front_on", |ctx| front_back_culling_impl(ctx, 3)); }
#[test] fn cull_front_on_CW_front() { run_test("cull_front_on_CW_front", |ctx| front_back_culling_impl(ctx, 4)); }
#[test] fn cull_front_and_back() { run_test("cull_front_and_back", |ctx| front_back_culling_impl(ctx, 5)); }
#[test] fn front_pnt_back_fill() { run_test("front_pnt_back_fill", |ctx| front_back_culling_impl(ctx, 6)); }
#[test] fn front_fill_back_pnt() { run_test("front_fill_back_pnt", |ctx| front_back_culling_impl(ctx, 7)); }
#[test] fn front_line_back_fill() { run_test("front_line_back_fill", |ctx| front_back_culling_impl(ctx, 8)); }
#[test] fn front_fill_back_line() { run_test("front_fill_back_line", |ctx| front_back_culling_impl(ctx, 9)); }
#[test] fn front_line_back_point() { run_test("front_line_back_point", |ctx| front_back_culling_impl(ctx, 10)); }
#[test] fn front_line_back_point_CW() { run_test("front_line_back_point_CW", |ctx| front_back_culling_impl(ctx, 11)); }

// ============================================================================
// Test: clip_xy (argc=0..6)
// ============================================================================
fn clip_xy_impl(ctx: &mut GlContext, argc: i32) {
    #[rustfmt::skip]
    let points: [f32; 72] = [
        -0.7, 0.8, 0.0,  -0.3, 0.8, 0.0,  -0.5, 1.2, 0.0,
         0.3, 1.2, 0.0,   0.7, 1.2, 0.0,   0.5, 0.8, 0.0,
        -0.3,-0.8, 0.0,  -0.7,-0.8, 0.0,  -0.5,-1.2, 0.0,
         0.3,-1.2, 0.0,   0.7,-1.2, 0.0,   0.5,-0.8, 0.0,
        -0.8,-0.7, 0.0,  -0.8,-0.3, 0.0,  -1.2,-0.5, 0.0,
        -1.2, 0.3, 0.0,  -1.2, 0.7, 0.0,  -0.8, 0.5, 0.0,
         0.8,-0.3, 0.0,   0.8,-0.7, 0.0,   1.2,-0.5, 0.0,
         1.2, 0.7, 0.0,   1.2, 0.3, 0.0,   0.8, 0.5, 0.0,
    ];

    match argc {
        1 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_LINE); }
        2 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_POINT); }
        3 => { ctx.gl_line_width(8.0); ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_LINE); }
        4 => { ctx.gl_point_size(8.0); ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_POINT); }
        5 => { ctx.gl_line_width(32.0); ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_LINE); }
        6 => { ctx.gl_point_size(32.0); ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_POINT); }
        _ => {}
    }

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 24);
}

#[test] fn clip_xy_fill() { run_test("clip_xy_fill", |ctx| clip_xy_impl(ctx, 0)); }
#[test] fn clip_xy_line() { run_test("clip_xy_line", |ctx| clip_xy_impl(ctx, 1)); }
#[test] fn clip_xy_point() { run_test("clip_xy_point", |ctx| clip_xy_impl(ctx, 2)); }
#[test] fn clip_xy_line_8() { run_test("clip_xy_line_8", |ctx| clip_xy_impl(ctx, 3)); }
#[test] fn clip_xy_point_8() { run_test("clip_xy_point_8", |ctx| clip_xy_impl(ctx, 4)); }
#[test] fn clip_xy_line_32() { run_test("clip_xy_line_32", |ctx| clip_xy_impl(ctx, 5)); }
#[test] fn clip_xy_point_32() { run_test("clip_xy_point_32", |ctx| clip_xy_impl(ctx, 6)); }

// ============================================================================
// Test: clip_z (argc=0..7)
// ============================================================================
fn clip_z_impl(ctx: &mut GlContext, argc: i32) {
    #[rustfmt::skip]
    let points: [f32; 54] = [
        -0.9,-0.8, 0.0,  -0.5,-0.8, 0.0,  -0.7,-0.3, 0.0,
        -0.2,-0.8, 0.0,   0.2,-0.8, 0.0,   0.0,-0.3,-1.3,
         0.5,-0.8, 1.3,   0.9,-0.8, 1.3,   0.7,-0.3, 0.0,
        -0.9, 0.8, 0.0,  -0.5, 0.8, 0.0,  -0.7, 0.3, 0.0,
        -0.2, 0.8, 0.0,   0.2, 0.8, 0.0,   0.0, 0.3,-1.3,
         0.5, 0.8, 1.3,   0.9, 0.8, 1.3,   0.7, 0.3, 0.0,
    ];

    match argc {
        1 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_LINE); }
        2 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_POINT); }
        3 => { ctx.gl_line_width(8.0); ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_LINE); }
        4 => { ctx.gl_point_size(8.0); ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_POINT); }
        5 => { ctx.gl_line_width(32.0); ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_LINE); }
        6 => { ctx.gl_point_size(32.0); ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_POINT); }
        7 => { ctx.gl_enable(GL_DEPTH_CLAMP); }
        _ => {}
    }

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(0);
    ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, 0);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 18);
}

#[test] fn clip_z_fill() { run_test("clip_z_fill", |ctx| clip_z_impl(ctx, 0)); }
#[test] fn clip_z_line() { run_test("clip_z_line", |ctx| clip_z_impl(ctx, 1)); }
#[test] fn clip_z_point() { run_test("clip_z_point", |ctx| clip_z_impl(ctx, 2)); }
#[test] fn clip_z_line_8() { run_test("clip_z_line_8", |ctx| clip_z_impl(ctx, 3)); }
#[test] fn clip_z_point_8() { run_test("clip_z_point_8", |ctx| clip_z_impl(ctx, 4)); }
#[test] fn clip_z_line_32() { run_test("clip_z_line_32", |ctx| clip_z_impl(ctx, 5)); }
#[test] fn clip_z_point_32() { run_test("clip_z_point_32", |ctx| clip_z_impl(ctx, 6)); }
#[test] fn depth_clamp() { run_test("depth_clamp", |ctx| clip_z_impl(ctx, 7)); }

// ============================================================================
// Test: clip_pnts_lns (argc=0..2)
// ============================================================================
fn clip_pnts_lns_impl(ctx: &mut GlContext, argc: i32) {
    #[rustfmt::skip]
    let points: [f32; 36] = [
        -1.1, 0.7, 0.0,   -0.7, 1.1, 0.0,
         1.1,-0.7, 0.0,    0.7,-1.1, 0.0,
        -0.3, 0.5, 1.5,    0.3,-0.5,-1.5,
        -0.3,-0.3, 1.2,    0.3, 0.3,-1.2,
        -0.9, 0.5, 0.0,    0.9, 0.5, 0.0,
        -1.02,-0.5, 0.0,   1.02,-0.5, 0.0,
    ];

    match argc {
        1 => { ctx.gl_point_size(8.0); ctx.gl_line_width(8.0); }
        2 => { ctx.gl_point_size(32.0); ctx.gl_line_width(32.0); }
        _ => {}
    }

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(0);
    ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, 0);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_LINES, 0, 6);
    ctx.gl_draw_arrays(GL_POINTS, 6, 6);
}

#[test] fn clip_pnts_lns() { run_test("clip_pnts_lns", |ctx| clip_pnts_lns_impl(ctx, 0)); }
#[test] fn clip_pnts_lns8() { run_test("clip_pnts_lns8", |ctx| clip_pnts_lns_impl(ctx, 1)); }
#[test] fn clip_pnts_lns32() { run_test("clip_pnts_lns32", |ctx| clip_pnts_lns_impl(ctx, 2)); }

// ============================================================================
// Test: clip_projection
// ============================================================================
unsafe extern "C" fn skybox_vs(
    vs_output: *mut f32, vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let u = &*(uniforms as *const PglUniforms);
    *(vs_output as *mut Vec4) = *vertex_attribs.add(1);
    let pos = mult_m4_v4(u.mvp_mat, *vertex_attribs.add(0));
    (*builtins).gl_Position = Vec4::new(pos.x, pos.y, pos.w, pos.w);
}

unsafe extern "C" fn skybox_fs(
    fs_input: *mut f32, builtins: *mut ShaderBuiltins, _uniforms: *mut c_void,
) {
    let color = *(fs_input as *const Vec3);
    (*builtins).gl_FragColor = Vec4::new(color.x, color.y, color.z, 1.0);
}

#[test]
fn clip_projection() {
    run_test("clip_projection", |ctx| {
        #[rustfmt::skip]
        let skybox: [f32; 216] = [
            // front (red)
            -1.0, 1.0,-1.0, 1.0,0.0,0.0,  -1.0,-1.0,-1.0, 1.0,0.0,0.0,   1.0,-1.0,-1.0, 1.0,0.0,0.0,
             1.0,-1.0,-1.0, 1.0,0.0,0.0,   1.0, 1.0,-1.0, 1.0,0.0,0.0,  -1.0, 1.0,-1.0, 1.0,0.0,0.0,
            // left (green)
            -1.0,-1.0, 1.0, 0.0,1.0,0.0,  -1.0,-1.0,-1.0, 0.0,1.0,0.0,  -1.0, 1.0,-1.0, 0.0,1.0,0.0,
            -1.0, 1.0,-1.0, 0.0,1.0,0.0,  -1.0, 1.0, 1.0, 0.0,1.0,0.0,  -1.0,-1.0, 1.0, 0.0,1.0,0.0,
            // right (blue)
             1.0,-1.0,-1.0, 0.0,0.0,1.0,   1.0,-1.0, 1.0, 0.0,0.0,1.0,   1.0, 1.0, 1.0, 0.0,0.0,1.0,
             1.0, 1.0, 1.0, 0.0,0.0,1.0,   1.0, 1.0,-1.0, 0.0,0.0,1.0,   1.0,-1.0,-1.0, 0.0,0.0,1.0,
            // back (yellow)
            -1.0,-1.0, 1.0, 1.0,1.0,0.0,  -1.0, 1.0, 1.0, 1.0,1.0,0.0,   1.0, 1.0, 1.0, 1.0,1.0,0.0,
             1.0, 1.0, 1.0, 1.0,1.0,0.0,   1.0,-1.0, 1.0, 1.0,1.0,0.0,  -1.0,-1.0, 1.0, 1.0,1.0,0.0,
            // top (cyan)
            -1.0, 1.0,-1.0, 0.0,1.0,1.0,   1.0, 1.0,-1.0, 0.0,1.0,1.0,   1.0, 1.0, 1.0, 0.0,1.0,1.0,
             1.0, 1.0, 1.0, 0.0,1.0,1.0,  -1.0, 1.0, 1.0, 0.0,1.0,1.0,  -1.0, 1.0,-1.0, 0.0,1.0,1.0,
            // bottom (magenta)
            -1.0,-1.0,-1.0, 1.0,0.0,1.0,  -1.0,-1.0, 1.0, 1.0,0.0,1.0,   1.0,-1.0,-1.0, 1.0,0.0,1.0,
             1.0,-1.0,-1.0, 1.0,0.0,1.0,  -1.0,-1.0, 1.0, 1.0,0.0,1.0,   1.0,-1.0, 1.0, 1.0,0.0,1.0,
        ];

        setup_vbo(ctx, &skybox);
        ctx.gl_enable_vertex_attrib_array(0);
        ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 6 * 4, 0);
        ctx.gl_enable_vertex_attrib_array(1);
        ctx.gl_vertex_attrib_pointer(1, 3, GL_FLOAT, false, 6 * 4, 3 * 4);

        let flat3: [GLenum; 3] = [PGL_FLAT; 3];
        let shader = ctx.pgl_create_program(skybox_vs, skybox_fs, 3, &flat3, false);
        ctx.gl_use_program(shader);

        let mut uniforms = PglUniforms::default();
        let proj = make_perspective_m4(radians(45.0), WIDTH as f32 / HEIGHT as f32, 0.1, 100.0);
        let eye = Vec3::new(0.0, 0.0, 0.0);
        let up = Vec3::new(0.0, 1.0, 0.0);
        let forward = Vec3::new(0.0, 0.0, 1.0);
        let nf = norm_v3(forward);
        let view = look_at(eye, add_v3s(eye, nf), up);
        uniforms.mvp_mat = mult_m4_m4(proj, view);

        ctx.pgl_set_uniform(&mut uniforms as *mut PglUniforms as *mut c_void);

        ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        ctx.gl_depth_func(GL_LEQUAL);
        ctx.gl_use_program(shader);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 36);
    });
}

// ============================================================================
// Test: viewport (argc=0..2)
// ============================================================================
fn test_viewport_impl(ctx: &mut GlContext, argc: i32) {
    let points: [f32; 9] = [-0.5,-0.5,0.0, 0.5,-0.5,0.0, 0.0,0.5,0.0];

    match argc {
        1 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_LINE); }
        2 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_POINT); }
        _ => {}
    }

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    let std_shaders = pgl_init_std_shaders(ctx);
    ctx.gl_use_program(std_shaders[PGL_SHADER_IDENTITY]);

    let mut u = PglUniforms::default();
    ctx.pgl_set_uniform(&mut u as *mut PglUniforms as *mut c_void);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);

    u.color = Vec4::new(1.0, 0.0, 0.0, 1.0);
    ctx.gl_viewport(0, 0, WIDTH / 2, HEIGHT / 2);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

    u.color = Vec4::new(0.0, 1.0, 0.0, 1.0);
    ctx.gl_viewport(0, HEIGHT / 2, WIDTH / 2, HEIGHT / 2);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

    u.color = Vec4::new(0.0, 0.0, 1.0, 1.0);
    ctx.gl_viewport(WIDTH / 2, HEIGHT / 2, WIDTH / 2, HEIGHT / 2);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

    u.color = Vec4::new(1.0, 0.0, 1.0, 1.0);
    ctx.gl_viewport(WIDTH / 2, 0, WIDTH / 2, HEIGHT / 2);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

    u.color = Vec4::new(0.0, 1.0, 1.0, 1.0);
    ctx.gl_viewport(-WIDTH / 4, -HEIGHT / 4, WIDTH / 2, HEIGHT / 2);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

    u.color = Vec4::new(1.0, 1.0, 0.0, 1.0);
    ctx.gl_viewport(-WIDTH / 4, 3 * HEIGHT / 4, WIDTH / 2, HEIGHT / 2);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

    u.color = Vec4::new(1.0, 1.0, 1.0, 1.0);
    ctx.gl_viewport(3 * WIDTH / 4, 3 * HEIGHT / 4, WIDTH / 2, HEIGHT / 2);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

    u.color = Vec4::new(0.5, 0.5, 0.5, 0.5);
    ctx.gl_viewport(3 * WIDTH / 4, -HEIGHT / 4, WIDTH / 2, HEIGHT / 2);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);
}

#[test] fn viewport_fill() { run_test("viewport_fill", |ctx| test_viewport_impl(ctx, 0)); }
#[test] fn viewport_line() { run_test("viewport_line", |ctx| test_viewport_impl(ctx, 1)); }
#[test] fn viewport_point() { run_test("viewport_point", |ctx| test_viewport_impl(ctx, 2)); }

// ============================================================================
// Test: blend_test
// ============================================================================
#[test]
fn blend_test() {
    run_test("blend_test", |ctx| {
        #[rustfmt::skip]
        let points: [f32; 108] = [
            -0.75, 0.75, 0.0,  -0.75, 0.25, 0.0,  -0.25, 0.75, 0.0,  -0.25, 0.25, 0.0,
             0.25, 0.75, 0.0,   0.25, 0.25, 0.0,   0.75, 0.75, 0.0,   0.75, 0.25, 0.0,
            -0.75,-0.25, 0.0,  -0.75,-0.75, 0.0,  -0.25,-0.25, 0.0,  -0.25,-0.75, 0.0,
             0.25,-0.25, 0.0,   0.25,-0.75, 0.0,   0.75,-0.25, 0.0,   0.75,-0.75, 0.0,
            -0.15, 0.15,-0.1,  -0.15,-0.15,-0.1,   0.15, 0.15,-0.1,   0.15,-0.15,-0.1,
            -0.40, 0.65,-0.1,  -0.40, 0.35,-0.1,  -0.10, 0.65,-0.1,  -0.10, 0.35,-0.1,
             0.10, 0.65,-0.1,   0.10, 0.35,-0.1,   0.40, 0.65,-0.1,   0.40, 0.35,-0.1,
            -0.40,-0.35,-0.1,  -0.40,-0.65,-0.1,  -0.10,-0.35,-0.1,  -0.10,-0.65,-0.1,
             0.10,-0.35,-0.1,   0.10,-0.65,-0.1,   0.40,-0.35,-0.1,   0.40,-0.65,-0.1,
        ];

        let mut u = PglUniforms::default();

        setup_vbo(ctx, &points);
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

        let std_shaders = pgl_init_std_shaders(ctx);
        ctx.gl_use_program(std_shaders[PGL_SHADER_IDENTITY]);
        ctx.pgl_set_uniform(&mut u as *mut PglUniforms as *mut c_void);

        ctx.gl_clear_color(1.0, 1.0, 1.0, 1.0);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);

        u.color = Vec4::new(1.0, 0.0, 0.0, 1.0);
        ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 0, 4);
        u.color = Vec4::new(0.0, 1.0, 0.0, 1.0);
        ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 4, 4);
        u.color = Vec4::new(0.0, 0.0, 1.0, 1.0);
        ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 8, 4);
        u.color = Vec4::new(0.0, 0.0, 0.0, 1.0);
        ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 12, 4);

        ctx.gl_enable(GL_BLEND);
        ctx.gl_blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        u.color = Vec4::new(1.0, 0.0, 0.0, 0.5);
        ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 16, 4);
        ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 20, 4);
        ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 24, 4);
        ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 28, 4);
        ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 32, 4);
        ctx.gl_disable(GL_BLEND);
    });
}

// ============================================================================
// Test: stencil_test
// ============================================================================
#[test]
fn stencil_test() {
    run_test("stencil_test", |ctx| {
        let points: [f32; 9] = [-0.5,-0.5,0.0, 0.5,-0.5,0.0, 0.0,0.5,0.0];
        let color_array: [f32; 12] = [
            1.0,0.0,0.0,1.0,  0.0,1.0,0.0,1.0,  0.0,0.0,1.0,1.0,
        ];

        let _vb = setup_vbo(ctx, &points);
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

        let _cb = setup_vbo(ctx, &color_array);
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_COLOR);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_COLOR, 4, GL_FLOAT, false, 0, 0);

        let mut u = PglUniforms::default();
        u.mvp_mat = Mat4::identity();
        u.color = Vec4::new(1.0, 0.0, 0.0, 1.0);

        let std_shaders = pgl_init_std_shaders(ctx);
        let myshader = std_shaders[PGL_SHADER_SHADED];
        ctx.gl_use_program(myshader);
        ctx.pgl_set_uniform(&mut u as *mut PglUniforms as *mut c_void);

        let basic = std_shaders[PGL_SHADER_FLAT];
        ctx.gl_use_program(basic);
        ctx.pgl_set_uniform(&mut u as *mut PglUniforms as *mut c_void);

        ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.gl_enable(GL_STENCIL_TEST);
        ctx.gl_stencil_func(GL_NOTEQUAL, 1, 0xFF);
        ctx.gl_stencil_op(GL_KEEP, GL_REPLACE, GL_REPLACE);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        ctx.gl_use_program(myshader);
        ctx.gl_stencil_func(GL_ALWAYS, 1, 0xFF);
        ctx.gl_stencil_mask(0xFF);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

        ctx.gl_use_program(basic);
        ctx.gl_stencil_func(GL_NOTEQUAL, 1, 0xFF);
        ctx.gl_stencil_mask(0x00);

        scale_m4(&mut u.mvp_mat, 1.2, 1.2, 1.2);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);
    });
}

// ============================================================================
// Test: zbuf_test (argc=0..4)
// ============================================================================
fn zbuf_test_impl(ctx: &mut GlContext, argc: i32) {
    #[rustfmt::skip]
    let points: [f32; 27] = [
        -1.0, 1.0, 0.9,  -1.0,-1.0, 0.9,   1.0,-1.0,-0.9,
         1.0, 1.0, 0.9,  -1.0,-1.0,-0.9,   1.0,-1.0, 0.9,
        -0.5,-0.5, 0.0,   0.5,-0.5, 0.0,   0.0, 0.5, 0.0,
    ];

    let mut u = PglUniforms::default();
    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    let std_shaders = pgl_init_std_shaders(ctx);
    ctx.gl_use_program(std_shaders[PGL_SHADER_IDENTITY]);
    ctx.pgl_set_uniform(&mut u as *mut PglUniforms as *mut c_void);

    if argc >= 1 { ctx.gl_enable(GL_DEPTH_TEST); }
    if argc == 2 { ctx.gl_clear_depth(0.0); ctx.gl_depth_func(GL_GREATER); }
    else if argc == 3 { ctx.gl_depth_range(1.0, 0.0); }

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    u.color = Vec4::new(1.0, 0.0, 0.0, 1.0);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

    if argc == 4 { ctx.gl_depth_mask(false); }
    u.color = Vec4::new(0.0, 1.0, 0.0, 1.0);
    ctx.gl_draw_arrays(GL_TRIANGLES, 6, 3);
    if argc == 4 { ctx.gl_depth_mask(true); }

    u.color = Vec4::new(0.0, 0.0, 1.0, 1.0);
    ctx.gl_draw_arrays(GL_TRIANGLES, 3, 3);
}

#[test] fn zbuf_depthoff() { run_test("zbuf_depthoff", |ctx| zbuf_test_impl(ctx, 0)); }
#[test] fn zbuf_depthon() { run_test("zbuf_depthon", |ctx| zbuf_test_impl(ctx, 1)); }
#[test] fn zbuf_depthon_greater() { run_test("zbuf_depthon_greater", |ctx| zbuf_test_impl(ctx, 2)); }
#[test] fn zbuf_depthon_fliprange() { run_test("zbuf_depthon_fliprange", |ctx| zbuf_test_impl(ctx, 3)); }
#[test] fn zbuf_depthon_maskoff() { run_test("zbuf_depthon_maskoff", |ctx| zbuf_test_impl(ctx, 4)); }

// ============================================================================
// Test: primitives_test
// ============================================================================
#[test]
fn primitives_test() {
    run_test("primitives_test", |ctx| {
        #[rustfmt::skip]
        let points: [f32; 66] = [
            // triangle strip (6 verts)
            -0.8, 0.0, 0.0,  -0.8,-0.8, 0.0,  -0.4, 0.0, 0.0,
            -0.4,-0.8, 0.0,   0.0, 0.0, 0.0,   0.0,-0.8, 0.0,
            // triangle fan (5 verts)
             0.0, 0.0, 0.0,   0.5, 0.0, 0.0,   0.5, 0.5, 0.0,
             0.0, 0.5, 0.0,  -0.5, 0.5, 0.0,
            // lines (4 verts)
            -0.95, 0.95, 0.0, -0.8, 0.8, 0.0,
            -0.75, 0.95, 0.0, -0.6, 0.95, 0.0,
            // line loop (4 verts)
            -0.5, 0.95, 0.0, -0.4, 0.95, 0.0, -0.4, 0.85, 0.0, -0.5, 0.85, 0.0,
            // line strip (3 verts)
            -0.3, 0.85, 0.0, -0.1, 0.65, 0.0,  0.1, 0.95, 0.0,
        ];
        #[rustfmt::skip]
        let color_array: [f32; 88] = [
            // triangle strip
            1.0,0.0,0.0,1.0, 0.0,1.0,0.0,1.0, 0.0,0.0,1.0,1.0,
            1.0,0.0,0.0,1.0, 0.0,1.0,0.0,1.0, 0.0,0.0,1.0,1.0,
            // triangle fan
            1.0,0.0,0.0,1.0, 0.0,1.0,0.0,1.0, 0.0,0.0,1.0,1.0,
            1.0,0.0,0.0,1.0, 0.0,1.0,0.0,1.0,
            // lines
            0.0,0.0,1.0,1.0, 1.0,0.0,0.0,1.0,
            0.0,1.0,0.0,1.0, 0.0,0.0,1.0,1.0,
            // line loop
            1.0,0.0,0.0,1.0, 0.0,1.0,0.0,1.0, 0.0,0.0,1.0,1.0, 1.0,0.0,0.0,1.0,
            // line strip
            0.0,1.0,0.0,1.0, 0.0,0.0,1.0,1.0, 1.0,0.0,0.0,1.0,
        ];

        let mut u = PglUniforms::default();
        setup_vbo(ctx, &points);
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

        setup_vbo(ctx, &color_array);
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_COLOR);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_COLOR, 4, GL_FLOAT, false, 0, 0);

        ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
        let std_shaders = pgl_init_std_shaders(ctx);
        ctx.gl_use_program(std_shaders[PGL_SHADER_SHADED]);
        ctx.pgl_set_uniform(&mut u as *mut PglUniforms as *mut c_void);
        u.mvp_mat = Mat4::identity();

        ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 0, 6);
        ctx.gl_draw_arrays(GL_TRIANGLE_FAN, 6, 5);
        ctx.gl_draw_arrays(GL_LINES, 11, 4);
        ctx.gl_draw_arrays(GL_LINE_LOOP, 15, 4);
        ctx.gl_draw_arrays(GL_LINE_STRIP, 19, 3);
    });
}

// ============================================================================
// Test: test_edges
// ============================================================================
#[test]
fn test_edges() {
    run_test("test_edges", |ctx| {
        let points: [f32; 12] = [-1.0,1.0,0.0, 1.0,1.0,0.0, 1.0,-1.0,0.0, -1.0,-1.0,0.0];
        setup_vbo(ctx, &points);
        ctx.gl_enable_vertex_attrib_array(0);
        ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, 0);
        ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        ctx.gl_draw_arrays(GL_LINE_LOOP, 0, 4);
    });
}

// ============================================================================
// Test: color_masking
// ============================================================================
#[test]
fn color_masking() {
    run_test("color_masking", |ctx| {
        let points_n_colors: [f32; 18] = [
            -0.5,-0.5,0.0,  1.0,0.0,0.0,
             0.5,-0.5,0.0,  0.0,1.0,0.0,
             0.0, 0.5,0.0,  0.0,0.0,1.0,
        ];

        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_COLOR);
        setup_vbo(ctx, &points_n_colors);
        let stride = (6 * std::mem::size_of::<f32>()) as GLsizei;
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, stride, 0);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_COLOR, 3, GL_FLOAT, false, stride, (3 * std::mem::size_of::<f32>()) as GLsizeiptr);

        let std_shaders = pgl_init_std_shaders(ctx);
        ctx.gl_use_program(std_shaders[PGL_SHADER_SHADED]);

        let mut u = PglUniforms::default();
        u.mvp_mat = Mat4::identity();
        ctx.pgl_set_uniform(&mut u as *mut PglUniforms as *mut c_void);

        // Clear to 0
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);

        // Clear to white but mask out blue
        ctx.gl_clear_color(1.0, 1.0, 1.0, 1.0);
        ctx.gl_color_mask(true, true, false, true);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);

        // Draw triangle but mask out red
        ctx.gl_color_mask(false, true, true, true);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);
    });
}

// ============================================================================
// Test: mapped_vbuffer
// ============================================================================
#[test]
fn map_vbuffer() {
    run_test("map_vbuffer", |ctx| {
        let points_n_colors: [f32; 18] = [
            -0.5,-0.5,0.0,  1.0,0.0,0.0,
             0.5,-0.5,0.0,  0.0,1.0,0.0,
             0.0, 0.5,0.0,  0.0,0.0,1.0,
        ];

        setup_vbo(ctx, &points_n_colors);
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
        let stride = (6 * std::mem::size_of::<f32>()) as GLsizei;
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, stride, 0);
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_COLOR);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_COLOR, 3, GL_FLOAT, false, stride, (3 * std::mem::size_of::<f32>()) as GLsizeiptr);

        let pts = ctx.gl_map_buffer(GL_ARRAY_BUFFER, GL_READ_WRITE) as *mut f32;
        unsafe {
            *pts.add(0) = -1.0;   // modify x of first vertex
            *pts.add(5) = 1.0;    // modify blue of first vertex color
        }

        let std_shaders = pgl_init_std_shaders(ctx);
        ctx.gl_use_program(std_shaders[PGL_SHADER_SHADED]);

        let mut u = PglUniforms::default();
        u.mvp_mat = Mat4::identity();
        ctx.pgl_set_uniform(&mut u as *mut PglUniforms as *mut c_void);

        ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);
    });
}

// ============================================================================
// Test: mapped_nvbuffer (glMapNamedBuffer)
// ============================================================================
#[test]
fn map_nvbuffer() {
    run_test("map_nvbuffer", |ctx| {
        let points_n_colors: [f32; 18] = [
            -0.5,-0.5,0.0,  1.0,0.0,0.0,
             0.5,-0.5,0.0,  0.0,1.0,0.0,
             0.0, 0.5,0.0,  0.0,0.0,1.0,
        ];

        let buf = setup_vbo(ctx, &points_n_colors);
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
        let stride = (6 * std::mem::size_of::<f32>()) as GLsizei;
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, stride, 0);
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_COLOR);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_COLOR, 3, GL_FLOAT, false, stride, (3 * std::mem::size_of::<f32>()) as GLsizeiptr);

        let pts = ctx.gl_map_named_buffer(buf, GL_READ_WRITE) as *mut f32;
        unsafe {
            *pts.add(6) = 1.0;    // modify x of second vertex
            *pts.add(11) = 1.0;   // modify blue of second vertex color
        }

        let std_shaders = pgl_init_std_shaders(ctx);
        ctx.gl_use_program(std_shaders[PGL_SHADER_SHADED]);

        let mut u = PglUniforms::default();
        u.mvp_mat = Mat4::identity();
        ctx.pgl_set_uniform(&mut u as *mut PglUniforms as *mut c_void);

        ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);
    });
}

// ============================================================================
// Test: pglbufferdata (pgl_buffer_data stores raw pointer, not copy)
// ============================================================================
#[test]
fn pglbufferdata() {
    run_test("pglbufferdata", |ctx| {
        let mut points_n_colors: [f32; 18] = [
            -0.5,-0.5,0.0,  1.0,0.0,0.0,
             0.5,-0.5,0.0,  0.0,1.0,0.0,
             0.0, 0.5,0.0,  0.0,0.0,1.0,
        ];

        let bufs = ctx.gl_gen_buffers(1);
        ctx.gl_bind_buffer(GL_ARRAY_BUFFER, bufs[0]);
        ctx.pgl_buffer_data(
            GL_ARRAY_BUFFER,
            (points_n_colors.len() * std::mem::size_of::<f32>()) as GLsizeiptr,
            points_n_colors.as_mut_ptr() as *mut u8,
            false,
        );
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
        let stride = (6 * std::mem::size_of::<f32>()) as GLsizei;
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, stride, 0);
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_COLOR);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_COLOR, 3, GL_FLOAT, false, stride, (3 * std::mem::size_of::<f32>()) as GLsizeiptr);

        // Modify original array - pgl_buffer_data stores a pointer so changes are reflected
        points_n_colors[13] = 1.0;
        points_n_colors[15] = 1.0;

        let std_shaders = pgl_init_std_shaders(ctx);
        ctx.gl_use_program(std_shaders[PGL_SHADER_SHADED]);

        let mut u = PglUniforms::default();
        u.mvp_mat = Mat4::identity();
        ctx.pgl_set_uniform(&mut u as *mut PglUniforms as *mut c_void);

        ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);
    });
}

// ============================================================================
// Test: instancing (argc=0..3)
// ============================================================================
unsafe extern "C" fn instancing_vs(
    vs_output: *mut f32, vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins, _uniforms: *mut c_void,
) {
    let mut vert = *vertex_attribs.add(0);
    let offset = *vertex_attribs.add(1);
    vert.x += offset.x;
    vert.y += offset.y;
    let color = *vertex_attribs.add(2);
    *(vs_output as *mut Vec3) = Vec3::new(color.x, color.y, color.z);
    (*builtins).gl_Position = vert;
}

unsafe extern "C" fn instancing_fs(
    fs_input: *mut f32, builtins: *mut ShaderBuiltins, _uniforms: *mut c_void,
) {
    (*builtins).gl_FragColor = Vec4::new(*fs_input, *fs_input.add(1), *fs_input.add(2), 1.0);
}

fn test_instancing_impl(ctx: &mut GlContext, argc: i32) {
    let points: [f32; 9] = [-0.05,-0.05,0.0, 0.05,-0.05,0.0, 0.0,0.05,0.0];
    let indices: [GLuint; 3] = [0, 1, 2];

    let mut positions = [Vec2::new(0.0, 0.0); 100];
    let mut i = 0;
    let offset = 0.1f32;
    let mut y = -10i32;
    while y < 10 {
        let mut x = -10i32;
        while x < 10 {
            positions[i] = Vec2::new(x as f32 / 10.0 + offset, y as f32 / 10.0 + offset);
            i += 1;
            x += 2;
        }
        y += 2;
    }

    #[rustfmt::skip]
    let inst_colors: [Vec3; 10] = [
        Vec3::new(0.783099, 0.394383, 0.840188),
        Vec3::new(0.197551, 0.911647, 0.798440),
        Vec3::new(0.277775, 0.768230, 0.335223),
        Vec3::new(0.628871, 0.477397, 0.553970),
        Vec3::new(0.952230, 0.513401, 0.364784),
        Vec3::new(0.717297, 0.635712, 0.916195),
        Vec3::new(0.016301, 0.606969, 0.141603),
        Vec3::new(0.804177, 0.137232, 0.242887),
        Vec3::new(0.129790, 0.400944, 0.156679),
        Vec3::new(0.218257, 0.998924, 0.108809),
    ];

    if argc < 2 {
        // VBO path for instance data
        let pos_buf = ctx.gl_gen_buffers(1)[0];
        ctx.gl_bind_buffer(GL_ARRAY_BUFFER, pos_buf);
        let _ = ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&positions), GL_STATIC_DRAW);
        ctx.gl_enable_vertex_attrib_array(1);
        ctx.gl_vertex_attrib_pointer(1, 2, GL_FLOAT, false, 0, 0);
        ctx.gl_vertex_attrib_divisor(1, 1);

        let col_buf = ctx.gl_gen_buffers(1)[0];
        ctx.gl_bind_buffer(GL_ARRAY_BUFFER, col_buf);
        let _ = ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&inst_colors), GL_STATIC_DRAW);
        ctx.gl_enable_vertex_attrib_array(2);
        ctx.gl_vertex_attrib_pointer(2, 3, GL_FLOAT, false, 0, 0);
        ctx.gl_vertex_attrib_divisor(2, 10);

        setup_vbo(ctx, &points);
        ctx.gl_enable_vertex_attrib_array(0);
        ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, 0);
    } else {
        // Client array path
        ctx.gl_bind_buffer(GL_ARRAY_BUFFER, 0);
        ctx.gl_enable_vertex_attrib_array(1);
        ctx.gl_vertex_attrib_pointer(1, 2, GL_FLOAT, false, 0, positions.as_ptr() as GLsizeiptr);
        ctx.gl_vertex_attrib_divisor(1, 1);

        ctx.gl_enable_vertex_attrib_array(2);
        ctx.gl_vertex_attrib_pointer(2, 3, GL_FLOAT, false, 0, inst_colors.as_ptr() as GLsizeiptr);
        ctx.gl_vertex_attrib_divisor(2, 10);

        ctx.gl_enable_vertex_attrib_array(0);
        ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, points.as_ptr() as GLsizeiptr);
    }

    if argc == 1 {
        setup_ebo(ctx, &indices);
    }

    let flat3: [GLenum; 3] = [PGL_FLAT; 3];
    let myshader = ctx.pgl_create_program(instancing_vs, instancing_fs, 3, &flat3, false);
    ctx.gl_use_program(myshader);
    ctx.pgl_set_uniform(core::ptr::null_mut());

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);

    if argc == 0 || argc >= 2 {
        if argc >= 2 {
            ctx.gl_bind_buffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            ctx.gl_draw_elements_instanced(GL_TRIANGLES, 3, GL_UNSIGNED_INT, indices.as_ptr() as usize, 100);
        } else {
            ctx.gl_draw_arrays_instanced(GL_TRIANGLES, 0, 3, 100);
        }
    } else {
        ctx.gl_draw_elements_instanced(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0, 100);
    }
}

#[test] fn instancing_arrays() { run_test("instancing_arrays", |ctx| test_instancing_impl(ctx, 0)); }
#[test] fn instancing_elements() { run_test("instancing_elements", |ctx| test_instancing_impl(ctx, 1)); }
#[test] fn client_arrays7() { run_test("client_arrays7", |ctx| test_instancing_impl(ctx, 2)); }
#[test] fn client_arrays8() { run_test("client_arrays8", |ctx| test_instancing_impl(ctx, 3)); }

// ============================================================================
// Test: baseinstance (argc=0..3)
// ============================================================================
fn test_baseinstance_impl(ctx: &mut GlContext, argc: i32) {
    let points: [f32; 9] = [-0.05,-0.05,0.0, 0.05,-0.05,0.0, 0.0,0.05,0.0];
    let indices: [GLuint; 3] = [0, 1, 2];

    let mut positions = [Vec2::new(0.0, 0.0); 100];
    let mut i = 0;
    let offset = 0.1f32;
    let mut y = -10i32;
    while y < 10 {
        let mut x = -10i32;
        while x < 10 {
            positions[i] = Vec2::new(x as f32 / 10.0 + offset, y as f32 / 10.0 + offset);
            i += 1;
            x += 2;
        }
        y += 2;
    }

    let colors: [Vec3; 10] = [
        Vec3::new(0.783099, 0.394383, 0.840188),
        Vec3::new(0.197551, 0.911647, 0.798440),
        Vec3::new(0.277775, 0.768230, 0.335223),
        Vec3::new(0.628871, 0.477397, 0.553970),
        Vec3::new(0.952230, 0.513401, 0.364784),
        Vec3::new(0.717297, 0.635712, 0.916195),
        Vec3::new(0.016301, 0.606969, 0.141603),
        Vec3::new(0.804177, 0.137232, 0.242887),
        Vec3::new(0.129790, 0.400944, 0.156679),
        Vec3::new(0.218257, 0.998924, 0.108809),
    ];

    let base_instance: GLuint = 20;
    let mut inst_colors = [Vec3::new(0.0, 0.0, 0.0); 100];
    for i in 0..100 {
        inst_colors[i] = colors[(i + (base_instance as usize / 10)) % 10];
    }

    if argc < 3 {
        let pos_buf = ctx.gl_gen_buffers(1)[0];
        ctx.gl_bind_buffer(GL_ARRAY_BUFFER, pos_buf);
        let _ = ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&positions), GL_STATIC_DRAW);
        ctx.gl_enable_vertex_attrib_array(1);
        ctx.gl_vertex_attrib_pointer(1, 2, GL_FLOAT, false, 0, 0);
        ctx.gl_vertex_attrib_divisor(1, 1);

        let col_buf = ctx.gl_gen_buffers(1)[0];
        ctx.gl_bind_buffer(GL_ARRAY_BUFFER, col_buf);
        let _ = ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&inst_colors), GL_STATIC_DRAW);
        ctx.gl_enable_vertex_attrib_array(2);
        ctx.gl_vertex_attrib_pointer(2, 3, GL_FLOAT, false, 0, 0);
        ctx.gl_vertex_attrib_divisor(2, 10);

        setup_vbo(ctx, &points);
        ctx.gl_enable_vertex_attrib_array(0);
        ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, 0);
    } else {
        ctx.gl_bind_buffer(GL_ARRAY_BUFFER, 0);
        ctx.gl_enable_vertex_attrib_array(1);
        ctx.gl_vertex_attrib_pointer(1, 2, GL_FLOAT, false, 0, positions.as_ptr() as GLsizeiptr);
        ctx.gl_vertex_attrib_divisor(1, 1);
        ctx.gl_enable_vertex_attrib_array(2);
        ctx.gl_vertex_attrib_pointer(2, 3, GL_FLOAT, false, 0, inst_colors.as_ptr() as GLsizeiptr);
        ctx.gl_vertex_attrib_divisor(2, 10);
        ctx.gl_enable_vertex_attrib_array(0);
        ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, points.as_ptr() as GLsizeiptr);
    }

    if argc == 1 {
        setup_ebo(ctx, &indices);
    }

    let flat3: [GLenum; 3] = [PGL_FLAT; 3];
    let myshader = ctx.pgl_create_program(instancing_vs, instancing_fs, 3, &flat3, false);
    ctx.gl_use_program(myshader);
    ctx.pgl_set_uniform(core::ptr::null_mut());

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);

    match argc {
        0 => ctx.gl_draw_arrays_instanced_base_instance(GL_TRIANGLES, 0, 3, 60, base_instance),
        1 => ctx.gl_draw_elements_instanced_base_instance(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0, 60, base_instance),
        _ => {
            ctx.gl_bind_buffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            ctx.gl_draw_elements_instanced_base_instance(GL_TRIANGLES, 3, GL_UNSIGNED_INT, indices.as_ptr() as usize, 60, base_instance);
        }
    }
}

#[test] fn baseinstance_arrays() { run_test("baseinstance_arrays", |ctx| test_baseinstance_impl(ctx, 0)); }
#[test] fn baseinstance_elements() { run_test("baseinstance_elements", |ctx| test_baseinstance_impl(ctx, 1)); }
#[test] fn baseinstance_elements2() { run_test("baseinstance_elements2", |ctx| test_baseinstance_impl(ctx, 2)); }
#[test] fn baseinstance_elements3() { run_test("baseinstance_elements3", |ctx| test_baseinstance_impl(ctx, 3)); }

// ============================================================================
// Test: instanceid
// ============================================================================
#[repr(C)]
struct InstanceIdUniforms {
    inst_offsets: *const Vec2,
    inst_colors: *const Vec3,
}

unsafe extern "C" fn glinstanceid_vs(
    _vs_output: *mut f32, vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let u = &*(uniforms as *const InstanceIdUniforms);
    let mut vert = *vertex_attribs.add(0);
    let offset = *u.inst_offsets.add((*builtins).gl_InstanceID as usize);
    vert.x += offset.x;
    vert.y += offset.y;
    (*builtins).gl_Position = vert;
}

unsafe extern "C" fn glinstanceid_fs(
    _fs_input: *mut f32, builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let u = &*(uniforms as *const InstanceIdUniforms);
    let color = *u.inst_colors.add((*builtins).gl_InstanceID as usize / 5);
    (*builtins).gl_FragColor = Vec4::new(color.x, color.y, color.z, 1.0);
}

#[test]
fn instanceid() {
    run_test("instanceid", |ctx| {
        let points: [f32; 9] = [-0.05,-0.05,0.0, 0.05,-0.05,0.0, 0.0,0.05,0.0];

        let mut positions = [Vec2::new(0.0, 0.0); 100];
        let mut i = 0;
        let offset = 0.1f32;
        let mut y = -10i32;
        while y < 10 {
            let mut x = -10i32;
            while x < 10 {
                positions[i] = Vec2::new(x as f32 / 10.0 + offset, y as f32 / 10.0 + offset);
                i += 1;
                x += 2;
            }
            y += 2;
        }

        #[rustfmt::skip]
        let inst_colors: [Vec3; 20] = [
            Vec3::new(0.783099, 0.394383, 0.840188), Vec3::new(0.197551, 0.911647, 0.798440),
            Vec3::new(0.277775, 0.768230, 0.335223), Vec3::new(0.628871, 0.477397, 0.553970),
            Vec3::new(0.952230, 0.513401, 0.364784), Vec3::new(0.717297, 0.635712, 0.916195),
            Vec3::new(0.016301, 0.606969, 0.141603), Vec3::new(0.804177, 0.137232, 0.242887),
            Vec3::new(0.129790, 0.400944, 0.156679), Vec3::new(0.218257, 0.998924, 0.108809),
            Vec3::new(0.612640, 0.839112, 0.512932), Vec3::new(0.524287, 0.637552, 0.296032),
            Vec3::new(0.292517, 0.972775, 0.493583), Vec3::new(0.769914, 0.526745, 0.771358),
            Vec3::new(0.283315, 0.891529, 0.400229), Vec3::new(0.919026, 0.807725, 0.352458),
            Vec3::new(0.525995, 0.949327, 0.069755), Vec3::new(0.663227, 0.192214, 0.086056),
            Vec3::new(0.064171, 0.348893, 0.890233), Vec3::new(0.063096, 0.457702, 0.020023),
        ];

        let mut the_uniforms = InstanceIdUniforms {
            inst_offsets: positions.as_ptr(),
            inst_colors: inst_colors.as_ptr(),
        };

        setup_vbo(ctx, &points);
        ctx.gl_enable_vertex_attrib_array(0);
        ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, 0);

        let empty: [GLenum; 0] = [];
        let myshader = ctx.pgl_create_program(glinstanceid_vs, glinstanceid_fs, 0, &empty, false);
        ctx.gl_use_program(myshader);
        ctx.pgl_set_uniform(&mut the_uniforms as *mut InstanceIdUniforms as *mut c_void);

        ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);
        ctx.gl_draw_arrays_instanced(GL_TRIANGLES, 0, 3, 100);
    });
}

// ============================================================================
// Test: multidraw (argc=0..1)
// ============================================================================
fn test_multidraw_impl(ctx: &mut GlContext, argc: i32) {
    let mut tri_strips: Vec<Vec3> = Vec::new();
    let mut colors: Vec<Vec3> = Vec::new();
    let mut strip_elems: Vec<GLuint> = Vec::new();

    let sq_dim = 20;
    let mut firsts: Vec<GLint> = Vec::new();
    let mut first_elems: Vec<GLintptr> = Vec::new();
    let mut counts: Vec<GLsizei> = Vec::new();

    let cols = 25;
    let rows = 25;
    let offset_x = 10.0f32;
    let offset_y = 10.0f32;

    for j in 0..rows {
        for i in 0..cols {
            firsts.push(tri_strips.len() as GLint);
            first_elems.push((strip_elems.len() * std::mem::size_of::<GLuint>()) as GLintptr);
            counts.push(4);

            let x = i * (sq_dim + 5);
            let y = j * (sq_dim + 5);
            tri_strips.push(Vec3::new((x as f32) + offset_x, (y as f32) + offset_y, 0.0));
            tri_strips.push(Vec3::new((x as f32) + offset_x, (y + sq_dim) as f32 + offset_y, 0.0));
            tri_strips.push(Vec3::new((x + sq_dim) as f32 + offset_x, (y as f32) + offset_y, 0.0));
            tri_strips.push(Vec3::new((x + sq_dim) as f32 + offset_x, (y + sq_dim) as f32 + offset_y, 0.0));

            colors.push(Vec3::new(1.0, 0.0, 0.0));
            colors.push(Vec3::new(0.0, 1.0, 0.0));
            colors.push(Vec3::new(0.0, 0.0, 1.0));
            colors.push(Vec3::new(0.0, 0.0, 0.0));

            let base = (j * cols + i) as GLuint * 4;
            strip_elems.push(base);
            strip_elems.push(base + 1);
            strip_elems.push(base + 2);
            strip_elems.push(base + 3);
        }
    }

    let sq_buf = ctx.gl_gen_buffers(1)[0];
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, sq_buf);
    let _ = ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&tri_strips), GL_STATIC_DRAW);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);

    let col_buf = ctx.gl_gen_buffers(1)[0];
    ctx.gl_bind_buffer(GL_ARRAY_BUFFER, col_buf);
    let _ = ctx.gl_buffer_data(GL_ARRAY_BUFFER, as_bytes(&colors), GL_STATIC_DRAW);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_COLOR, 3, GL_FLOAT, false, 0, 0);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_COLOR);

    let elem_buf = ctx.gl_gen_buffers(1)[0];
    ctx.gl_bind_buffer(GL_ELEMENT_ARRAY_BUFFER, elem_buf);
    let _ = ctx.gl_buffer_data(GL_ELEMENT_ARRAY_BUFFER, as_bytes(&strip_elems), GL_STATIC_DRAW);

    let std_shaders = pgl_init_std_shaders(ctx);
    ctx.gl_use_program(std_shaders[PGL_SHADER_SHADED]);

    let mut u = PglUniforms::default();
    ctx.pgl_set_uniform(&mut u as *mut PglUniforms as *mut c_void);
    u.mvp_mat = make_orthographic_m4(0.0, (WIDTH - 1) as f32, 0.0, (HEIGHT - 1) as f32, 1.0, -1.0);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);

    if argc == 0 {
        ctx.gl_multi_draw_arrays(GL_TRIANGLE_STRIP, &firsts[..300], &counts[..300]);
    } else {
        let elem_offsets: Vec<usize> = first_elems.iter().map(|&e| e as usize).collect();
        ctx.gl_multi_draw_elements(GL_TRIANGLE_STRIP, &counts, GL_UNSIGNED_INT, &elem_offsets);
    }
}

#[test] fn multidraw_arrays() { run_test("multidraw_arrays", |ctx| test_multidraw_impl(ctx, 0)); }
#[test] fn multidraw_elements() { run_test("multidraw_elements", |ctx| test_multidraw_impl(ctx, 1)); }

// ============================================================================
// Test: scissoring_test1 (argc=0..4)
// ============================================================================
fn scissoring_test1_impl(ctx: &mut GlContext, argc: i32) {
    #[rustfmt::skip]
    let points: [f32; 45] = [
        -0.5, -0.5, -0.1,
         0.5, -0.5, -0.1,
         0.0,  0.5, -0.1,

        -0.5, -0.5, -0.3,
         0.5, -0.5, -0.3,
         0.0,  0.5, -0.3,

        -0.5, -0.5, -0.5,
         0.5, -0.5, -0.5,
         0.0,  0.5, -0.5,

        -0.5, -0.5, -0.7,
         0.5, -0.5, -0.7,
         0.0,  0.5, -0.7,

        -0.5, -0.5, -0.9,
         0.5, -0.5, -0.9,
         0.0,  0.5, -0.9,
    ];

    match argc {
        1 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_LINE); }
        2 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_POINT); }
        3 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_LINE); ctx.gl_line_width(8.0); }
        4 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_POINT); ctx.gl_point_size(8.0); }
        _ => {}
    }

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    let std_shaders = pgl_init_std_shaders(ctx);
    ctx.gl_use_program(std_shaders[PGL_SHADER_IDENTITY]);

    let mut the_uniforms = PglUniforms::default();
    ctx.pgl_set_uniform(&mut the_uniforms as *mut PglUniforms as *mut c_void);

    ctx.gl_enable(GL_SCISSOR_TEST);
    ctx.gl_enable(GL_DEPTH_TEST);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Cut off all sides
    ctx.gl_scissor(220, 220, 200, 200);
    the_uniforms.color = Vec4::new(1.0, 0.0, 0.0, 1.0);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

    // Allow right side
    ctx.gl_scissor(420, 220, 500, 200);
    the_uniforms.color = Vec4::new(0.0, 1.0, 0.0, 1.0);
    ctx.gl_draw_arrays(GL_TRIANGLES, 3, 3);

    // Allow bottom
    ctx.gl_scissor(220, 0, 200, 220);
    the_uniforms.color = Vec4::new(0.0, 0.0, 1.0, 1.0);
    ctx.gl_draw_arrays(GL_TRIANGLES, 6, 3);

    // Allow left
    ctx.gl_scissor(0, 220, 220, 400);
    the_uniforms.color = Vec4::new(1.0, 0.0, 1.0, 1.0);
    ctx.gl_draw_arrays(GL_TRIANGLES, 9, 3);

    // Allow top
    ctx.gl_scissor(220, 420, 200, 550);
    the_uniforms.color = Vec4::new(0.0, 1.0, 1.0, 1.0);
    ctx.gl_draw_arrays(GL_TRIANGLES, 12, 3);
}

#[test] fn scissor1_fill() { run_test("scissor1_fill", |ctx| scissoring_test1_impl(ctx, 0)); }
#[test] fn scissor1_ln() { run_test("scissor1_ln", |ctx| scissoring_test1_impl(ctx, 1)); }
#[test] fn scissor1_pnt() { run_test("scissor1_pnt", |ctx| scissoring_test1_impl(ctx, 2)); }
#[test] fn scissor1_ln8() { run_test("scissor1_ln8", |ctx| scissoring_test1_impl(ctx, 3)); }
#[test] fn scissor1_pnt8() { run_test("scissor1_pnt8", |ctx| scissoring_test1_impl(ctx, 4)); }

// ============================================================================
// Test: scissoring_test2 (argc=0..4)
// ============================================================================
fn scissoring_test2_impl(ctx: &mut GlContext, argc: i32) {
    #[rustfmt::skip]
    let points: [f32; 45] = [
        -0.5, -0.5, 0.9,
         0.5, -0.5, 0.9,
         0.0,  0.5, 0.9,

        -0.5, -0.5, 0.7,
         0.5, -0.5, 0.7,
         0.0,  0.5, 0.7,

        -0.5, -0.5, 0.5,
         0.5, -0.5, 0.5,
         0.0,  0.5, 0.5,

        -0.5, -0.5, 0.3,
         0.5, -0.5, 0.3,
         0.0,  0.5, 0.3,

        -0.5, -0.5, 0.1,
         0.5, -0.5, 0.1,
         0.0,  0.5, 0.1,
    ];

    match argc {
        1 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_LINE); }
        2 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_POINT); }
        3 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_LINE); ctx.gl_line_width(8.0); }
        4 => { ctx.gl_polygon_mode(GL_FRONT_AND_BACK, GL_POINT); ctx.gl_point_size(8.0); }
        _ => {}
    }

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    let std_shaders = pgl_init_std_shaders(ctx);
    ctx.gl_use_program(std_shaders[PGL_SHADER_IDENTITY]);

    let mut the_uniforms = PglUniforms::default();
    ctx.pgl_set_uniform(&mut the_uniforms as *mut PglUniforms as *mut c_void);

    ctx.gl_enable(GL_SCISSOR_TEST);
    ctx.gl_enable(GL_DEPTH_TEST);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Only top
    ctx.gl_scissor(0, 420, 640, 550);
    the_uniforms.color = Vec4::new(1.0, 0.0, 0.0, 1.0);
    ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

    // Only right side
    ctx.gl_scissor(420, 0, 500, 640);
    the_uniforms.color = Vec4::new(0.0, 1.0, 0.0, 1.0);
    ctx.gl_draw_arrays(GL_TRIANGLES, 3, 3);

    // Only bottom
    ctx.gl_scissor(0, 0, 640, 220);
    the_uniforms.color = Vec4::new(0.0, 0.0, 1.0, 1.0);
    ctx.gl_draw_arrays(GL_TRIANGLES, 6, 3);

    // Only left
    ctx.gl_scissor(0, 0, 220, 640);
    the_uniforms.color = Vec4::new(1.0, 0.0, 1.0, 1.0);
    ctx.gl_draw_arrays(GL_TRIANGLES, 9, 3);

    // Cut off all sides
    ctx.gl_scissor(220, 220, 200, 200);
    the_uniforms.color = Vec4::new(0.0, 1.0, 1.0, 1.0);
    ctx.gl_draw_arrays(GL_TRIANGLES, 12, 3);
}

#[test] fn scissor2_fill() { run_test("scissor2_fill", |ctx| scissoring_test2_impl(ctx, 0)); }
#[test] fn scissor2_ln() { run_test("scissor2_ln", |ctx| scissoring_test2_impl(ctx, 1)); }
#[test] fn scissor2_pnt() { run_test("scissor2_pnt", |ctx| scissoring_test2_impl(ctx, 2)); }
#[test] fn scissor2_ln8() { run_test("scissor2_ln8", |ctx| scissoring_test2_impl(ctx, 3)); }
#[test] fn scissor2_pnt8() { run_test("scissor2_pnt8", |ctx| scissoring_test2_impl(ctx, 4)); }

// ============================================================================
// Test: scissoring_test3 (scissor_clear_color)
// ============================================================================
#[test]
fn scissor_clear_color() {
    run_test("scissor_clear_color", |ctx| {
        #[rustfmt::skip]
        let points: [f32; 9] = [
            -0.5, -0.5, 0.0,
             0.5, -0.5, 0.0,
             0.0,  0.5, 0.0,
        ];

        setup_vbo(ctx, &points);
        ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
        ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

        let std_shaders = pgl_init_std_shaders(ctx);
        ctx.gl_use_program(std_shaders[PGL_SHADER_IDENTITY]);

        let mut the_uniforms = PglUniforms::default();
        ctx.pgl_set_uniform(&mut the_uniforms as *mut PglUniforms as *mut c_void);

        ctx.gl_enable(GL_SCISSOR_TEST);

        ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);

        the_uniforms.color = Vec4::new(1.0, 0.0, 0.0, 1.0);
        ctx.gl_draw_arrays(GL_TRIANGLES, 0, 3);

        ctx.gl_scissor(WIDTH / 2, 0, WIDTH / 2, HEIGHT);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);

        ctx.gl_disable(GL_SCISSOR_TEST);
        ctx.gl_scissor(0, HEIGHT / 2, WIDTH, HEIGHT / 2);
        ctx.gl_enable(GL_SCISSOR_TEST);
        ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    });
}

// ============================================================================
// Test: scissoring_test4 (argc=0..2)
// ============================================================================
fn scissoring_test4_impl(ctx: &mut GlContext, argc: i32) {
    #[rustfmt::skip]
    let points_n_lines: [f32; 30] = [
        // test -x and +y
        -1.1,  0.6, 0.0,
        -0.6,  1.1, 0.0,

        // +x and -y
         1.1, -0.6, 0.0,
         0.6, -1.1, 0.0,

        // more clipping
        -1.0,  0.9, 0.0,
         1.0, -0.9, 0.0,

        // points below
        -0.9,   0.5, 0.0,
         0.9,   0.5, 0.0,

        -1.02, -0.5, 0.0,
         1.02, -0.5, 0.0,
    ];

    match argc {
        1 => { ctx.gl_point_size(8.0); ctx.gl_line_width(8.0); }
        2 => { ctx.gl_point_size(32.0); ctx.gl_line_width(32.0); }
        _ => {}
    }

    setup_vbo(ctx, &points_n_lines);
    ctx.gl_enable_vertex_attrib_array(0);
    ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, 0);

    ctx.gl_clear_color(0.0, 0.0, 0.0, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);

    // No shader or uniform needed - uses default shader 0
    ctx.gl_enable(GL_SCISSOR_TEST);
    ctx.gl_scissor(
        (WIDTH as f32 / 20.0) as i32,
        (HEIGHT as f32 / 20.0) as i32,
        (9.0 * WIDTH as f32 / 10.0) as i32,
        (9.0 * HEIGHT as f32 / 10.0) as i32,
    );

    ctx.gl_draw_arrays(GL_LINES, 0, 6);
    ctx.gl_draw_arrays(GL_POINTS, 6, 4);
}

#[test] fn scissor4_pnt_ln() { run_test("scissor4_pnt_ln", |ctx| scissoring_test4_impl(ctx, 0)); }
#[test] fn scissor4_pnt_ln8() { run_test("scissor4_pnt_ln8", |ctx| scissoring_test4_impl(ctx, 1)); }
#[test] fn scissor4_pnt_ln32() { run_test("scissor4_pnt_ln32", |ctx| scissoring_test4_impl(ctx, 2)); }

// ============================================================================
// Texture2D filtering tests (from test_texturing.cpp test_tex2D_filtering)
// ============================================================================

fn test_tex2d_filtering_impl(ctx: &mut GlContext, argc: i32) {
    let points: [f32; 12] = [
        -0.8,  0.8, -0.1,
        -0.8, -0.8, -0.1,
         0.8,  0.8, -0.1,
         0.8, -0.8, -0.1,
    ];
    let tex_coords: [f32; 8] = [
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ];
    let test_texture: [u8; 16] = [
        255, 255, 255, 255,
          0,   0,   0, 255,
          0,   0,   0, 255,
        255, 255, 255, 255,
    ];

    let textures = ctx.gl_gen_textures(1);
    ctx.gl_bind_texture(GL_TEXTURE_2D, textures[0]).unwrap();
    ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as GLint);
    ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
        if argc == 0 { GL_NEAREST as GLint } else { GL_LINEAR as GLint });
    ctx.gl_tex_image_2d(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA as GLint, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, Some(&test_texture));

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    setup_vbo(ctx, &tex_coords);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_TEXCOORD0);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_TEXCOORD0, 2, GL_FLOAT, false, 0, 0);

    let std_shaders = pgl_init_std_shaders(ctx);
    ctx.gl_use_program(std_shaders[PGL_SHADER_TEX_REPLACE]);

    let mut the_uniforms = PglUniforms::default();
    the_uniforms.mvp_mat = Mat4::identity();
    the_uniforms.tex0 = textures[0];
    the_uniforms.ctx = ctx as *const GlContext;
    ctx.pgl_set_uniform(&mut the_uniforms as *mut PglUniforms as *mut c_void);

    ctx.gl_clear_color(0.25, 0.25, 0.25, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 0, 4);
}

#[test] fn texture2D_nearest() { run_test("texture2D_nearest", |ctx| test_tex2d_filtering_impl(ctx, 0)); }
#[test] fn texture2D_linear() { run_test("texture2D_linear", |ctx| test_tex2d_filtering_impl(ctx, 1)); }

// ============================================================================
// Texture2D wrap mode tests (from test_texturing.cpp test_tex2D_wrap_modes)
// ============================================================================

fn test_tex2d_wrap_modes_impl(ctx: &mut GlContext, argc: i32) {
    let points: [f32; 12] = [
        -0.8,  0.8, -0.1,
        -0.8, -0.8, -0.1,
         0.8,  0.8, -0.1,
         0.8, -0.8, -0.1,
    ];
    let tex_coords: [f32; 8] = [
        -1.0, -1.0,
        -1.0,  2.0,
         2.0, -1.0,
         2.0,  2.0,
    ];
    let test_texture: [u8; 16] = [
        255, 255, 255, 255,
          0,   0,   0, 255,
          0,   0,   0, 255,
        255, 255, 255, 255,
    ];

    let textures = ctx.gl_gen_textures(1);
    ctx.gl_bind_texture(GL_TEXTURE_2D, textures[0]).unwrap();
    ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as GLint);
    ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as GLint);

    match argc {
        0 => {
            ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT as GLint);
            ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT as GLint);
        }
        1 => {
            ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE as GLint);
            ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE as GLint);
        }
        2 => {
            ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT as GLint);
            ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT as GLint);
        }
        3 => {
            ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER as GLint);
            ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER as GLint);
            let green: [GLfloat; 4] = [0.0, 1.0, 0.0, 1.0];
            ctx.gl_tex_parameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, &green);
        }
        _ => {}
    }

    ctx.gl_tex_image_2d(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA as GLint, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, Some(&test_texture));

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    setup_vbo(ctx, &tex_coords);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_TEXCOORD0);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_TEXCOORD0, 2, GL_FLOAT, false, 0, 0);

    let std_shaders = pgl_init_std_shaders(ctx);
    ctx.gl_use_program(std_shaders[PGL_SHADER_TEX_REPLACE]);

    let mut the_uniforms = PglUniforms::default();
    the_uniforms.mvp_mat = Mat4::identity();
    the_uniforms.tex0 = textures[0];
    the_uniforms.ctx = ctx as *const GlContext;
    ctx.pgl_set_uniform(&mut the_uniforms as *mut PglUniforms as *mut c_void);

    ctx.gl_clear_color(0.25, 0.25, 0.25, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 0, 4);
}

#[test] fn texture2D_repeat() { run_test("texture2D_repeat", |ctx| test_tex2d_wrap_modes_impl(ctx, 0)); }
#[test] fn texture2D_clamp2edge() { run_test("texture2D_clamp2edge", |ctx| test_tex2d_wrap_modes_impl(ctx, 1)); }
#[test] fn texture2D_mirroredrepeat() { run_test("texture2D_mirroredrepeat", |ctx| test_tex2d_wrap_modes_impl(ctx, 2)); }
#[test] fn texture2D_clamp2border() { run_test("texture2D_clamp2border", |ctx| test_tex2d_wrap_modes_impl(ctx, 3)); }

// ============================================================================
// Texture1D tests (from test_tex1D.cpp test_texture1D)
// ============================================================================

#[repr(C)]
struct Tex1DUniforms {
    tex: GLuint,
    ctx: *const GlContext,
}

unsafe extern "C" fn tex1d_replace_vs(
    vs_output: *mut f32, vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins, _uniforms: *mut c_void,
) {
    *vs_output = (*vertex_attribs.add(2)).x; // tex_coord
    (*builtins).gl_Position = *vertex_attribs.add(0);
}

unsafe extern "C" fn tex1d_replace_fs(
    fs_input: *mut f32, builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let tex_coords_x = *fs_input;
    let u = &*(uniforms as *const Tex1DUniforms);
    (*builtins).gl_FragColor = (*u.ctx).texture1d(u.tex, tex_coords_x);
}

fn test_texture1d_impl(ctx: &mut GlContext, argc: i32) {
    let points: [f32; 12] = [
        -0.8,  0.8, -0.1,
        -0.8, -0.8, -0.1,
         0.8,  0.8, -0.1,
         0.8, -0.8, -0.1,
    ];
    let tex_coords1: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
    let tex_coords2: [f32; 4] = [0.0, 0.0, 2.0, 2.0];

    let test_texture: [u8; 8] = [
        255, 255, 255, 255,
          0,   0,   0, 255,
    ];

    let textures = ctx.gl_gen_textures(1);

    ctx.gl_bind_texture(GL_TEXTURE_1D, textures[0]).unwrap();
    ctx.gl_tex_parameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as GLint);
    ctx.gl_tex_parameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER,
        if argc != 1 { GL_NEAREST as GLint } else { GL_LINEAR as GLint });
    ctx.gl_tex_image_1d(GL_TEXTURE_1D, 0, GL_COMPRESSED_RGBA as GLint, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, Some(&test_texture));

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(0);
    ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, 0);

    let tex_coords: &[f32] = if argc >= 2 { &tex_coords2 } else { &tex_coords1 };
    setup_vbo(ctx, tex_coords);

    // default wrap is GL_REPEAT for argc == 2
    if argc == 3 {
        ctx.gl_tex_parameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE as GLint);
    } else if argc == 4 {
        ctx.gl_tex_parameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT as GLint);
    }

    ctx.gl_enable_vertex_attrib_array(2);
    ctx.gl_vertex_attrib_pointer(2, 1, GL_FLOAT, false, 0, 0);

    let smooth1: [GLenum; 1] = [PGL_SMOOTH];
    let texture_shader = ctx.pgl_create_program(tex1d_replace_vs, tex1d_replace_fs, 1, &smooth1, false);
    ctx.gl_use_program(texture_shader);

    let mut the_uniforms = Tex1DUniforms { tex: textures[0], ctx: ctx as *const GlContext };
    ctx.pgl_set_uniform(&mut the_uniforms as *mut Tex1DUniforms as *mut c_void);

    ctx.gl_clear_color(0.25, 0.25, 0.25, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 0, 4);
}

#[test] fn texture1D_nearest() { run_test("texture1D_nearest", |ctx| test_texture1d_impl(ctx, 0)); }
#[test] fn texture1D_linear() { run_test("texture1D_linear", |ctx| test_texture1d_impl(ctx, 1)); }
#[test] fn texture1D_repeat() { run_test("texture1D_repeat", |ctx| test_texture1d_impl(ctx, 2)); }
#[test] fn texture1D_clamp2edge() { run_test("texture1D_clamp2edge", |ctx| test_texture1d_impl(ctx, 3)); }
#[test] fn texture1D_mirroredrepeat() { run_test("texture1D_mirroredrepeat", |ctx| test_texture1d_impl(ctx, 4)); }

// ============================================================================
// pglteximage2D test (from pglteximage2D.cpp test_pglteximage2D)
// ============================================================================

fn test_pglteximage2d_impl(ctx: &mut GlContext) {
    let points: [f32; 12] = [
        -0.8,  0.8, -0.1,
        -0.8, -0.8, -0.1,
         0.8,  0.8, -0.1,
         0.8, -0.8, -0.1,
    ];
    let tex_coords: [f32; 8] = [
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ];
    let mut test_texture: [u8; 16] = [
        255, 255, 255, 255,
          0,   0,   0, 255,
          0,   0,   0, 255,
        255, 255, 255, 255,
    ];

    let textures = ctx.gl_gen_textures(1);
    ctx.gl_bind_texture(GL_TEXTURE_2D, textures[0]).unwrap();
    ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as GLint);
    ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as GLint);
    // In C, pglTexImage2D stores a raw pointer (no copy), so modifications after
    // the call are visible during rendering. In Rust, pgl_tex_image_2d copies the
    // data, so we apply the modification BEFORE uploading to match the C output.
    test_texture[4] = 255; // test_texture[1].r = 255 in C

    ctx.pgl_tex_image_2d(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA as GLint, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, test_texture.as_mut_ptr());

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    setup_vbo(ctx, &tex_coords);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_TEXCOORD0);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_TEXCOORD0, 2, GL_FLOAT, false, 0, 0);

    let std_shaders = pgl_init_std_shaders(ctx);
    ctx.gl_use_program(std_shaders[PGL_SHADER_TEX_REPLACE]);

    let mut the_uniforms = PglUniforms::default();
    the_uniforms.mvp_mat = Mat4::identity();
    the_uniforms.tex0 = textures[0];
    the_uniforms.ctx = ctx as *const GlContext;
    ctx.pgl_set_uniform(&mut the_uniforms as *mut PglUniforms as *mut c_void);

    ctx.gl_clear_color(0.25, 0.25, 0.25, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 0, 4);
}

#[test] fn pglteximage2D() { run_test("pglteximage2D", |ctx| test_pglteximage2d_impl(ctx)); }

// ============================================================================
// pglteximage1D test (from pglteximage1D.cpp test_pglteximage1D)
// ============================================================================

fn test_pglteximage1d_impl(ctx: &mut GlContext) {
    let points: [f32; 12] = [
        -0.8,  0.8, -0.1,
        -0.8, -0.8, -0.1,
         0.8,  0.8, -0.1,
         0.8, -0.8, -0.1,
    ];
    let tex_coords: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

    let mut test_texture: [u8; 8] = [
        255, 255, 255, 255,
          0,   0,   0, 255,
    ];

    let textures = ctx.gl_gen_textures(1);
    ctx.gl_bind_texture(GL_TEXTURE_1D, textures[0]).unwrap();
    ctx.gl_tex_parameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as GLint);
    ctx.gl_tex_parameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as GLint);
    // In C, pglTexImage1D stores a raw pointer (no copy), so modifications after
    // the call are visible during rendering. In Rust, pgl_tex_image_1d copies the
    // data, so we apply the modification BEFORE uploading to match the C output.
    test_texture[6] = 255; // test_texture[1].b = 255 in C

    ctx.pgl_tex_image_1d(GL_TEXTURE_1D, 0, GL_COMPRESSED_RGBA as GLint, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, test_texture.as_mut_ptr());

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(0);
    ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, 0);

    setup_vbo(ctx, &tex_coords);
    ctx.gl_enable_vertex_attrib_array(2);
    ctx.gl_vertex_attrib_pointer(2, 1, GL_FLOAT, false, 0, 0);

    let smooth1: [GLenum; 1] = [PGL_SMOOTH];
    let texture_shader = ctx.pgl_create_program(tex1d_replace_vs, tex1d_replace_fs, 1, &smooth1, false);
    ctx.gl_use_program(texture_shader);

    let mut the_uniforms = Tex1DUniforms { tex: textures[0], ctx: ctx as *const GlContext };
    ctx.pgl_set_uniform(&mut the_uniforms as *mut Tex1DUniforms as *mut c_void);

    ctx.gl_clear_color(0.25, 0.25, 0.25, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 0, 4);
}

#[test] fn pglteximage1D() { run_test("pglteximage1D", |ctx| test_pglteximage1d_impl(ctx)); }

// ============================================================================
// Unpack alignment test (from test_unpackalignment.cpp)
// ============================================================================

#[repr(C)]
struct UnpackUniforms {
    tex: GLuint,
    ctx: *const GlContext,
}

unsafe extern "C" fn unpack_tex_replace_vs(
    vs_output: *mut f32, vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins, _uniforms: *mut c_void,
) {
    let tc = *vertex_attribs.add(2);
    *(vs_output as *mut Vec2) = Vec2::new(tc.x, tc.y);
    (*builtins).gl_Position = *vertex_attribs.add(0);
}

unsafe extern "C" fn unpack_tex_replace_fs(
    fs_input: *mut f32, builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let tex_coords = *(fs_input as *const Vec2);
    let u = &*(uniforms as *const UnpackUniforms);
    (*builtins).gl_FragColor = (*u.ctx).texture2d(u.tex, tex_coords.x, tex_coords.y);
}

fn test_unpackalignment_impl(ctx: &mut GlContext) {
    let points: [f32; 12] = [
        -0.8,  0.8, -0.1,
        -0.8, -0.8, -0.1,
         0.8,  0.8, -0.1,
         0.8, -0.8, -0.1,
    ];
    let tex_coords: [f32; 8] = [
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ];

    // 3x3 texture with padding bytes for alignment=8
    // Each row is 3 pixels * 4 bytes = 12 bytes, padded to 16 bytes (next multiple of 8)
    let test_texture: [u8; 48] = [
        255, 255, 255, 255,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0, 255,
          0,   0,   0, 255, 255, 255, 255, 255,   0,   0,   0, 255,   0, 255,   0, 255,
        255, 255, 255, 255,   0,   0,   0, 255, 255, 255, 255, 255,   0,   0, 255, 255,
    ];

    let textures = ctx.gl_gen_textures(1);
    ctx.gl_bind_texture(GL_TEXTURE_2D, textures[0]).unwrap();
    ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as GLint);

    ctx.gl_pixel_storei(GL_UNPACK_ALIGNMENT, 8);
    ctx.gl_tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as GLint);
    ctx.gl_tex_image_2d(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA as GLint, 3, 3, 0, GL_RGBA, GL_UNSIGNED_BYTE, Some(&test_texture));

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(0);
    ctx.gl_vertex_attrib_pointer(0, 3, GL_FLOAT, false, 0, 0);

    setup_vbo(ctx, &tex_coords);
    ctx.gl_enable_vertex_attrib_array(2);
    ctx.gl_vertex_attrib_pointer(2, 2, GL_FLOAT, false, 0, 0);

    let smooth2: [GLenum; 2] = [PGL_SMOOTH; 2];
    let texture_shader = ctx.pgl_create_program(unpack_tex_replace_vs, unpack_tex_replace_fs, 2, &smooth2, false);
    ctx.gl_use_program(texture_shader);

    let mut the_uniforms = UnpackUniforms { tex: textures[0], ctx: ctx as *const GlContext };
    ctx.pgl_set_uniform(&mut the_uniforms as *mut UnpackUniforms as *mut c_void);

    ctx.gl_clear_color(0.25, 0.25, 0.25, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 0, 4);
}

#[test] fn unpack_alignment() { run_test("unpack_alignment", |ctx| test_unpackalignment_impl(ctx)); }

// ============================================================================
// Helper: load a PNG file as RGBA u8 bytes (for use as texture data)
// ============================================================================
fn load_png_as_rgba_bytes(path: &str) -> (Vec<u8>, u32, u32) {
    let (pixels, w, h) = load_png(path)
        .unwrap_or_else(|| panic!("Failed to load PNG: {}", path));
    // load_png returns u32 in ABGR format; convert back to RGBA bytes
    let mut bytes = Vec::with_capacity((w * h * 4) as usize);
    for &px in &pixels {
        let [r, g, b, a] = abgr_to_rgba(px);
        bytes.push(r);
        bytes.push(g);
        bytes.push(b);
        bytes.push(a);
    }
    (bytes, w, h)
}

// ============================================================================
// Texrect filtering tests (from test_texturing.cpp test_texrect_filtering)
// ============================================================================

fn test_texrect_filtering_impl(ctx: &mut GlContext, argc: i32) {
    let points: [f32; 12] = [
        -0.8,  0.8, -0.1,
        -0.8, -0.8, -0.1,
         0.8,  0.8, -0.1,
         0.8, -0.8, -0.1,
    ];
    let tex_coords: [f32; 8] = [
        0.0, 0.0,
        0.0, 511.0,
        511.0, 0.0,
        511.0, 511.0,
    ];

    let (tex_data, tw, th) = load_png_as_rgba_bytes("testing/media/tex04.png");

    let textures = ctx.gl_gen_textures(1);
    ctx.gl_bind_texture(GL_TEXTURE_RECTANGLE, textures[0]).unwrap();

    let magfilter = if argc != 0 { GL_LINEAR } else { GL_NEAREST };
    let wrapping = GL_REPEAT;

    ctx.gl_tex_parameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, wrapping as GLint);
    ctx.gl_tex_parameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, wrapping as GLint);
    ctx.gl_tex_parameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST as GLint);
    ctx.gl_tex_parameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, magfilter as GLint);

    let green: [GLfloat; 4] = [0.0, 1.0, 0.0, 1.0];
    ctx.gl_tex_parameterfv(GL_TEXTURE_RECTANGLE, GL_TEXTURE_BORDER_COLOR, &green);

    ctx.gl_pixel_storei(GL_UNPACK_ALIGNMENT, 1);
    ctx.gl_tex_image_2d(GL_TEXTURE_RECTANGLE, 0, GL_RGBA as GLint, tw as GLsizei, th as GLsizei, 0, GL_RGBA, GL_UNSIGNED_BYTE, Some(&tex_data));

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    setup_vbo(ctx, &tex_coords);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_TEXCOORD0);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_TEXCOORD0, 2, GL_FLOAT, false, 0, 0);

    let std_shaders = pgl_init_std_shaders(ctx);
    ctx.gl_use_program(std_shaders[PGL_SHADER_TEX_RECT_REPLACE]);

    let mut the_uniforms = PglUniforms::default();
    the_uniforms.mvp_mat = Mat4::identity();
    the_uniforms.tex0 = textures[0];
    the_uniforms.ctx = ctx as *const GlContext;
    ctx.pgl_set_uniform(&mut the_uniforms as *mut PglUniforms as *mut c_void);

    ctx.gl_clear_color(0.25, 0.25, 0.25, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 0, 4);
}

#[test] fn texrect_nearest() { run_test("texrect_nearest", |ctx| test_texrect_filtering_impl(ctx, 0)); }
#[test] fn texrect_linear() { run_test("texrect_linear", |ctx| test_texrect_filtering_impl(ctx, 1)); }

// ============================================================================
// Texrect wrap mode tests (from test_texturing.cpp test_texrect_wrap_modes)
// ============================================================================

fn test_texrect_wrap_modes_impl(ctx: &mut GlContext, argc: i32) {
    let points: [f32; 12] = [
        -0.8,  0.8, -0.1,
        -0.8, -0.8, -0.1,
         0.8,  0.8, -0.1,
         0.8, -0.8, -0.1,
    ];
    let tex_coords: [f32; 8] = [
        -512.0, -512.0,
        -512.0, 1024.0,
        1024.0, -512.0,
        1024.0, 1024.0,
    ];

    let (tex_data, tw, th) = load_png_as_rgba_bytes("testing/media/tex04.png");

    let textures = ctx.gl_gen_textures(1);
    ctx.gl_bind_texture(GL_TEXTURE_RECTANGLE, textures[0]).unwrap();

    let wrapping = match argc {
        0 => GL_REPEAT,
        1 => GL_CLAMP_TO_EDGE,
        2 => GL_MIRRORED_REPEAT,
        3 => GL_CLAMP_TO_BORDER,
        _ => GL_REPEAT,
    };

    ctx.gl_tex_parameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, wrapping as GLint);
    ctx.gl_tex_parameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, wrapping as GLint);
    ctx.gl_tex_parameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST as GLint);
    ctx.gl_tex_parameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST as GLint);

    let green: [GLfloat; 4] = [0.0, 1.0, 0.0, 1.0];
    ctx.gl_tex_parameterfv(GL_TEXTURE_RECTANGLE, GL_TEXTURE_BORDER_COLOR, &green);

    ctx.gl_pixel_storei(GL_UNPACK_ALIGNMENT, 1);
    ctx.gl_tex_image_2d(GL_TEXTURE_RECTANGLE, 0, GL_RGBA as GLint, tw as GLsizei, th as GLsizei, 0, GL_RGBA, GL_UNSIGNED_BYTE, Some(&tex_data));

    setup_vbo(ctx, &points);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_VERT);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_VERT, 3, GL_FLOAT, false, 0, 0);

    setup_vbo(ctx, &tex_coords);
    ctx.gl_enable_vertex_attrib_array(PGL_ATTR_TEXCOORD0);
    ctx.gl_vertex_attrib_pointer(PGL_ATTR_TEXCOORD0, 2, GL_FLOAT, false, 0, 0);

    let std_shaders = pgl_init_std_shaders(ctx);
    ctx.gl_use_program(std_shaders[PGL_SHADER_TEX_RECT_REPLACE]);

    let mut the_uniforms = PglUniforms::default();
    the_uniforms.mvp_mat = Mat4::identity();
    the_uniforms.tex0 = textures[0];
    the_uniforms.ctx = ctx as *const GlContext;
    ctx.pgl_set_uniform(&mut the_uniforms as *mut PglUniforms as *mut c_void);

    ctx.gl_clear_color(0.25, 0.25, 0.25, 1.0);
    ctx.gl_clear(GL_COLOR_BUFFER_BIT);
    ctx.gl_draw_arrays(GL_TRIANGLE_STRIP, 0, 4);
}

#[test] fn texrect_repeat() { run_test("texrect_repeat", |ctx| test_texrect_wrap_modes_impl(ctx, 0)); }
#[test] fn texrect_clamp2edge() { run_test("texrect_clamp2edge", |ctx| test_texrect_wrap_modes_impl(ctx, 1)); }
#[test] fn texrect_mirroredrepeat() { run_test("texrect_mirroredrepeat", |ctx| test_texrect_wrap_modes_impl(ctx, 2)); }
#[test] fn texrect_clamp2border() { run_test("texrect_clamp2border", |ctx| test_texrect_wrap_modes_impl(ctx, 3)); }

// ============================================================
// Math tests (ported from math_testing.cpp)
// ============================================================

fn cmp_m4(a: &Mat4, b: &Mat4, eps: f32) -> bool {
    for i in 0..16 {
        if (a.0[i] - b.0[i]).abs() >= eps {
            return false;
        }
    }
    true
}

#[test]
fn math_perspective() {
    use portablegl::math::make_perspective_m4;
    let fov = 45.0f32.to_radians();
    let aspect = 640.0 / 480.0;
    let near = 0.1f32;
    let far = 100.0f32;

    let m = make_perspective_m4(fov, aspect, near, far);

    // Reference values from GLM glm::perspective(fov, aspect, near, far)
    let t = near * (fov * 0.5).tan();
    let b = -t;
    let r = t * aspect;
    let l = -r;
    let expected = Mat4([
        2.0 * near / (r - l), 0.0,                  0.0,                            0.0,
        0.0,                  2.0 * near / (t - b),  0.0,                            0.0,
        0.0,                  0.0,                  -(far + near) / (far - near),   -1.0,
        0.0,                  0.0,                  -2.0 * far * near / (far - near), 0.0,
    ]);

    assert!(cmp_m4(&m, &expected, 1e-6), "perspective matrix mismatch");
}

#[test]
fn math_ortho_perspective_decomposition() {
    use portablegl::math::{make_perspective_m4, make_orthographic_m4, make_pers_m4, mult_m4_m4};
    let fov = 45.0f32.to_radians();
    let aspect = 640.0 / 480.0;
    let near = 0.1f32;
    let far = 100.0f32;

    let pers_proj = make_perspective_m4(fov, aspect, near, far);

    // Decompose: perspective = ortho * pers
    let pers_mat = make_pers_m4(near, far);
    let t = near * (fov * 0.5).tan();
    let b = -t;
    let l = b * aspect;
    let r = -l;
    let ortho = make_orthographic_m4(l, r, b, t, -near, -far);
    let result = mult_m4_m4(ortho, pers_mat);

    assert!(cmp_m4(&result, &pers_proj, 1e-6), "O*P != perspective projection");
}
