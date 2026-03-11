//! Internal rendering pipeline for the PortableGL Rust port.
//!
//! Contains the core rendering pipeline: vertex processing, clipping, rasterization
//! (triangle fill, line drawing, point drawing), fragment processing (depth test,
//! stencil, blending), and pixel output.

#![allow(
    non_upper_case_globals,
    non_snake_case,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::identity_op,
    dead_code
)]

#[cfg(feature = "no_std")]
use alloc::vec;
#[cfg(feature = "no_std")]
use alloc::vec::Vec;

use crate::float_math::F32Ext;
use crate::gl_context::*;
use crate::gl_types::*;
use crate::math::*;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CLIP_EPSILON: f32 = 1e-5;

/// Smallest depth increment for polygon offset units.
/// Matches the C PortableGL constant used in polygon offset calculations.
const POLYGON_OFFSET_UNIT_INCR: f32 = 0.000001;

// ---------------------------------------------------------------------------
// Pixel format masks (compile-time via cfg features)
// ---------------------------------------------------------------------------
//
// Each pixel format defines how RGBA bytes are packed into a u32 (or u16 for
// the 565 formats).  Only one pixel format feature should be enabled at a time.

// --- ABGR32 (default) ---
// Memory byte order: R G B A  =>  packed u32: 0xAABBGGRR
#[cfg(feature = "abgr32")]
pub const PGL_RED_SHIFT: u32 = 0;
#[cfg(feature = "abgr32")]
pub const PGL_GREEN_SHIFT: u32 = 8;
#[cfg(feature = "abgr32")]
pub const PGL_BLUE_SHIFT: u32 = 16;
#[cfg(feature = "abgr32")]
pub const PGL_ALPHA_SHIFT: u32 = 24;

// --- RGBA32 ---
// Packed u32: 0xRRGGBBAA
#[cfg(feature = "rgba32")]
pub const PGL_RED_SHIFT: u32 = 24;
#[cfg(feature = "rgba32")]
pub const PGL_GREEN_SHIFT: u32 = 16;
#[cfg(feature = "rgba32")]
pub const PGL_BLUE_SHIFT: u32 = 8;
#[cfg(feature = "rgba32")]
pub const PGL_ALPHA_SHIFT: u32 = 0;

// --- ARGB32 ---
// Packed u32: 0xAARRGGBB
#[cfg(feature = "argb32")]
pub const PGL_RED_SHIFT: u32 = 16;
#[cfg(feature = "argb32")]
pub const PGL_GREEN_SHIFT: u32 = 8;
#[cfg(feature = "argb32")]
pub const PGL_BLUE_SHIFT: u32 = 0;
#[cfg(feature = "argb32")]
pub const PGL_ALPHA_SHIFT: u32 = 24;

// --- BGRA32 ---
// Packed u32: 0xBBGGRRAA
#[cfg(feature = "bgra32")]
pub const PGL_RED_SHIFT: u32 = 8;
#[cfg(feature = "bgra32")]
pub const PGL_GREEN_SHIFT: u32 = 16;
#[cfg(feature = "bgra32")]
pub const PGL_BLUE_SHIFT: u32 = 24;
#[cfg(feature = "bgra32")]
pub const PGL_ALPHA_SHIFT: u32 = 0;

// Derived masks for 32-bit formats
#[cfg(any(feature = "abgr32", feature = "rgba32", feature = "argb32", feature = "bgra32"))]
pub const PGL_RMASK: u32 = 0xFF << PGL_RED_SHIFT;
#[cfg(any(feature = "abgr32", feature = "rgba32", feature = "argb32", feature = "bgra32"))]
pub const PGL_GMASK: u32 = 0xFF << PGL_GREEN_SHIFT;
#[cfg(any(feature = "abgr32", feature = "rgba32", feature = "argb32", feature = "bgra32"))]
pub const PGL_BMASK: u32 = 0xFF << PGL_BLUE_SHIFT;
#[cfg(any(feature = "abgr32", feature = "rgba32", feature = "argb32", feature = "bgra32"))]
pub const PGL_AMASK: u32 = 0xFF << PGL_ALPHA_SHIFT;

/// Bytes per pixel for the selected 32-bit format.
#[cfg(any(feature = "abgr32", feature = "rgba32", feature = "argb32", feature = "bgra32"))]
pub const PGL_BPP: usize = 4;

/// Bytes per pixel for 16-bit 565 formats.
#[cfg(any(feature = "rgb565", feature = "bgr565"))]
pub const PGL_BPP: usize = 2;

// --- RGB565 ---
#[cfg(feature = "rgb565")]
pub const PGL_RED_SHIFT_565: u32 = 11;
#[cfg(feature = "rgb565")]
pub const PGL_GREEN_SHIFT_565: u32 = 5;
#[cfg(feature = "rgb565")]
pub const PGL_BLUE_SHIFT_565: u32 = 0;

// --- BGR565 ---
#[cfg(feature = "bgr565")]
pub const PGL_RED_SHIFT_565: u32 = 0;
#[cfg(feature = "bgr565")]
pub const PGL_GREEN_SHIFT_565: u32 = 5;
#[cfg(feature = "bgr565")]
pub const PGL_BLUE_SHIFT_565: u32 = 11;

// ---------------------------------------------------------------------------
// Pixel format helpers
// ---------------------------------------------------------------------------

/// Pack RGBA bytes into a u32 pixel in the selected 32-bit format.
#[cfg(any(feature = "abgr32", feature = "rgba32", feature = "argb32", feature = "bgra32"))]
#[inline]
pub fn rgba_to_pixel(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((a as u32) << PGL_ALPHA_SHIFT)
        | ((b as u32) << PGL_BLUE_SHIFT)
        | ((g as u32) << PGL_GREEN_SHIFT)
        | ((r as u32) << PGL_RED_SHIFT)
}

/// Pack RGBA bytes into a u16 pixel in the selected 565 format (alpha is discarded).
#[cfg(any(feature = "rgb565", feature = "bgr565"))]
#[inline]
pub fn rgba_to_pixel(r: u8, g: u8, b: u8, _a: u8) -> u32 {
    let r5 = ((r as u32) >> 3) & 0x1F;
    let g6 = ((g as u32) >> 2) & 0x3F;
    let b5 = ((b as u32) >> 3) & 0x1F;
    (r5 << PGL_RED_SHIFT_565) | (g6 << PGL_GREEN_SHIFT_565) | (b5 << PGL_BLUE_SHIFT_565)
}

/// Unpack a u32 pixel (selected 32-bit format) into a Color.
#[cfg(any(feature = "abgr32", feature = "rgba32", feature = "argb32", feature = "bgra32"))]
#[inline]
pub fn pixel_to_color(p: u32) -> Color {
    Color::new(
        ((p >> PGL_RED_SHIFT) & 0xFF) as u8,
        ((p >> PGL_GREEN_SHIFT) & 0xFF) as u8,
        ((p >> PGL_BLUE_SHIFT) & 0xFF) as u8,
        ((p >> PGL_ALPHA_SHIFT) & 0xFF) as u8,
    )
}

/// Unpack a u16 pixel (565 format) into a Color. Alpha is always 255.
#[cfg(any(feature = "rgb565", feature = "bgr565"))]
#[inline]
pub fn pixel_to_color(p: u32) -> Color {
    let r5 = ((p >> PGL_RED_SHIFT_565) & 0x1F) as u8;
    let g6 = ((p >> PGL_GREEN_SHIFT_565) & 0x3F) as u8;
    let b5 = ((p >> PGL_BLUE_SHIFT_565) & 0x1F) as u8;
    // Expand 5/6 bit values to 8 bit by replicating top bits into bottom bits
    Color::new(
        (r5 << 3) | (r5 >> 2),
        (g6 << 2) | (g6 >> 4),
        (b5 << 3) | (b5 >> 2),
        255,
    )
}

/// Convert a Color to a Vec4 with each component in [0.0, 1.0].
#[inline]
pub fn color_to_vec4(c: Color) -> Vec4 {
    c.to_vec4()
}

/// Convert a Vec4 (clamped to [0,1]) to a Color.
#[inline]
pub fn vec4_to_color(v: Vec4) -> Color {
    Color::from_vec4(v)
}

// ---------------------------------------------------------------------------
// Depth/stencil buffer format helpers
// ---------------------------------------------------------------------------
//
// The depth/stencil buffer is stored as raw bytes in GlFramebuffer::buf.
// For D24S8: each entry is 4 bytes (u32) where upper 24 bits = depth, lower 8 = stencil.
// For D16: each entry is 2 bytes (u16) holding 16-bit depth, no stencil.
// For no_depth: depth operations are no-ops.

/// Maximum depth value for the selected depth format.
#[cfg(feature = "d24s8")]
pub const PGL_MAX_DEPTH: u32 = 0x00FFFFFF;
#[cfg(feature = "d16")]
pub const PGL_MAX_DEPTH: u32 = 0xFFFF;
#[cfg(feature = "no_depth")]
pub const PGL_MAX_DEPTH: u32 = 0;

/// Bytes per depth/stencil entry.
#[cfg(feature = "d24s8")]
pub const PGL_DEPTH_BPP: usize = 4;
#[cfg(feature = "d16")]
pub const PGL_DEPTH_BPP: usize = 2;
#[cfg(feature = "no_depth")]
pub const PGL_DEPTH_BPP: usize = 0;

/// Convert a float z in [0,1] to a depth buffer integer value.
#[inline]
pub fn z_to_depth(z: f32) -> u32 {
    (z.clamp_(0.0, 1.0) * PGL_MAX_DEPTH as f32) as u32
}

/// Read the depth portion from a depth/stencil buffer entry.
#[cfg(feature = "d24s8")]
#[inline]
pub fn get_z(zbuf_val: u32) -> u32 {
    zbuf_val >> 8
}

#[cfg(feature = "d16")]
#[inline]
pub fn get_z(zbuf_val: u32) -> u32 {
    zbuf_val & 0xFFFF
}

#[cfg(feature = "no_depth")]
#[inline]
pub fn get_z(_zbuf_val: u32) -> u32 {
    0
}

/// Write a depth value into a depth/stencil buffer entry, preserving stencil.
#[cfg(feature = "d24s8")]
#[inline]
pub fn set_z(zbuf_val: u32, z: u32) -> u32 {
    (z << 8) | (zbuf_val & 0xFF)
}

#[cfg(feature = "d16")]
#[inline]
pub fn set_z(_zbuf_val: u32, z: u32) -> u32 {
    z & 0xFFFF
}

#[cfg(feature = "no_depth")]
#[inline]
pub fn set_z(zbuf_val: u32, _z: u32) -> u32 {
    zbuf_val
}

/// Read the stencil portion from a depth/stencil buffer entry.
#[cfg(feature = "d24s8")]
#[inline]
pub fn get_stencil(zbuf_val: u32) -> u8 {
    (zbuf_val & 0xFF) as u8
}

#[cfg(feature = "d16")]
#[inline]
pub fn get_stencil(_zbuf_val: u32) -> u8 {
    0
}

#[cfg(feature = "no_depth")]
#[inline]
pub fn get_stencil(_zbuf_val: u32) -> u8 {
    0
}

/// Write a stencil value into a depth/stencil buffer entry, preserving depth.
#[cfg(feature = "d24s8")]
#[inline]
pub fn set_stencil(zbuf_val: u32, stencil: u8) -> u32 {
    (zbuf_val & !0xFF) | (stencil as u32)
}

#[cfg(feature = "d16")]
#[inline]
pub fn set_stencil(zbuf_val: u32, _stencil: u8) -> u32 {
    zbuf_val
}

#[cfg(feature = "no_depth")]
#[inline]
pub fn set_stencil(zbuf_val: u32, _stencil: u8) -> u32 {
    zbuf_val
}

// ---------------------------------------------------------------------------
// Depth/stencil buffer read/write
// ---------------------------------------------------------------------------

/// Read a raw value from the depth/stencil buffer at the given pixel index.
#[cfg(feature = "d24s8")]
#[inline]
fn read_zbuf(c: &GlContext, idx: usize) -> u32 {
    let data = &c.zbuf.buf;
    let off = idx * 4;
    if off + 4 > data.len() {
        return 0;
    }
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

#[cfg(feature = "d16")]
#[inline]
fn read_zbuf(c: &GlContext, idx: usize) -> u32 {
    let data = &c.zbuf.buf;
    let off = idx * 2;
    if off + 2 > data.len() {
        return 0;
    }
    u16::from_le_bytes([data[off], data[off + 1]]) as u32
}

#[cfg(feature = "no_depth")]
#[inline]
fn read_zbuf(_c: &GlContext, _idx: usize) -> u32 {
    0
}

/// Write a raw value to the depth/stencil buffer at the given pixel index.
#[cfg(feature = "d24s8")]
#[inline]
fn write_zbuf(c: &mut GlContext, idx: usize, val: u32) {
    let data = &mut c.zbuf.buf;
    let off = idx * 4;
    if off + 4 > data.len() {
        return;
    }
    let bytes = val.to_le_bytes();
    data[off] = bytes[0];
    data[off + 1] = bytes[1];
    data[off + 2] = bytes[2];
    data[off + 3] = bytes[3];
}

#[cfg(feature = "d16")]
#[inline]
fn write_zbuf(c: &mut GlContext, idx: usize, val: u32) {
    let data = &mut c.zbuf.buf;
    let off = idx * 2;
    if off + 2 > data.len() {
        return;
    }
    let bytes = (val as u16).to_le_bytes();
    data[off] = bytes[0];
    data[off + 1] = bytes[1];
}

#[cfg(feature = "no_depth")]
#[inline]
fn write_zbuf(_c: &mut GlContext, _idx: usize, _val: u32) {}

// ---------------------------------------------------------------------------
// Back buffer read/write
// ---------------------------------------------------------------------------

/// Read a pixel from the back buffer at the given buffer index.
#[cfg(any(feature = "abgr32", feature = "rgba32", feature = "argb32", feature = "bgra32"))]
#[inline]
fn read_backbuf_pixel(c: &GlContext, idx: usize) -> u32 {
    let data = &c.back_buffer.buf;
    let off = idx * 4;
    if off + 4 > data.len() {
        return 0;
    }
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

#[cfg(any(feature = "rgb565", feature = "bgr565"))]
#[inline]
fn read_backbuf_pixel(c: &GlContext, idx: usize) -> u32 {
    let data = &c.back_buffer.buf;
    let off = idx * 2;
    if off + 2 > data.len() {
        return 0;
    }
    u16::from_le_bytes([data[off], data[off + 1]]) as u32
}

/// Write a pixel to the back buffer at the given buffer index.
#[cfg(any(feature = "abgr32", feature = "rgba32", feature = "argb32", feature = "bgra32"))]
#[inline]
fn write_backbuf_pixel(c: &mut GlContext, idx: usize, val: u32) {
    let data = &mut c.back_buffer.buf;
    let off = idx * 4;
    if off + 4 > data.len() {
        return;
    }
    let bytes = val.to_le_bytes();
    data[off] = bytes[0];
    data[off + 1] = bytes[1];
    data[off + 2] = bytes[2];
    data[off + 3] = bytes[3];
}

#[cfg(any(feature = "rgb565", feature = "bgr565"))]
#[inline]
fn write_backbuf_pixel(c: &mut GlContext, idx: usize, val: u32) {
    let data = &mut c.back_buffer.buf;
    let off = idx * 2;
    if off + 2 > data.len() {
        return;
    }
    let bytes = (val as u16).to_le_bytes();
    data[off] = bytes[0];
    data[off + 1] = bytes[1];
}

// ---------------------------------------------------------------------------
// Clip code computation
// ---------------------------------------------------------------------------

/// Compute clip codes for a point in clip space. Each bit indicates a
/// violated half-plane. If `depth_clamp` is true, near/far clip planes
/// are not tested.
///
/// Bit layout:
///   bit 0 (1)  : z < -w  (near)
///   bit 1 (2)  : z >  w  (far)
///   bit 2 (4)  : x < -w  (left)
///   bit 3 (8)  : x >  w  (right)
///   bit 4 (16) : y < -w  (bottom)
///   bit 5 (32) : y >  w  (top)
pub fn gl_clipcode(pt: Vec4, depth_clamp: bool) -> i32 {
    let w = pt.w * (1.0 + CLIP_EPSILON);
    let mut code = 0;
    if !depth_clamp {
        if pt.z < -w {
            code |= 1;
        }
        if pt.z > w {
            code |= 2;
        }
    }
    if pt.x < -w {
        code |= 4;
    }
    if pt.x > w {
        code |= 8;
    }
    if pt.y < -w {
        code |= 16;
    }
    if pt.y > w {
        code |= 32;
    }
    code
}

// ---------------------------------------------------------------------------
// Front face determination
// ---------------------------------------------------------------------------

/// Determine if a triangle is front-facing based on its screen-space winding
/// and the current front_face convention (GL_CCW or GL_CW).
pub fn is_front_facing(v0: &GlVertex, v1: &GlVertex, v2: &GlVertex, front_face: GLenum) -> bool {
    // Compute signed area from perspective-divided screen coordinates
    let p0 = v4_to_v3h(v0.screen_space);
    let p1 = v4_to_v3h(v1.screen_space);
    let p2 = v4_to_v3h(v2.screen_space);

    let mut a = p0.x * p1.y - p1.x * p0.y
              + p1.x * p2.y - p2.x * p1.y
              + p2.x * p0.y - p0.x * p2.y;

    if front_face == GL_CW {
        a = -a;
    }
    a > 0.0
}

// ---------------------------------------------------------------------------
// Vertex attribute fetching
// ---------------------------------------------------------------------------

/// Read a vertex attribute from buffers, converting from the source type
/// to a Vec4. Handles GL_FLOAT, GL_UNSIGNED_BYTE, GL_BYTE, GL_UNSIGNED_SHORT,
/// GL_SHORT, GL_UNSIGNED_INT, GL_INT, GL_DOUBLE with optional normalization.
///
/// Components not present in the source are filled with (0, 0, 0, 1).
pub fn get_v_attrib(v: &GlVertexAttrib, i: GLsizei, buffers: &[GlBuffer]) -> Vec4 {
    let num_comps = v.size as usize;

    // Compute stride: if 0, tightly packed
    let elem_size = match v.type_ {
        GL_FLOAT => 4usize,
        GL_UNSIGNED_BYTE | GL_BYTE => 1,
        GL_UNSIGNED_SHORT | GL_SHORT => 2,
        GL_UNSIGNED_INT | GL_INT => 4,
        GL_DOUBLE => 8,
        _ => 4,
    };
    let stride = if v.stride == 0 {
        (num_comps * elem_size) as usize
    } else {
        v.stride as usize
    };

    let offset = stride * (i as usize);

    // Client array: v.buf == 0 means v.offset is a raw pointer to client memory
    let data: &[u8] = if v.buf == 0 {
        let ptr = v.offset as *const u8;
        unsafe { core::slice::from_raw_parts(ptr.add(offset), stride) }
    } else {
        let buf = &buffers[v.buf as usize];
        if !buf.user_data.is_null() {
            // User-owned buffer: read from raw pointer
            unsafe { core::slice::from_raw_parts(buf.user_data.add(v.offset as usize + offset), stride) }
        } else {
            &buf.data[(v.offset as usize + offset)..]
        }
    };

    let mut result = Vec4::new(0.0, 0.0, 0.0, 1.0);
    let comps: [&mut f32; 4] = unsafe {
        [
            &mut *(&mut result.x as *mut f32),
            &mut *(&mut result.y as *mut f32),
            &mut *(&mut result.z as *mut f32),
            &mut *(&mut result.w as *mut f32),
        ]
    };

    match v.type_ {
        GL_FLOAT => {
            for j in 0..num_comps {
                let off = j * 4;
                let bytes = [data[off], data[off + 1], data[off + 2], data[off + 3]];
                *comps[j] = f32::from_le_bytes(bytes);
            }
        }
        GL_UNSIGNED_BYTE => {
            for j in 0..num_comps {
                let val = data[j] as f32;
                *comps[j] = if v.normalized {
                    rsw_mapf(val, 0.0, 255.0, 0.0, 1.0)
                } else {
                    val
                };
            }
        }
        GL_BYTE => {
            for j in 0..num_comps {
                let val = data[j] as i8 as f32;
                *comps[j] = if v.normalized {
                    rsw_mapf(val, -128.0, 127.0, -1.0, 1.0)
                } else {
                    val
                };
            }
        }
        GL_UNSIGNED_SHORT => {
            for j in 0..num_comps {
                let off = j * 2;
                let val = u16::from_le_bytes([data[off], data[off + 1]]) as f32;
                *comps[j] = if v.normalized {
                    rsw_mapf(val, 0.0, 65535.0, 0.0, 1.0)
                } else {
                    val
                };
            }
        }
        GL_SHORT => {
            for j in 0..num_comps {
                let off = j * 2;
                let val = i16::from_le_bytes([data[off], data[off + 1]]) as f32;
                *comps[j] = if v.normalized {
                    rsw_mapf(val, -32768.0, 32767.0, -1.0, 1.0)
                } else {
                    val
                };
            }
        }
        GL_UNSIGNED_INT => {
            for j in 0..num_comps {
                let off = j * 4;
                let val =
                    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
                        as f32;
                *comps[j] = if v.normalized {
                    rsw_mapf(val, 0.0, u32::MAX as f32, 0.0, 1.0)
                } else {
                    val
                };
            }
        }
        GL_INT => {
            for j in 0..num_comps {
                let off = j * 4;
                let val =
                    i32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
                        as f32;
                *comps[j] = if v.normalized {
                    rsw_mapf(val, i32::MIN as f32, i32::MAX as f32, -1.0, 1.0)
                } else {
                    val
                };
            }
        }
        GL_DOUBLE => {
            for j in 0..num_comps {
                let off = j * 8;
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&data[off..off + 8]);
                *comps[j] = f64::from_le_bytes(bytes) as f32;
            }
        }
        _ => {}
    }

    result
}

// ---------------------------------------------------------------------------
// Vertex stage
// ---------------------------------------------------------------------------

/// Process a single vertex: copy attributes, run vertex shader, store output.
pub fn do_vertex(
    c: &mut GlContext,
    v: &[GlVertexAttrib],
    enabled: &[usize],
    num_enabled: usize,
    i: GLsizei,
    vert: usize,
) {
    // Copy enabled vertex attributes
    for e in 0..num_enabled {
        let idx = enabled[e];
        c.vertex_attribs_vs[idx] = get_v_attrib(&v[idx], i, &c.buffers);
    }

    let vs_output_size = c.vs_output.size as usize;
    let start = vert * vs_output_size;
    let end = start + vs_output_size;

    // Make sure vs_output buffer is large enough
    if c.vs_output.output_buf.len() < end {
        c.vs_output.output_buf.resize(end, 0.0);
    }

    // Run vertex shader
    let program = &c.programs[c.cur_program as usize];
    let vs = program.vertex_shader;
    let uniform = program.uniform;
    let vs_out = &mut c.vs_output.output_buf[start..end];

    unsafe {
        (vs)(
            vs_out.as_mut_ptr(),
            c.vertex_attribs_vs.as_mut_ptr(),
            &mut c.builtins as *mut ShaderBuiltins,
            uniform,
        );
    }

    // Store clip space position
    c.glverts[vert].clip_space = c.builtins.gl_Position;
    c.glverts[vert].clip_code = gl_clipcode(c.builtins.gl_Position, c.depth_clamp);

    // Copy vertex shader outputs
    c.glverts[vert].vs_out.resize(vs_output_size, 0.0);
    c.glverts[vert].vs_out[..vs_output_size].copy_from_slice(vs_out);
}

/// Run the vertex stage for all vertices in the draw call.
pub fn vertex_stage(
    c: &mut GlContext,
    first_or_indices: usize,
    count: GLsizei,
    instance_id: GLsizei,
    base_instance: GLuint,
    use_elems_type: GLenum,
) {
    let vao_idx = c.cur_vertex_array as usize;
    let v = c.vertex_arrays[vao_idx].vertex_attribs.clone();

    // Gather enabled attribute indices
    let mut enabled = [0usize; GL_MAX_VERTEX_ATTRIBS];
    let mut num_enabled = 0;
    for j in 0..GL_MAX_VERTEX_ATTRIBS {
        if v[j].enabled {
            enabled[num_enabled] = j;
            num_enabled += 1;
        }
    }

    // Handle instanced attributes
    for j in 0..num_enabled {
        let idx = enabled[j];
        if v[idx].divisor != 0 {
            let inst_idx = ((instance_id as u32 / v[idx].divisor) + base_instance) as GLsizei;
            c.vertex_attribs_vs[idx] = get_v_attrib(&v[idx], inst_idx, &c.buffers);
        }
    }

    c.builtins.gl_InstanceID = instance_id;
    c.builtins.gl_BaseInstance = base_instance as GLint;

    // Ensure glverts is large enough
    if c.glverts.len() < count as usize {
        c.glverts.resize(count as usize, GlVertex::default());
    }

    for i in 0..count {
        // Determine vertex index
        let vertex_idx = if use_elems_type != 0 {
            let elem_buf_idx = c.vertex_arrays[vao_idx].element_buffer as usize;
            if elem_buf_idx == 0 {
                // Client array: first_or_indices is a raw pointer to element data
                let ptr = first_or_indices as *const u8;
                match use_elems_type {
                    GL_UNSIGNED_BYTE => unsafe {
                        *ptr.add(i as usize) as GLsizei
                    },
                    GL_UNSIGNED_SHORT => unsafe {
                        let p = ptr.add((i as usize) * 2) as *const u16;
                        *p as GLsizei
                    },
                    GL_UNSIGNED_INT => unsafe {
                        let p = ptr.add((i as usize) * 4) as *const u32;
                        *p as GLsizei
                    },
                    _ => i,
                }
            } else {
                let elem_buf = &c.buffers[elem_buf_idx];
                let elem_data: &[u8] = if !elem_buf.user_data.is_null() {
                    unsafe { core::slice::from_raw_parts(elem_buf.user_data, elem_buf.size as usize) }
                } else {
                    &elem_buf.data
                };
                match use_elems_type {
                    GL_UNSIGNED_BYTE => {
                        let off = first_or_indices + i as usize;
                        elem_data[off] as GLsizei
                    }
                    GL_UNSIGNED_SHORT => {
                        let off = first_or_indices + (i as usize) * 2;
                        u16::from_le_bytes([elem_data[off], elem_data[off + 1]]) as GLsizei
                    }
                    GL_UNSIGNED_INT => {
                        let off = first_or_indices + (i as usize) * 4;
                        u32::from_le_bytes([
                            elem_data[off],
                            elem_data[off + 1],
                            elem_data[off + 2],
                            elem_data[off + 3],
                        ]) as GLsizei
                    }
                    _ => i,
                }
            }
        } else {
            (first_or_indices as GLsizei) + i
        };

        // Filter out instanced attributes for do_vertex
        let mut non_inst_enabled = [0usize; GL_MAX_VERTEX_ATTRIBS];
        let mut num_non_inst = 0;
        for e in 0..num_enabled {
            let idx = enabled[e];
            if v[idx].divisor == 0 {
                non_inst_enabled[num_non_inst] = idx;
                num_non_inst += 1;
            }
        }

        do_vertex(c, &v, &non_inst_enabled, num_non_inst, vertex_idx, i as usize);
    }
}

// ---------------------------------------------------------------------------
// Pipeline runner
// ---------------------------------------------------------------------------

/// Run the full rendering pipeline: vertex stage then primitive assembly and rasterization.
///
/// `mode` is the primitive type (GL_POINTS, GL_LINES, GL_TRIANGLES, etc.).
/// `first` is the starting index (or byte offset into the element buffer if `use_elements`).
/// `count` is the number of vertices/indices.
/// `instance_id` and `base_instance` are for instanced rendering.
/// `use_elements` indicates indexed drawing.
pub fn run_pipeline(
    c: &mut GlContext,
    mode: GLenum,
    first: usize,
    count: GLsizei,
    instance_id: GLuint,
    base_instance: GLuint,
    use_elements: bool,
) {
    let use_elems_type = if use_elements {
        // For indexed drawing the element type is determined by the caller;
        // we default to GL_UNSIGNED_INT.  A more complete implementation would
        // accept the type as a parameter.
        GL_UNSIGNED_INT
    } else {
        0
    };

    vertex_stage(
        c,
        first as usize,
        count,
        instance_id as GLsizei,
        base_instance,
        use_elems_type,
    );

    match mode {
        GL_POINTS => {
            for i in 0..count as usize {
                // Only clip Z, allow partial points outside XY to show
                if c.glverts[i].clip_code & 0x3 != 0 {
                    continue;
                }
                c.glverts[i].screen_space = mult_m4_v4(c.vp_mat, c.glverts[i].clip_space);
                let vert = c.glverts[i].clone();
                draw_point(c, &vert, 0.0);
            }
        }
        GL_LINES => {
            let n = (count / 2) as usize;
            for i in 0..n {
                let v1 = c.glverts[i * 2].clone();
                let v2 = c.glverts[i * 2 + 1].clone();
                draw_line_clip(c, &v1, &v2);
            }
        }
        GL_LINE_STRIP => {
            if count >= 2 {
                for i in 0..(count - 1) as usize {
                    let v1 = c.glverts[i].clone();
                    let v2 = c.glverts[i + 1].clone();
                    draw_line_clip(c, &v1, &v2);
                }
            }
        }
        GL_LINE_LOOP => {
            if count >= 2 {
                for i in 0..(count - 1) as usize {
                    let v1 = c.glverts[i].clone();
                    let v2 = c.glverts[i + 1].clone();
                    draw_line_clip(c, &v1, &v2);
                }
                // Close the loop
                let v1 = c.glverts[(count - 1) as usize].clone();
                let v2 = c.glverts[0].clone();
                draw_line_clip(c, &v1, &v2);
            }
        }
        GL_TRIANGLES => {
            let n = (count / 3) as usize;
            let provoke_last = c.provoking_vert == GL_LAST_VERTEX_CONVENTION;
            for i in 0..n {
                let i0 = i * 3;
                let i1 = i * 3 + 1;
                let i2 = i * 3 + 2;
                let provoke = if provoke_last { i2 } else { i0 };
                draw_triangle(c, i0, i1, i2, provoke);
            }
        }
        GL_TRIANGLE_STRIP => {
            if count >= 3 {
                let provoke_last = c.provoking_vert == GL_LAST_VERTEX_CONVENTION;
                for i in 0..(count - 2) as usize {
                    let (i0, i1, i2) = if i % 2 == 0 {
                        (i, i + 1, i + 2)
                    } else {
                        (i + 1, i, i + 2)
                    };
                    let provoke = if provoke_last { i + 2 } else { i };
                    draw_triangle(c, i0, i1, i2, provoke);
                }
            }
        }
        GL_TRIANGLE_FAN => {
            if count >= 3 {
                let provoke_last = c.provoking_vert == GL_LAST_VERTEX_CONVENTION;
                for i in 1..(count - 1) as usize {
                    let provoke = if provoke_last { i + 1 } else { i };
                    draw_triangle(c, 0, i, i + 1, provoke);
                }
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Depth testing
// ---------------------------------------------------------------------------

/// Compare depth values based on the given depth function.
/// The comparison is: does `zval` pass the test against `zbufval`?
#[inline]
pub fn depthtest(zval: u32, zbufval: u32, depth_func: GLenum) -> bool {
    match depth_func {
        GL_LESS => zval < zbufval,
        GL_LEQUAL => zval <= zbufval,
        GL_GREATER => zval > zbufval,
        GL_GEQUAL => zval >= zbufval,
        GL_EQUAL => zval == zbufval,
        GL_NOTEQUAL => zval != zbufval,
        GL_ALWAYS => true,
        GL_NEVER => false,
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// Stencil operations
// ---------------------------------------------------------------------------

/// Apply a stencil operation to produce a new stencil value.
pub fn stencil_op_func(op: GLenum, old_val: u8, ref_val: u8) -> u8 {
    match op {
        GL_KEEP => old_val,
        GL_ZERO => 0,
        GL_REPLACE => ref_val,
        GL_INCR => {
            if old_val < 255 {
                old_val + 1
            } else {
                255
            }
        }
        GL_INCR_WRAP => old_val.wrapping_add(1),
        GL_DECR => {
            if old_val > 0 {
                old_val - 1
            } else {
                0
            }
        }
        GL_DECR_WRAP => old_val.wrapping_sub(1),
        GL_INVERT => !old_val,
        _ => old_val,
    }
}

/// Perform the stencil comparison test.
pub fn stencil_test_func(func: GLenum, ref_val: i32, mask: u32, stencil_val: u8) -> bool {
    let masked_ref = (ref_val as u32) & mask;
    let masked_stencil = (stencil_val as u32) & mask;
    match func {
        GL_LESS => masked_ref < masked_stencil,
        GL_LEQUAL => masked_ref <= masked_stencil,
        GL_GREATER => masked_ref > masked_stencil,
        GL_GEQUAL => masked_ref >= masked_stencil,
        GL_EQUAL => masked_ref == masked_stencil,
        GL_NOTEQUAL => masked_ref != masked_stencil,
        GL_ALWAYS => true,
        GL_NEVER => false,
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// Logic ops
// ---------------------------------------------------------------------------

/// Apply a bitwise logic operation between source and destination pixels.
pub fn logic_ops_pixel(logic_func: GLenum, s: u32, d: u32) -> u32 {
    match logic_func {
        GL_CLEAR => 0,
        GL_SET => 0xFFFFFFFF,
        GL_COPY => s,
        GL_COPY_INVERTED => !s,
        GL_NOOP => d,
        GL_INVERT => !d,
        GL_AND => s & d,
        GL_NAND => !(s & d),
        GL_OR => s | d,
        GL_NOR => !(s | d),
        GL_XOR => s ^ d,
        GL_EQUIV => !(s ^ d),
        GL_AND_REVERSE => s & !d,
        GL_AND_INVERTED => !s & d,
        GL_OR_REVERSE => s | !d,
        GL_OR_INVERTED => !s | d,
        _ => s,
    }
}

// ---------------------------------------------------------------------------
// Fragment processing
// ---------------------------------------------------------------------------

/// Process a fragment through depth and stencil tests. Returns true if the
/// fragment passes all tests and should be drawn.
///
/// Uses lastrow addressing where pixel index = `(height - 1 - y) * width + x`.
pub fn fragment_processing(c: &mut GlContext, x: i32, y: i32, z: f32) -> bool {
    let buf_w = c.back_buffer.w;
    let buf_h = c.back_buffer.h;

    // Bounds check
    if x < 0 || x >= buf_w || y < 0 || y >= buf_h {
        return false;
    }

    let buf_idx = ((buf_h - 1 - y) * buf_w + x) as usize;

    // Convert float z to integer depth
    let z_val = z_to_depth(z);

    let stencil_test_enabled = c.stencil_test;
    let depth_test_enabled = c.depth_test;

    if stencil_test_enabled {
        let zbuf_data = read_zbuf(c, buf_idx);
        let old_stencil = get_stencil(zbuf_data);
        let old_depth = get_z(zbuf_data);

        // Use front-face stencil state; back-face selection is handled at
        // the triangle level by swapping the stencil state references.
        let stencil_func = c.stencil_func;
        let stencil_ref = c.stencil_ref;
        let stencil_valuemask = c.stencil_valuemask;
        let stencil_writemask = c.stencil_writemask as u8;

        let stencil_pass =
            stencil_test_func(stencil_func, stencil_ref, stencil_valuemask, old_stencil);

        if !stencil_pass {
            // Stencil fail
            let new_stencil =
                stencil_op_func(c.stencil_sfail, old_stencil, stencil_ref as u8);
            let masked_stencil =
                (old_stencil & !stencil_writemask) | (new_stencil & stencil_writemask);
            let new_zbuf = set_stencil(set_z(zbuf_data, old_depth), masked_stencil);
            write_zbuf(c, buf_idx, new_zbuf);
            return false;
        }

        if depth_test_enabled {
            let depth_pass = depthtest(z_val, old_depth, c.depth_func);
            if !depth_pass {
                // Stencil pass, depth fail
                let new_stencil =
                    stencil_op_func(c.stencil_dpfail, old_stencil, stencil_ref as u8);
                let masked_stencil =
                    (old_stencil & !stencil_writemask) | (new_stencil & stencil_writemask);
                let new_zbuf = set_stencil(set_z(zbuf_data, old_depth), masked_stencil);
                write_zbuf(c, buf_idx, new_zbuf);
                return false;
            }
            // Both pass
            let new_stencil =
                stencil_op_func(c.stencil_dppass, old_stencil, stencil_ref as u8);
            let masked_stencil =
                (old_stencil & !stencil_writemask) | (new_stencil & stencil_writemask);
            let new_depth = if c.depth_mask { z_val } else { old_depth };
            let new_zbuf = set_stencil(set_z(zbuf_data, new_depth), masked_stencil);
            write_zbuf(c, buf_idx, new_zbuf);
            return true;
        } else {
            // No depth test, stencil passed
            let new_stencil =
                stencil_op_func(c.stencil_dppass, old_stencil, stencil_ref as u8);
            let masked_stencil =
                (old_stencil & !stencil_writemask) | (new_stencil & stencil_writemask);
            let new_zbuf = set_stencil(zbuf_data, masked_stencil);
            write_zbuf(c, buf_idx, new_zbuf);
            return true;
        }
    }

    if depth_test_enabled {
        let zbuf_data = read_zbuf(c, buf_idx);
        let old_depth = get_z(zbuf_data);
        let depth_pass = depthtest(z_val, old_depth, c.depth_func);
        if !depth_pass {
            return false;
        }
        if c.depth_mask {
            let new_zbuf = set_z(zbuf_data, z_val);
            write_zbuf(c, buf_idx, new_zbuf);
        }
        return true;
    }

    true
}

// ---------------------------------------------------------------------------
// Blend pixel
// ---------------------------------------------------------------------------

/// Compute a blend factor vector for the given blend function enum.
#[inline]
fn blend_factor(factor: GLenum, src: Vec4, dst: Vec4, const_color: Vec4) -> Vec4 {
    match factor {
        GL_ZERO => Vec4::new(0.0, 0.0, 0.0, 0.0),
        GL_ONE => Vec4::new(1.0, 1.0, 1.0, 1.0),
        GL_SRC_COLOR => src,
        GL_ONE_MINUS_SRC_COLOR => {
            Vec4::new(1.0 - src.x, 1.0 - src.y, 1.0 - src.z, 1.0 - src.w)
        }
        GL_DST_COLOR => dst,
        GL_ONE_MINUS_DST_COLOR => {
            Vec4::new(1.0 - dst.x, 1.0 - dst.y, 1.0 - dst.z, 1.0 - dst.w)
        }
        GL_SRC_ALPHA => {
            Vec4::new(src.w, src.w, src.w, src.w)
        }
        GL_ONE_MINUS_SRC_ALPHA => {
            let a = 1.0 - src.w;
            Vec4::new(a, a, a, a)
        }
        GL_DST_ALPHA => {
            Vec4::new(dst.w, dst.w, dst.w, dst.w)
        }
        GL_ONE_MINUS_DST_ALPHA => {
            let a = 1.0 - dst.w;
            Vec4::new(a, a, a, a)
        }
        GL_CONSTANT_COLOR => const_color,
        GL_ONE_MINUS_CONSTANT_COLOR => Vec4::new(
            1.0 - const_color.x,
            1.0 - const_color.y,
            1.0 - const_color.z,
            1.0 - const_color.w,
        ),
        GL_CONSTANT_ALPHA => {
            Vec4::new(const_color.w, const_color.w, const_color.w, const_color.w)
        }
        GL_ONE_MINUS_CONSTANT_ALPHA => {
            let a = 1.0 - const_color.w;
            Vec4::new(a, a, a, a)
        }
        GL_SRC_ALPHA_SATURATE => {
            let f = src.w.min_(1.0 - dst.w);
            Vec4::new(f, f, f, 1.0)
        }
        _ => Vec4::new(1.0, 1.0, 1.0, 1.0),
    }
}

/// Blend source and destination colors according to the current blend state.
/// Returns the blended result as a Color.
pub fn blend_pixel(c: &GlContext, src: Vec4, dst: Vec4) -> Color {
    let const_color = c.blend_color;

    let sf_rgb = blend_factor(c.blend_srgb, src, dst, const_color);
    let sf_a = blend_factor(c.blend_sa, src, dst, const_color);
    let df_rgb = blend_factor(c.blend_drgb, src, dst, const_color);
    let df_a = blend_factor(c.blend_da, src, dst, const_color);

    // Source * srcFactor, Dest * dstFactor
    let s_rgb = Vec4::new(
        src.x * sf_rgb.x,
        src.y * sf_rgb.y,
        src.z * sf_rgb.z,
        src.w * sf_a.w,
    );
    let d_rgb = Vec4::new(
        dst.x * df_rgb.x,
        dst.y * df_rgb.y,
        dst.z * df_rgb.z,
        dst.w * df_a.w,
    );

    // RGB equation
    let (r, g, b) = match c.blend_eq_rgb {
        GL_FUNC_ADD => (s_rgb.x + d_rgb.x, s_rgb.y + d_rgb.y, s_rgb.z + d_rgb.z),
        GL_FUNC_SUBTRACT => (s_rgb.x - d_rgb.x, s_rgb.y - d_rgb.y, s_rgb.z - d_rgb.z),
        GL_FUNC_REVERSE_SUBTRACT => (d_rgb.x - s_rgb.x, d_rgb.y - s_rgb.y, d_rgb.z - s_rgb.z),
        GL_MIN => (src.x.min_(dst.x), src.y.min_(dst.y), src.z.min_(dst.z)),
        GL_MAX => (src.x.max_(dst.x), src.y.max_(dst.y), src.z.max_(dst.z)),
        _ => (s_rgb.x + d_rgb.x, s_rgb.y + d_rgb.y, s_rgb.z + d_rgb.z),
    };

    // Alpha equation
    let a = match c.blend_eq_a {
        GL_FUNC_ADD => s_rgb.w + d_rgb.w,
        GL_FUNC_SUBTRACT => s_rgb.w - d_rgb.w,
        GL_FUNC_REVERSE_SUBTRACT => d_rgb.w - s_rgb.w,
        GL_MIN => src.w.min_(dst.w),
        GL_MAX => src.w.max_(dst.w),
        _ => s_rgb.w + d_rgb.w,
    };

    Color::from_vec4(Vec4::new(
        r.clamp_(0.0, 1.0),
        g.clamp_(0.0, 1.0),
        b.clamp_(0.0, 1.0),
        a.clamp_(0.0, 1.0),
    ))
}

// ---------------------------------------------------------------------------
// Pixel output
// ---------------------------------------------------------------------------

/// Draw a pixel to the back buffer, handling fragment processing, blending,
/// logic ops, and color masking.
///
/// `do_frag_processing` should be true when the fragment shader may write
/// gl_FragDepth or call discard, meaning depth/stencil tests were deferred.
pub fn draw_pixel(c: &mut GlContext, cf: Vec4, x: i32, y: i32, z: f32, do_frag_processing: bool) {
    let buf_w = c.back_buffer.w;
    let buf_h = c.back_buffer.h;

    // Scissor test
    if c.scissor_test {
        if x < c.scissor_lx
            || y < c.scissor_ly
            || x >= c.scissor_lx + c.scissor_w
            || y >= c.scissor_ly + c.scissor_h
        {
            return;
        }
    }

    if x < 0 || x >= buf_w || y < 0 || y >= buf_h {
        return;
    }

    if do_frag_processing {
        if !fragment_processing(c, x, y, z) {
            return;
        }
    }

    let buf_idx = ((buf_h - 1 - y) * buf_w + x) as usize;

    let dest_pixel = read_backbuf_pixel(c, buf_idx);
    let mut pixel_val;

    if c.blend {
        let dst_color = pixel_to_color(dest_pixel).to_vec4();
        let blended = blend_pixel(c, cf, dst_color);
        pixel_val = rgba_to_pixel(blended.r, blended.g, blended.b, blended.a);
    } else {
        let color = Color::from_vec4(cf);
        pixel_val = rgba_to_pixel(color.r, color.g, color.b, color.a);
    }

    if c.logic_ops {
        pixel_val = logic_ops_pixel(c.logic_func, pixel_val, dest_pixel);
    }

    // Apply color mask
    if c.color_mask != 0xFFFFFFFF {
        pixel_val = (pixel_val & c.color_mask) | (dest_pixel & !c.color_mask);
    }

    write_backbuf_pixel(c, buf_idx, pixel_val);
}

// ---------------------------------------------------------------------------
// Setup fragment shader input (interpolation for lines)
// ---------------------------------------------------------------------------

/// Interpolate vertex shader outputs for line fragments.
/// Uses smooth (perspective-correct), noperspective (linear), or flat interpolation.
///
/// `t` is the parametric position along the line [0, 1].
/// `wa`, `wb` are the clip-space w values for perspective correction.
/// `provoke` is the index of the provoking vertex in glverts (for flat shading).
pub fn setup_fs_input(
    c: &mut GlContext,
    t: f32,
    v1_out: &[f32],
    v2_out: &[f32],
    wa: f32,
    wb: f32,
    provoke: usize,
) {
    let vs_output_size = c.vs_output.size as usize;
    let program = &c.programs[c.cur_program as usize];
    let interp = program.interpolation;

    // Get provoking vertex outputs for flat shading
    let provoke_out = if provoke < c.glverts.len() {
        c.glverts[provoke].vs_out.clone()
    } else {
        v1_out.to_vec()
    };

    let one_minus_t = 1.0 - t;

    for j in 0..vs_output_size {
        match interp[j] {
            PGL_SMOOTH => {
                // Perspective-correct interpolation:
                //   val = (a/wa * (1-t) + b/wb * t) / ((1-t)/wa + t/wb)
                let inv_w = one_minus_t / wa + t / wb;
                if inv_w.abs_() > 1e-10 {
                    let val = v1_out[j] / wa * one_minus_t + v2_out[j] / wb * t;
                    c.fs_input[j] = val / inv_w;
                } else {
                    c.fs_input[j] = v1_out[j] * one_minus_t + v2_out[j] * t;
                }
            }
            PGL_NOPERSPECTIVE => {
                c.fs_input[j] = v1_out[j] * one_minus_t + v2_out[j] * t;
            }
            PGL_FLAT => {
                c.fs_input[j] = provoke_out[j];
            }
            _ => {
                c.fs_input[j] = v1_out[j] * one_minus_t + v2_out[j] * t;
            }
        }
    }
}

/// Interpolate vertex shader outputs for triangle fragments using barycentric
/// coordinates and the three vertex outputs.
///
/// `alpha`, `beta`, `gamma` are the barycentric weights.
/// `w0`, `w1`, `w2` are the clip-space w values for perspective correction.
fn setup_fs_input_triangle(
    c: &mut GlContext,
    alpha: f32,
    beta: f32,
    gamma: f32,
    v0_out: &[f32],
    v1_out: &[f32],
    v2_out: &[f32],
    w0: f32,
    w1: f32,
    w2: f32,
    provoke_out: &[f32],
    interp: &[GLenum],
) {
    let vs_output_size = c.vs_output.size as usize;

    for j in 0..vs_output_size {
        match interp[j] {
            PGL_SMOOTH => {
                // Perspective-correct interpolation
                let val = alpha * v0_out[j] / w0
                    + beta * v1_out[j] / w1
                    + gamma * v2_out[j] / w2;
                let w_interp = alpha / w0 + beta / w1 + gamma / w2;
                if w_interp.abs_() > 1e-10 {
                    c.fs_input[j] = val / w_interp;
                } else {
                    c.fs_input[j] =
                        alpha * v0_out[j] + beta * v1_out[j] + gamma * v2_out[j];
                }
            }
            PGL_NOPERSPECTIVE => {
                c.fs_input[j] =
                    alpha * v0_out[j] + beta * v1_out[j] + gamma * v2_out[j];
            }
            PGL_FLAT => {
                c.fs_input[j] = provoke_out[j];
            }
            _ => {
                c.fs_input[j] =
                    alpha * v0_out[j] + beta * v1_out[j] + gamma * v2_out[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Line clipping
// ---------------------------------------------------------------------------

/// Cohen-Sutherland style parametric line clipping helper.
/// Returns true if the line segment is (still) potentially visible.
///
/// Used to clip a parametric line segment [tmin, tmax] against a half-plane.
/// `denom` and `num` define the half-plane equation evaluated at the line.
#[inline]
pub fn clip_line(denom: f32, num: f32, tmin: &mut f32, tmax: &mut f32) -> bool {
    if denom > 0.0 {
        let t = num / denom;
        if t > *tmax {
            return false;
        }
        if t > *tmin {
            *tmin = t;
        }
    } else if denom < 0.0 {
        let t = num / denom;
        if t < *tmin {
            return false;
        }
        if t < *tmax {
            *tmax = t;
        }
    } else if num > 0.0 {
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Line drawing
// ---------------------------------------------------------------------------

/// Clip a line against the view volume, then draw it using `draw_thick_line`.
pub fn draw_line_clip(c: &mut GlContext, v1: &GlVertex, v2: &GlVertex) {
    let cc1 = v1.clip_code;
    let cc2 = v2.clip_code;

    // Trivial reject: both vertices outside the same plane
    if (cc1 & cc2) != 0 {
        return;
    }

    // Compute provoking vertex index (matches C pointer arithmetic)
    let provoke = 0usize; // TODO: proper provoke index from glverts

    // Trivial accept: both inside
    if (cc1 | cc2) == 0 {
        let t1 = mult_m4_v4(c.vp_mat, v1.clip_space);
        let t2 = mult_m4_v4(c.vp_mat, v2.clip_space);

        let hp1 = v4_to_v3h(t1);
        let hp2 = v4_to_v3h(t2);

        if c.line_smooth {
            draw_aa_line(
                c,
                hp1, hp2, t1.w, t2.w,
                &v1.vs_out, &v2.vs_out, provoke, 0.0,
            );
        } else {
            draw_thick_line(
                c,
                hp1, hp2, t1.w, t2.w,
                &v1.vs_out, &v2.vs_out, provoke, 0.0,
            );
        }
        return;
    }

    // Parametric clipping against all 6 frustum planes
    let d = Vec4::sub(v2.clip_space, v1.clip_space);
    let p = v1.clip_space;

    let mut tmin = 0.0f32;
    let mut tmax = 1.0f32;

    if !clip_line(d.x + d.w, -p.x - p.w, &mut tmin, &mut tmax) { return; }
    if !clip_line(-d.x + d.w, p.x - p.w, &mut tmin, &mut tmax) { return; }
    if !clip_line(d.y + d.w, -p.y - p.w, &mut tmin, &mut tmax) { return; }
    if !clip_line(-d.y + d.w, p.y - p.w, &mut tmin, &mut tmax) { return; }
    if !clip_line(d.z + d.w, -p.z - p.w, &mut tmin, &mut tmax) { return; }
    if !clip_line(-d.z + d.w, p.z - p.w, &mut tmin, &mut tmax) { return; }

    // Compute clipped endpoints in clip space
    let c1 = Vec4::add(p, d * tmin);
    let c2 = Vec4::add(p, d * tmax);

    let t1 = mult_m4_v4(c.vp_mat, c1);
    let t2 = mult_m4_v4(c.vp_mat, c2);

    let hp1 = v4_to_v3h(t1);
    let hp2 = v4_to_v3h(t2);

    // Interpolate vs_out for clipped endpoints
    let vs_size = c.vs_output.size as usize;
    let mut v1_out_clipped = vec![0.0f32; vs_size];
    let mut v2_out_clipped = vec![0.0f32; vs_size];

    for j in 0..vs_size {
        v1_out_clipped[j] = v1.vs_out[j] + (v2.vs_out[j] - v1.vs_out[j]) * tmin;
        v2_out_clipped[j] = v1.vs_out[j] + (v2.vs_out[j] - v1.vs_out[j]) * tmax;
    }

    if c.line_smooth {
        draw_aa_line(
            c,
            hp1, hp2, t1.w, t2.w,
            &v1_out_clipped, &v2_out_clipped, provoke, 0.0,
        );
    } else {
        draw_thick_line(
            c,
            hp1, hp2, t1.w, t2.w,
            &v1_out_clipped, &v2_out_clipped, provoke, 0.0,
        );
    }
}

/// Draw a line with width support using midpoint line algorithm.
///
/// Matches the C PortableGL algorithm: 4 slope cases with implicit line
/// function for stepping decisions.
#[cfg(not(feature = "better_thick_lines"))]
pub fn draw_thick_line(
    c: &mut GlContext,
    hp1: Vec3,
    hp2: Vec3,
    mut w1: f32,
    mut w2: f32,
    v1_out: &[f32],
    v2_out: &[f32],
    provoke: usize,
    poly_offset: f32,
) {
    let mut x1 = hp1.x;
    let mut y1 = hp1.y;
    let mut z1 = hp1.z;
    let mut x2 = hp2.x;
    let mut y2 = hp2.y;
    let mut z2 = hp2.z;

    // Always draw from left to right
    let mut out_a = v1_out;
    let mut out_b = v2_out;
    if x2 < x1 {
        core::mem::swap(&mut x1, &mut x2);
        core::mem::swap(&mut y1, &mut y2);
        core::mem::swap(&mut z1, &mut z2);
        core::mem::swap(&mut w1, &mut w2);
        core::mem::swap(&mut out_a, &mut out_b);
    }

    let m = (y2 - y1) / (x2 - x1);
    let line = Line::new(x1, y1, x2, y2);

    let p1x = x1;
    let p1y = y1;
    let sub_x = x2 - x1;
    let sub_y = y2 - y1;
    let line_length_squared = sub_x * sub_x + sub_y * sub_y;

    let vs_output_size = c.vs_output.size as usize;
    let fragdepth_or_discard = c.fragdepth_or_discard;
    let program_idx = c.cur_program as usize;

    let i_x1 = x1.floor_() + 0.5;
    let i_y1 = y1.floor_() + 0.5;
    let i_x2 = x2.floor_() + 0.5;
    let i_y2 = y2.floor_() + 0.5;



    let x_min = i_x1;
    let x_max = i_x2;
    let (y_min, y_max) = if m <= 0.0 {
        (i_y2, i_y1)
    } else {
        (i_y1, i_y2)
    };

    // Map z to depth range
    z1 = rsw_mapf(z1, -1.0, 1.0, c.depth_range_near, c.depth_range_far);
    z2 = rsw_mapf(z2, -1.0, 1.0, c.depth_range_near, c.depth_range_far);

    let width = c.line_width.round_();
    let width = if width == 0.0 { 1.0 } else { width };
    let half_w = width * 0.5;

    // Helper macro-like closure for the inner fragment loop
    // We use a nested function approach to avoid code duplication across 4 cases
    macro_rules! process_fragment {
        ($c:expr, $fx:expr, $fy:expr, $z:expr, $w:expr, $t:expr) => {{
            // Use float comparison for clip test, matching C's CLIPXY_TEST macro
            // which compares float coords against int bounds (int promoted to float)
            let fx_val = $fx;
            let fy_val = $fy;
            let jx = fx_val as i32;
            let jy = fy_val as i32;
            if fx_val >= $c.lx as f32 && fx_val < $c.ux as f32 && fy_val >= $c.ly as f32 && fy_val < $c.uy as f32 {
                if fragdepth_or_discard || fragment_processing($c, jx, jy, $z) {
                    $c.builtins.gl_FragCoord = Vec4::new($fx, $fy, $z, 1.0 / $w);
                    $c.builtins.discard = false;
                    $c.builtins.gl_FragDepth = $z;
                    setup_fs_input($c, $t, out_a, out_b, w1, w2, provoke);

                    let program = &$c.programs[program_idx];
                    let fs = program.fragment_shader;
                    let uniform = program.uniform;
                    let mut fs_input_copy: Vec<f32> = $c.fs_input[..vs_output_size].to_vec();
                    unsafe {
                        (fs)(
                            fs_input_copy.as_mut_ptr(),
                            &mut $c.builtins as *mut ShaderBuiltins,
                            uniform,
                        );
                    }
                    if !$c.builtins.discard {
                        draw_pixel($c, $c.builtins.gl_FragColor, jx, jy,
                            $c.builtins.gl_FragDepth, fragdepth_or_discard);
                    }
                }
            }
        }};
    }

    if m <= -1.0 {
        // Slope in (-inf, -1]: step along y (decreasing), x increases
        let mut x = x_min;
        let mut y = y_max;
        while y >= y_min && x <= x_max {
            let pr_x = x;
            let pr_y = y;
            let mut t = ((pr_x - p1x) * sub_x + (pr_y - p1y) * sub_y) / line_length_squared;
            t = clamp_01(t);
            let z = (1.0 - t) * z1 + t * z2 + poly_offset;
            let w = (1.0 - t) * w1 + t * w2;

            let mut j = x - half_w;
            while j < x + half_w {
                process_fragment!(c, j, y, z, w, t);
                j += 1.0;
            }

            if line.func(x + 0.5, y - 1.0) < 0.0 {
                x += 1.0;
            }
            y -= 1.0;
        }
    } else if m <= 0.0 {
        // Slope in (-1, 0]: step along x, y decreases
        let mut x = x_min;
        let mut y = y_max;
        while x <= x_max && y >= y_min {
            let pr_x = x;
            let pr_y = y;
            let mut t = ((pr_x - p1x) * sub_x + (pr_y - p1y) * sub_y) / line_length_squared;
            t = clamp_01(t);
            let z = (1.0 - t) * z1 + t * z2 + poly_offset;
            let w = (1.0 - t) * w1 + t * w2;

            let mut j = y - half_w;
            while j < y + half_w {
                process_fragment!(c, x, j, z, w, t);
                j += 1.0;
            }

            if line.func(x + 1.0, y - 0.5) > 0.0 {
                y -= 1.0;
            }
            x += 1.0;
        }
    } else if m <= 1.0 {
        // Slope in (0, 1]: step along x, y increases
        let mut x = x_min;
        let mut y = y_min;
        while x <= x_max && y <= y_max {
            let pr_x = x;
            let pr_y = y;
            let mut t = ((pr_x - p1x) * sub_x + (pr_y - p1y) * sub_y) / line_length_squared;
            t = clamp_01(t);
            let z = (1.0 - t) * z1 + t * z2 + poly_offset;
            let w = (1.0 - t) * w1 + t * w2;

            let mut j = y - half_w;
            while j < y + half_w {
                process_fragment!(c, x, j, z, w, t);
                j += 1.0;
            }

            if line.func(x + 1.0, y + 0.5) < 0.0 {
                y += 1.0;
            }
            x += 1.0;
        }
    } else {
        // Slope in (1, +inf): step along y (increasing), x increases
        let mut x = x_min;
        let mut y = y_min;
        while y <= y_max && x <= x_max {
            let pr_x = x;
            let pr_y = y;
            let mut t = ((pr_x - p1x) * sub_x + (pr_y - p1y) * sub_y) / line_length_squared;
            t = clamp_01(t);
            let z = (1.0 - t) * z1 + t * z2 + poly_offset;
            let w = (1.0 - t) * w1 + t * w2;

            let mut j = x - half_w;
            while j < x + half_w {
                process_fragment!(c, j, y, z, w, t);
                j += 1.0;
            }

            if line.func(x + 0.5, y + 1.0) > 0.0 {
                x += 1.0;
            }
            y += 1.0;
        }
    }
}

/// Anti-aliased line drawing using Xiaolin Wu's algorithm.
///
/// Used when `GL_LINE_SMOOTH` is enabled. Only supports width 1 for now.
/// Uses coverage-based alpha blending for anti-aliased edges.
pub fn draw_aa_line(
    c: &mut GlContext,
    hp1: Vec3,
    hp2: Vec3,
    mut w1: f32,
    mut w2: f32,
    v1_out: &[f32],
    v2_out: &[f32],
    provoke: usize,
    poly_offset: f32,
) {
    let vs_output_size = c.vs_output.size as usize;
    let fragdepth_or_discard = c.fragdepth_or_discard;
    let program_idx = c.cur_program as usize;

    let mut x1 = hp1.x;
    let mut y1 = hp1.y;
    let mut z1 = hp1.z;
    let mut x2 = hp2.x;
    let mut y2 = hp2.y;
    let mut z2 = hp2.z;

    let mut out_a = v1_out;
    let mut out_b = v2_out;

    // Wu's algorithm helpers (matching C macros)
    #[inline(always)]
    fn ipart(x: f32) -> i32 { x as i32 }
    #[inline(always)]
    fn round(x: f32) -> i32 { (x + 0.5) as i32 }
    #[inline(always)]
    fn fpart(x: f32) -> f32 { x - (x as i32) as f32 }
    #[inline(always)]
    fn rfpart(x: f32) -> f32 { 1.0 - fpart(x) }

    // Macro for the AA fragment processing (runs shader, applies coverage to alpha)
    macro_rules! aa_fragment {
        ($c:expr, $x:expr, $y:expr, $z:expr, $w:expr, $t:expr, $coverage:expr) => {{
            let fx = $x as f32;
            let fy = $y as f32;
            if fx >= $c.lx as f32 && fx < $c.ux as f32 && fy >= $c.ly as f32 && fy < $c.uy as f32 {
                if fragdepth_or_discard || fragment_processing($c, $x, $y, $z) {
                    $c.builtins.gl_FragCoord = Vec4::new(fx, fy, $z, 1.0 / $w);
                    $c.builtins.discard = false;
                    $c.builtins.gl_FragDepth = $z;
                    setup_fs_input($c, $t, out_a, out_b, w1, w2, provoke);

                    let program = &$c.programs[program_idx];
                    let fs = program.fragment_shader;
                    let uniform = program.uniform;
                    let mut fs_input_copy: Vec<f32> = $c.fs_input[..vs_output_size].to_vec();
                    unsafe {
                        (fs)(
                            fs_input_copy.as_mut_ptr(),
                            &mut $c.builtins as *mut ShaderBuiltins,
                            uniform,
                        );
                    }
                    if !$c.builtins.discard {
                        $c.builtins.gl_FragColor.w *= $coverage;
                        draw_pixel($c, $c.builtins.gl_FragColor, $x, $y,
                            $c.builtins.gl_FragDepth, fragdepth_or_discard);
                    }
                }
            }
        }};
    }

    let dx = x2 - x1;
    let dy = y2 - y1;

    if dx.abs_() > dy.abs_() {
        // Mostly horizontal: sort left to right
        if x2 < x1 {
            core::mem::swap(&mut x1, &mut x2);
            core::mem::swap(&mut y1, &mut y2);
            core::mem::swap(&mut z1, &mut z2);
            core::mem::swap(&mut w1, &mut w2);
            core::mem::swap(&mut out_a, &mut out_b);
        }

        let p1x = x1;
        let p1y = y1;
        let sub_x = x2 - x1;
        let sub_y = y2 - y1;
        let line_length_squared = sub_x * sub_x + sub_y * sub_y;

        z1 = rsw_mapf(z1, -1.0, 1.0, c.depth_range_near, c.depth_range_far);
        z2 = rsw_mapf(z2, -1.0, 1.0, c.depth_range_near, c.depth_range_far);

        let gradient = dy / dx;
        let xend = round(x1);
        let yend = y1 + gradient * (xend as f32 - x1);
        let xgap = rfpart(x1 + 0.5);
        let xpxl1 = xend;
        let ypxl1 = ipart(yend);

        // First endpoint
        let t = 0.0f32;
        let z = z1 + poly_offset;
        let w = w1;
        aa_fragment!(c, xpxl1, ypxl1, z, w, t, rfpart(yend) * xgap);
        aa_fragment!(c, xpxl1, ypxl1 + 1, z, w, t, fpart(yend) * xgap);

        let mut intery = yend + gradient;

        // Second endpoint
        let xend2 = round(x2);
        let yend2 = y2 + gradient * (xend2 as f32 - x2);
        let xgap2 = fpart(x2 + 0.5);
        let xpxl2 = xend2;
        let ypxl2 = ipart(yend2);

        let t2 = 1.0f32;
        let z2_mapped = z2 + poly_offset;
        let w2_val = w2;
        aa_fragment!(c, xpxl2, ypxl2, z2_mapped, w2_val, t2, rfpart(yend2) * xgap2);
        aa_fragment!(c, xpxl2, ypxl2 + 1, z2_mapped, w2_val, t2, fpart(yend2) * xgap2);

        // Main loop
        for xi in (xpxl1 + 1)..xpxl2 {
            let pr_x = xi as f32;
            let pr_y = intery;
            let t = ((pr_x - p1x) * sub_x + (pr_y - p1y) * sub_y) / line_length_squared;
            let z = (1.0 - t) * z1 + t * z2 + poly_offset;
            let w = (1.0 - t) * w1 + t * w2;

            let yi = ipart(intery);
            aa_fragment!(c, xi, yi, z, w, t, rfpart(intery));
            aa_fragment!(c, xi, yi + 1, z, w, t, fpart(intery));

            intery += gradient;
        }
    } else {
        // Mostly vertical: sort bottom to top
        if y2 < y1 {
            core::mem::swap(&mut x1, &mut x2);
            core::mem::swap(&mut y1, &mut y2);
            core::mem::swap(&mut z1, &mut z2);
            core::mem::swap(&mut w1, &mut w2);
            core::mem::swap(&mut out_a, &mut out_b);
        }

        let p1x = x1;
        let p1y = y1;
        let sub_x = x2 - x1;
        let sub_y = y2 - y1;
        let line_length_squared = sub_x * sub_x + sub_y * sub_y;

        z1 = rsw_mapf(z1, -1.0, 1.0, c.depth_range_near, c.depth_range_far);
        z2 = rsw_mapf(z2, -1.0, 1.0, c.depth_range_near, c.depth_range_far);

        let gradient = dx / dy;
        let yend = round(y1);
        let xend = x1 + gradient * (yend as f32 - y1);
        let ygap = rfpart(y1 + 0.5);
        let ypxl1 = yend;
        let xpxl1 = ipart(xend);

        // First endpoint
        let t = 0.0f32;
        let z = z1 + poly_offset;
        let w = w1;
        aa_fragment!(c, xpxl1, ypxl1, z, w, t, rfpart(xend) * ygap);
        aa_fragment!(c, xpxl1 + 1, ypxl1, z, w, t, fpart(xend) * ygap);

        let mut interx = xend + gradient;

        // Second endpoint
        let yend2 = round(y2);
        let xend2 = x2 + gradient * (yend2 as f32 - y2);
        let ygap2 = fpart(y2 + 0.5);
        let ypxl2 = yend2;
        let xpxl2 = ipart(xend2);

        let t2 = 1.0f32;
        let z2_mapped = z2 + poly_offset;
        let w2_val = w2;
        aa_fragment!(c, xpxl2, ypxl2, z2_mapped, w2_val, t2, rfpart(xend2) * ygap2);
        aa_fragment!(c, xpxl2 + 1, ypxl2, z2_mapped, w2_val, t2, fpart(xend2) * ygap2);

        // Main loop
        for yi in (ypxl1 + 1)..ypxl2 {
            let pr_x = interx;
            let pr_y = yi as f32;
            let t = ((pr_x - p1x) * sub_x + (pr_y - p1y) * sub_y) / line_length_squared;
            let z = (1.0 - t) * z1 + t * z2 + poly_offset;
            let w = (1.0 - t) * w1 + t * w2;

            let xi = ipart(interx);
            aa_fragment!(c, xi, yi, z, w, t, rfpart(interx));
            aa_fragment!(c, xi + 1, yi, z, w, t, fpart(interx));

            interx += gradient;
        }
    }
}

/// "Better" thick line implementation using perpendicular expansion.
///
/// Instead of just duplicating scanlines, this expands the line into a
/// quad perpendicular to the line direction, giving more visually correct
/// thick lines (especially for diagonal lines).
#[cfg(feature = "better_thick_lines")]
pub fn draw_thick_line(
    c: &mut GlContext,
    hp1: Vec3,
    hp2: Vec3,
    w1: f32,
    w2: f32,
    v1_out: &[f32],
    v2_out: &[f32],
    provoke: usize,
    poly_offset: f32,
) {
    let x1 = hp1.x;
    let y1 = hp1.y;
    let z1 = hp1.z;
    let x2 = hp2.x;
    let y2 = hp2.y;
    let z2 = hp2.z;

    let line_w = c.line_width.max_(1.0);
    let half_w = line_w * 0.5;

    let vs_output_size = c.vs_output.size as usize;
    let fragdepth_or_discard = c.fragdepth_or_discard;
    let depth_range_near = c.depth_range_near;
    let depth_range_far = c.depth_range_far;
    let program_idx = c.cur_program as usize;

    // Compute perpendicular direction
    let dx = x2 - x1;
    let dy = y2 - y1;
    let line_len = (dx * dx + dy * dy).sqrt_();
    if line_len < 0.001 {
        return;
    }

    // Normal to the line direction
    let nx = -dy / line_len;
    let ny = dx / line_len;

    // Expand into a quad: 4 corners
    let px0 = x1 + nx * half_w;
    let py0 = y1 + ny * half_w;
    let px1 = x1 - nx * half_w;
    let py1 = y1 - ny * half_w;
    let px2 = x2 + nx * half_w;
    let py2 = y2 + ny * half_w;
    let px3 = x2 - nx * half_w;
    let py3 = y2 - ny * half_w;

    // Bounding box
    let min_x = px0.min_(px1).min_(px2).min_(px3).floor_() as i32;
    let max_x = px0.max_(px1).max_(px2).max_(px3).ceil_() as i32;
    let min_y = py0.min_(py1).min_(py2).min_(py3).floor_() as i32;
    let max_y = py0.max_(py1).max_(py2).max_(py3).ceil_() as i32;

    let min_x = min_x.max(c.lx);
    let max_x = max_x.min(c.ux);
    let min_y = min_y.max(c.ly);
    let max_y = max_y.min(c.uy);

    // Line direction unit vector
    let ldx = dx / line_len;
    let ldy = dy / line_len;

    for y in min_y..max_y {
        for x in min_x..max_x {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            // Project point onto line to get parametric t
            let t = ((px - x1) * ldx + (py - y1) * ldy) / line_len;
            if t < 0.0 || t > 1.0 {
                continue;
            }

            // Distance from line
            let dist = ((px - x1) * nx + (py - y1) * ny).abs_();
            if dist > half_w {
                continue;
            }

            let z = z1 * (1.0 - t) + z2 * t;
            let z_mapped =
                depth_range_near + (depth_range_far - depth_range_near) * (z + 1.0) * 0.5
                    + poly_offset;

            if !fragdepth_or_discard {
                if !fragment_processing(c, x, y, z_mapped) {
                    continue;
                }
            }

            setup_fs_input(c, t, v1_out, v2_out, w1, w2, provoke);

            c.builtins.gl_FragCoord = Vec4::new(px, py, z_mapped, 1.0);
            c.builtins.gl_FragDepth = z_mapped;
            c.builtins.discard = false;

            let program = &c.programs[program_idx];
            let fs = program.fragment_shader;
            let uniform = program.uniform;
            let mut fs_input_copy: Vec<f32> = c.fs_input[..vs_output_size].to_vec();

            unsafe {
                (fs)(
                    fs_input_copy.as_mut_ptr(),
                    &mut c.builtins as *mut ShaderBuiltins,
                    uniform,
                );
            }

            if c.builtins.discard {
                continue;
            }

            let cf = c.builtins.gl_FragColor;
            let final_z = if fragdepth_or_discard {
                c.builtins.gl_FragDepth
            } else {
                z_mapped
            };

            draw_pixel(c, cf, x, y, final_z, fragdepth_or_discard);
        }
    }
}

// ---------------------------------------------------------------------------
// Point drawing
// ---------------------------------------------------------------------------

/// Draw a point sprite with configurable size.
///
/// Points are drawn as axis-aligned squares centered on the vertex position.
/// gl_PointCoord is set up for each fragment.
pub fn draw_point(c: &mut GlContext, vert: &GlVertex, poly_offset: f32) {
    let point = v4_to_v3h(vert.screen_space);
    let z = rsw_mapf(
        point.z + poly_offset,
        -1.0,
        1.0,
        c.depth_range_near,
        c.depth_range_far,
    );

    let vs_output_size = c.vs_output.size as usize;
    let fragdepth_or_discard = c.fragdepth_or_discard;

    let p_size = c.point_size;
    let origin: f32 = if c.point_spr_origin == GL_UPPER_LEFT { -1.0 } else { 1.0 };

    // Accounting for pixel centers at 0.5, using truncation
    let x = point.x + 0.5;
    let y = point.y + 0.5;

    // Can easily clip whole point when point size <= 1
    // Use float comparison matching C's CLIPXY_TEST behavior
    if p_size <= 1.0 {
        if x < c.lx as f32 || y < c.ly as f32 || x >= c.ux as f32 || y >= c.uy as f32 {
            return;
        }
    }

    let program_idx = c.cur_program as usize;
    let half_ps = p_size / 2.0;

    let mut fi = y - half_ps;
    while fi < y + half_ps {
        if fi < c.ly as f32 || fi >= c.uy as f32 {
            fi += 1.0;
            continue;
        }

        let mut fj = x - half_ps;
        while fj < x + half_ps {
            if fj < c.lx as f32 || fj >= c.ux as f32 {
                fj += 1.0;
                continue;
            }

            let px = fj as i32;
            let py = fi as i32;

            if !fragdepth_or_discard {
                if !fragment_processing(c, px, py, z) {
                    fj += 1.0;
                    continue;
                }
            }

            // Compute gl_PointCoord per spec
            let pcx = 0.5 + (px as f32 + 0.5 - point.x) / p_size;
            let pcy = 0.5 + origin * (py as f32 + 0.5 - point.y) / p_size;
            c.builtins.gl_PointCoord = Vec2::new(pcx, pcy);

            // Copy vertex outputs to fs_input (points use flat shading for all)
            for j in 0..vs_output_size.min(vert.vs_out.len()) {
                c.fs_input[j] = vert.vs_out[j];
            }

            c.builtins.gl_FragCoord =
                Vec4::new(fj, fi, z, 1.0 / vert.screen_space.w);
            c.builtins.gl_FragDepth = z;
            c.builtins.discard = false;

            let program = &c.programs[program_idx];
            let fs = program.fragment_shader;
            let uniform = program.uniform;
            let mut fs_input_copy: Vec<f32> = c.fs_input[..vs_output_size].to_vec();

            unsafe {
                (fs)(
                    fs_input_copy.as_mut_ptr(),
                    &mut c.builtins as *mut ShaderBuiltins,
                    uniform,
                );
            }

            if c.builtins.discard {
                fj += 1.0;
                continue;
            }

            let cf = c.builtins.gl_FragColor;
            let final_z = if fragdepth_or_discard {
                c.builtins.gl_FragDepth
            } else {
                z
            };

            draw_pixel(c, cf, px, py, final_z, fragdepth_or_discard);
            fj += 1.0;
        }
        fi += 1.0;
    }
}

// ---------------------------------------------------------------------------
// Polygon offset
// ---------------------------------------------------------------------------

/// Calculate polygon offset from depth slopes of a triangle.
///
/// `hp0`, `hp1`, `hp2` are screen-space positions.
/// Returns `factor * max_slope + units * depth_resolution`.
pub fn calc_poly_offset(
    hp0: Vec3,
    hp1: Vec3,
    hp2: Vec3,
    poly_factor: f32,
    poly_units: f32,
) -> f32 {
    let dx01 = hp1.x - hp0.x;
    let dy01 = hp1.y - hp0.y;
    let dz01 = hp1.z - hp0.z;
    let dx02 = hp2.x - hp0.x;
    let dy02 = hp2.y - hp0.y;
    let dz02 = hp2.z - hp0.z;

    let det = dx01 * dy02 - dx02 * dy01;
    if det.abs_() < 1e-10 {
        return poly_units * POLYGON_OFFSET_UNIT_INCR;
    }

    let inv_det = 1.0 / det;
    let dz_dx = (dz01 * dy02 - dz02 * dy01) * inv_det;
    let dz_dy = (dx01 * dz02 - dx02 * dz01) * inv_det;

    let max_slope = dz_dx.abs_().max_(dz_dy.abs_());

    poly_factor * max_slope + poly_units * POLYGON_OFFSET_UNIT_INCR
}

// ---------------------------------------------------------------------------
// Triangle functions
// ---------------------------------------------------------------------------

/// Entry point for triangle drawing - checks clip codes, dispatches to
/// draw_triangle_final or draw_triangle_clip.
pub fn draw_triangle(
    c: &mut GlContext,
    v0_idx: usize,
    v1_idx: usize,
    v2_idx: usize,
    provoke: usize,
) {
    let cc0 = c.glverts[v0_idx].clip_code;
    let cc1 = c.glverts[v1_idx].clip_code;
    let cc2 = c.glverts[v2_idx].clip_code;

    // Trivial reject: all three outside the same plane
    if (cc0 & cc1 & cc2) != 0 {
        return;
    }

    // Set edge flags (needed for GL_LINE polygon mode wireframe rendering).
    // Must set here because vertices can be reused for multiple triangles
    // in STRIP and FAN modes.
    c.glverts[v0_idx].edge_flag = 1;
    c.glverts[v1_idx].edge_flag = 1;
    c.glverts[v2_idx].edge_flag = 1;

    // Trivial accept: all three inside
    if (cc0 | cc1 | cc2) == 0 {
        let v0 = c.glverts[v0_idx].clone();
        let v1 = c.glverts[v1_idx].clone();
        let v2 = c.glverts[v2_idx].clone();
        draw_triangle_final(c, &v0, &v1, &v2, provoke);
    } else {
        // Need clipping
        let v0 = c.glverts[v0_idx].clone();
        let v1 = c.glverts[v1_idx].clone();
        let v2 = c.glverts[v2_idx].clone();
        draw_triangle_clip(c, v0, v1, v2, provoke, 1);
    }
}

/// Apply viewport transform, face culling, then dispatch to fill/line/point rasterizer.
pub fn draw_triangle_final(
    c: &mut GlContext,
    v0: &GlVertex,
    v1: &GlVertex,
    v2: &GlVertex,
    provoke: usize,
) {
    // Compute screen space: viewport transform on clip space (preserves w for perspective)
    // v4_to_v3h(screen_space) gives pixel coords, screen_space.w = clip_space.w
    let mut sv0 = v0.clone();
    let mut sv1 = v1.clone();
    let mut sv2 = v2.clone();

    sv0.screen_space = mult_m4_v4(c.vp_mat, v0.clip_space);
    sv1.screen_space = mult_m4_v4(c.vp_mat, v1.clip_space);
    sv2.screen_space = mult_m4_v4(c.vp_mat, v2.clip_space);

    // Face culling
    let front_facing = is_front_facing(&sv0, &sv1, &sv2, c.front_face);

    if c.cull_face {
        if c.cull_mode == GL_FRONT && front_facing {
            return;
        }
        if c.cull_mode == GL_BACK && !front_facing {
            return;
        }
        if c.cull_mode == GL_FRONT_AND_BACK {
            return;
        }
    }

    c.builtins.gl_FrontFacing = front_facing;

    // Select polygon mode based on which face we're drawing
    let tri_func = if front_facing {
        c.draw_triangle_front
    } else {
        c.draw_triangle_back
    };

    match tri_func {
        TRIANGLE_FILL => draw_triangle_fill(c, &sv0, &sv1, &sv2, provoke),
        TRIANGLE_LINE => draw_triangle_line(c, &sv0, &sv1, &sv2, provoke),
        TRIANGLE_POINT => draw_triangle_point(c, &sv0, &sv1, &sv2, provoke),
        _ => draw_triangle_fill(c, &sv0, &sv1, &sv2, provoke),
    }
}

/// Sutherland-Hodgman triangle clipping against 6 frustum planes.
///
/// Recursively tests each clip plane (bit 1 through 32). When a plane
/// clips the triangle, it produces 1 or 2 sub-triangles that are passed
/// to the next plane.
pub fn draw_triangle_clip(
    c: &mut GlContext,
    v0: GlVertex,
    v1: GlVertex,
    v2: GlVertex,
    provoke: usize,
    clip_bit: i32,
) {
    // If we've tested all clip planes, draw the triangle
    if clip_bit > 32 {
        draw_triangle_final(c, &v0, &v1, &v2, provoke);
        return;
    }

    let cc = v0.clip_code | v1.clip_code | v2.clip_code;

    // If this plane doesn't affect any vertex, skip to next
    if (cc & clip_bit) == 0 {
        draw_triangle_clip(c, v0, v1, v2, provoke, clip_bit << 1);
        return;
    }

    // All three outside this plane - reject
    if (v0.clip_code & v1.clip_code & v2.clip_code & clip_bit) != 0 {
        return;
    }

    // Determine which vertices are inside/outside for this plane
    let v0_outside = (v0.clip_code & clip_bit) != 0;
    let v1_outside = (v1.clip_code & clip_bit) != 0;
    let v2_outside = (v2.clip_code & clip_bit) != 0;

    let num_outside = v0_outside as u32 + v1_outside as u32 + v2_outside as u32;

    if num_outside == 1 {
        // One vertex outside: clip into 2 triangles
        // Rotate so the outside vertex is q[0], inside are q[1], q[2]
        let (q0, q1, mut q2) = if v0_outside {
            (v0, v1, v2)
        } else if v1_outside {
            (v1, v2, v0)
        } else {
            (v2, v0, v1)
        };

        let t1 = clip_lerp_factor(clip_bit, &q0.clip_space, &q1.clip_space);
        let t2 = clip_lerp_factor(clip_bit, &q0.clip_space, &q2.clip_space);

        let mut tmp1 = interpolate_vertex(&q0, &q1, t1, c.depth_clamp);
        let mut tmp2 = interpolate_vertex(&q0, &q2, t2, c.depth_clamp);

        // Edge flag management matching C's draw_triangle_clip:
        // tmp1 gets q[0]'s edge flag (it's on the q[0]→q[1] edge)
        tmp1.edge_flag = q0.edge_flag;
        let edge_flag_tmp = q2.edge_flag;
        q2.edge_flag = 0; // suppress internal edge for first sub-tri
        draw_triangle_clip(c, tmp1.clone(), q1, q2.clone(), provoke, clip_bit << 1);

        tmp2.edge_flag = 0; // internal clip edge
        tmp1.edge_flag = 0; // internal edge between sub-triangles
        q2.edge_flag = edge_flag_tmp; // restore
        draw_triangle_clip(c, tmp2, tmp1, q2, provoke, clip_bit << 1);
    } else if num_outside == 2 {
        // Two vertices outside: clip into 1 triangle
        // Rotate so the inside vertex is q[0], outside are q[1], q[2]
        let (q0, q1, q2) = if !v0_outside {
            (v0, v1, v2)
        } else if !v1_outside {
            (v1, v2, v0)
        } else {
            (v2, v0, v1)
        };

        let t1 = clip_lerp_factor(clip_bit, &q0.clip_space, &q1.clip_space);
        let t2 = clip_lerp_factor(clip_bit, &q0.clip_space, &q2.clip_space);

        let mut tmp1 = interpolate_vertex(&q0, &q1, t1, c.depth_clamp);
        let mut tmp2 = interpolate_vertex(&q0, &q2, t2, c.depth_clamp);

        // Edge flag management:
        tmp1.edge_flag = 0; // internal clip edge
        tmp2.edge_flag = q2.edge_flag; // preserves original edge flag
        draw_triangle_clip(c, q0, tmp1, tmp2, provoke, clip_bit << 1);
    }
}

/// Compute the parametric lerp factor for clipping against a specific plane.
///
/// Each clip_bit corresponds to a frustum plane. We solve for the parameter `t`
/// where the edge from `a` to `b` intersects that plane.
fn clip_lerp_factor(clip_bit: i32, a: &Vec4, b: &Vec4) -> f32 {
    let d = Vec4::sub(*b, *a);
    let t = match clip_bit {
        1 => {
            // z < -w  =>  z + w = 0
            (-a.z - a.w) / (d.z + d.w)
        }
        2 => {
            // z > w  =>  z - w = 0
            (a.z - a.w) / (-d.z + d.w)
        }
        4 => {
            // x < -w  =>  x + w = 0
            (-a.x - a.w) / (d.x + d.w)
        }
        8 => {
            // x > w  =>  x - w = 0
            (a.x - a.w) / (-d.x + d.w)
        }
        16 => {
            // y < -w  =>  y + w = 0
            (-a.y - a.w) / (d.y + d.w)
        }
        32 => {
            // y > w  =>  y - w = 0
            (a.y - a.w) / (-d.y + d.w)
        }
        _ => 0.0,
    };
    t.clamp_(0.0, 1.0)
}

/// Interpolate between two vertices at parameter `t` (0 = v0, 1 = v1).
/// The resulting vertex has new clip-space coordinates and interpolated shader outputs.
fn interpolate_vertex(v0: &GlVertex, v1: &GlVertex, t: f32, depth_clamp: bool) -> GlVertex {
    let clip_space = Vec4::add(v0.clip_space, Vec4::sub(v1.clip_space, v0.clip_space) * t);

    let vs_size = v0.vs_out.len();
    let mut vs_out = vec![0.0f32; vs_size];
    for j in 0..vs_size {
        vs_out[j] = v0.vs_out[j] + (v1.vs_out[j] - v0.vs_out[j]) * t;
    }

    let clip_code = gl_clipcode(clip_space, depth_clamp);

    GlVertex {
        clip_space,
        screen_space: Vec4::default(),
        clip_code,
        edge_flag: v0.edge_flag,
        vs_out,
    }
}

// ---------------------------------------------------------------------------
// Triangle rasterization (fill)
// ---------------------------------------------------------------------------

/// Scanline triangle rasterization using barycentric coordinates with
/// edge equations.
///
/// This is the main filled-triangle rasterizer. It computes implicit line
/// equations for each edge, iterates over the bounding box, and for each
/// pixel inside the triangle:
///   1. Computes barycentric coordinates
///   2. Interpolates z and vertex shader outputs (perspective-correct, linear, or flat)
///   3. Runs the fragment shader
///   4. Performs depth/stencil test and blending
///   5. Writes to the back buffer
pub fn draw_triangle_fill(
    c: &mut GlContext,
    v0: &GlVertex,
    v1: &GlVertex,
    v2: &GlVertex,
    provoke: usize,
) {
    let hp0 = v4_to_v3h(v0.screen_space);
    let hp1 = v4_to_v3h(v1.screen_space);
    let hp2 = v4_to_v3h(v2.screen_space);

    // Compute polygon offset if enabled
    let poly_offset = if c.poly_offset_fill {
        calc_poly_offset(hp0, hp1, hp2, c.poly_factor, c.poly_units)
    } else {
        0.0
    };

    // Bounding box
    let min_x = hp0.x.min_(hp1.x).min_(hp2.x).floor_() as i32;
    let max_x = hp0.x.max_(hp1.x).max_(hp2.x).ceil_() as i32;
    let min_y = hp0.y.min_(hp1.y).min_(hp2.y).floor_() as i32;
    let max_y = hp0.y.max_(hp1.y).max_(hp2.y).ceil_() as i32;

    // Clamp to viewport/scissor bounds
    let min_x = min_x.max(c.lx);
    let max_x = max_x.min(c.ux);
    let min_y = min_y.max(c.ly);
    let max_y = max_y.min(c.uy);

    if min_x >= max_x || min_y >= max_y {
        return;
    }

    // Setup edge equations (implicit line functions)
    // l12 is the edge opposite v0, l20 opposite v1, l01 opposite v2.
    let l01 = Line::new(hp0.x, hp0.y, hp1.x, hp1.y);
    let l12 = Line::new(hp1.x, hp1.y, hp2.x, hp2.y);
    let l20 = Line::new(hp2.x, hp2.y, hp0.x, hp0.y);

    // Evaluate at the opposite vertex to get the denominator for barycentric coords
    let alpha_denom = l12.func(hp0.x, hp0.y);
    let beta_denom = l20.func(hp1.x, hp1.y);
    let gamma_denom = l01.func(hp2.x, hp2.y);

    if alpha_denom.abs_() < 1e-10 || beta_denom.abs_() < 1e-10 || gamma_denom.abs_() < 1e-10 {
        return; // Degenerate triangle
    }

    let inv_alpha = 1.0 / alpha_denom;
    let inv_beta = 1.0 / beta_denom;
    let inv_gamma = 1.0 / gamma_denom;

    let w0 = v0.screen_space.w;
    let w1 = v1.screen_space.w;
    let w2 = v2.screen_space.w;

    let vs_output_size = c.vs_output.size as usize;
    let fragdepth_or_discard = c.fragdepth_or_discard;
    let depth_range_near = c.depth_range_near;
    let depth_range_far = c.depth_range_far;
    let front_facing = c.builtins.gl_FrontFacing;

    let program_idx = c.cur_program as usize;
    let interp = c.programs[program_idx].interpolation;

    // Get provoking vertex outputs for flat shading
    let provoke_out = if provoke < c.glverts.len() {
        c.glverts[provoke].vs_out.clone()
    } else {
        v0.vs_out.clone()
    };

    // Edge tie-breaking: top-left rule
    // An edge is a "top" edge if it's horizontal and goes left.
    // An edge is a "left" edge if it goes up (a > 0).
    let bias0 = if (l12.a > 0.0) || (l12.a == 0.0 && l12.b < 0.0) {
        0.0
    } else {
        -1e-5
    };
    let bias1 = if (l20.a > 0.0) || (l20.a == 0.0 && l20.b < 0.0) {
        0.0
    } else {
        -1e-5
    };
    let bias2 = if (l01.a > 0.0) || (l01.a == 0.0 && l01.b < 0.0) {
        0.0
    } else {
        -1e-5
    };

    for y in min_y..max_y {
        for x in min_x..max_x {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            // Compute barycentric coordinates
            let alpha = l12.func(px, py) * inv_alpha;
            let beta = l20.func(px, py) * inv_beta;
            let gamma = l01.func(px, py) * inv_gamma;

            // Check if inside triangle (with edge bias for tie-breaking)
            if alpha + bias0 < 0.0 || beta + bias1 < 0.0 || gamma + bias2 < 0.0 {
                continue;
            }

            // Interpolate z
            let z = alpha * hp0.z + beta * hp1.z + gamma * hp2.z + poly_offset;

            // Map z to depth range
            let z_mapped =
                depth_range_near + (depth_range_far - depth_range_near) * (z + 1.0) * 0.5;

            // Early depth test if shader doesn't use fragdepth/discard
            if !fragdepth_or_discard {
                if !fragment_processing(c, x, y, z_mapped) {
                    continue;
                }
            }

            // Interpolate vertex outputs for fragment shader
            setup_fs_input_triangle(
                c,
                alpha,
                beta,
                gamma,
                &v0.vs_out,
                &v1.vs_out,
                &v2.vs_out,
                w0,
                w1,
                w2,
                &provoke_out,
                &interp,
            );

            c.builtins.gl_FragCoord = Vec4::new(px, py, z_mapped, 1.0);
            c.builtins.gl_FragDepth = z_mapped;
            c.builtins.discard = false;
            c.builtins.gl_FrontFacing = front_facing;

            let program = &c.programs[program_idx];
            let fs = program.fragment_shader;
            let uniform = program.uniform;
            let mut fs_input_copy: Vec<f32> = c.fs_input[..vs_output_size].to_vec();

            unsafe {
                (fs)(
                    fs_input_copy.as_mut_ptr(),
                    &mut c.builtins as *mut ShaderBuiltins,
                    uniform,
                );
            }

            if c.builtins.discard {
                continue;
            }

            let cf = c.builtins.gl_FragColor;
            let final_z = if fragdepth_or_discard {
                c.builtins.gl_FragDepth
            } else {
                z_mapped
            };

            draw_pixel(c, cf, x, y, final_z, fragdepth_or_discard);
        }
    }
}

/// Draw triangle edges as lines (wireframe mode, GL_LINE polygon mode).
pub fn draw_triangle_line(
    c: &mut GlContext,
    v0: &GlVertex,
    v1: &GlVertex,
    v2: &GlVertex,
    provoke: usize,
) {
    let hp0 = v4_to_v3h(v0.screen_space);
    let hp1 = v4_to_v3h(v1.screen_space);
    let hp2 = v4_to_v3h(v2.screen_space);
    let w0 = v0.screen_space.w;
    let w1 = v1.screen_space.w;
    let w2 = v2.screen_space.w;

    let poly_offset = if c.poly_offset_line {
        calc_poly_offset(hp0, hp1, hp2, c.poly_factor, c.poly_units)
    } else {
        0.0
    };

    // Draw the three edges, choosing AA or thick line based on line_smooth
    if c.line_smooth {
        if v0.edge_flag != 0 {
            draw_aa_line(c, hp0, hp1, w0, w1, &v0.vs_out, &v1.vs_out, provoke, poly_offset);
        }
        if v1.edge_flag != 0 {
            draw_aa_line(c, hp1, hp2, w1, w2, &v1.vs_out, &v2.vs_out, provoke, poly_offset);
        }
        if v2.edge_flag != 0 {
            draw_aa_line(c, hp2, hp0, w2, w0, &v2.vs_out, &v0.vs_out, provoke, poly_offset);
        }
    } else {
        if v0.edge_flag != 0 {
            draw_thick_line(c, hp0, hp1, w0, w1, &v0.vs_out, &v1.vs_out, provoke, poly_offset);
        }
        if v1.edge_flag != 0 {
            draw_thick_line(c, hp1, hp2, w1, w2, &v1.vs_out, &v2.vs_out, provoke, poly_offset);
        }
        if v2.edge_flag != 0 {
            draw_thick_line(c, hp2, hp0, w2, w0, &v2.vs_out, &v0.vs_out, provoke, poly_offset);
        }
    }
}

/// Draw triangle vertices as points (GL_POINT polygon mode).
pub fn draw_triangle_point(
    c: &mut GlContext,
    v0: &GlVertex,
    v1: &GlVertex,
    v2: &GlVertex,
    _provoke: usize,
) {
    let hp0 = v4_to_v3h(v0.screen_space);
    let hp1 = v4_to_v3h(v1.screen_space);
    let hp2 = v4_to_v3h(v2.screen_space);

    let poly_offset = if c.poly_offset_pt {
        calc_poly_offset(hp0, hp1, hp2, c.poly_factor, c.poly_units)
    } else {
        0.0
    };

    draw_point(c, v0, poly_offset);
    draw_point(c, v1, poly_offset);
    draw_point(c, v2, poly_offset);
}

// ---------------------------------------------------------------------------
// Texture format conversion
// ---------------------------------------------------------------------------

/// Convert various pixel formats to packed RGBA bytes (4 bytes per pixel).
///
/// Reads from `input` with the given `pitch` (bytes per row) and writes to
/// `output` in RGBA byte order. Supports GL_RED, GL_RG, GL_RGB, GL_BGR,
/// GL_RGBA, GL_BGRA, GL_ALPHA, GL_LUMINANCE, GL_LUMINANCE_ALPHA, PGL_ONE_ALPHA.
pub fn convert_format_to_packed_rgba(
    output: &mut [u8],
    input: &[u8],
    w: i32,
    h: i32,
    pitch: i32,
    format: GLenum,
) {
    let wp = w as usize;
    let hp = h as usize;
    let pp = pitch as usize;

    for y in 0..hp {
        for x in 0..wp {
            let out_idx = (y * wp + x) * 4;
            match format {
                GL_RED => {
                    let in_idx = y * pp + x;
                    if in_idx < input.len() {
                        output[out_idx] = input[in_idx];
                        output[out_idx + 1] = 0;
                        output[out_idx + 2] = 0;
                        output[out_idx + 3] = 255;
                    }
                }
                GL_RG => {
                    let in_idx = y * pp + x * 2;
                    if in_idx + 1 < input.len() {
                        output[out_idx] = input[in_idx];
                        output[out_idx + 1] = input[in_idx + 1];
                        output[out_idx + 2] = 0;
                        output[out_idx + 3] = 255;
                    }
                }
                GL_RGB => {
                    let in_idx = y * pp + x * 3;
                    if in_idx + 2 < input.len() {
                        output[out_idx] = input[in_idx];
                        output[out_idx + 1] = input[in_idx + 1];
                        output[out_idx + 2] = input[in_idx + 2];
                        output[out_idx + 3] = 255;
                    }
                }
                GL_BGR => {
                    let in_idx = y * pp + x * 3;
                    if in_idx + 2 < input.len() {
                        output[out_idx] = input[in_idx + 2];
                        output[out_idx + 1] = input[in_idx + 1];
                        output[out_idx + 2] = input[in_idx];
                        output[out_idx + 3] = 255;
                    }
                }
                GL_RGBA => {
                    let in_idx = y * pp + x * 4;
                    if in_idx + 3 < input.len() {
                        output[out_idx] = input[in_idx];
                        output[out_idx + 1] = input[in_idx + 1];
                        output[out_idx + 2] = input[in_idx + 2];
                        output[out_idx + 3] = input[in_idx + 3];
                    }
                }
                GL_BGRA => {
                    let in_idx = y * pp + x * 4;
                    if in_idx + 3 < input.len() {
                        output[out_idx] = input[in_idx + 2];
                        output[out_idx + 1] = input[in_idx + 1];
                        output[out_idx + 2] = input[in_idx];
                        output[out_idx + 3] = input[in_idx + 3];
                    }
                }
                GL_ALPHA => {
                    let in_idx = y * pp + x;
                    if in_idx < input.len() {
                        output[out_idx] = 0;
                        output[out_idx + 1] = 0;
                        output[out_idx + 2] = 0;
                        output[out_idx + 3] = input[in_idx];
                    }
                }
                GL_LUMINANCE => {
                    let in_idx = y * pp + x;
                    if in_idx < input.len() {
                        let l = input[in_idx];
                        output[out_idx] = l;
                        output[out_idx + 1] = l;
                        output[out_idx + 2] = l;
                        output[out_idx + 3] = 255;
                    }
                }
                GL_LUMINANCE_ALPHA => {
                    let in_idx = y * pp + x * 2;
                    if in_idx + 1 < input.len() {
                        let l = input[in_idx];
                        output[out_idx] = l;
                        output[out_idx + 1] = l;
                        output[out_idx + 2] = l;
                        output[out_idx + 3] = input[in_idx + 1];
                    }
                }
                PGL_ONE_ALPHA => {
                    let in_idx = y * pp + x;
                    if in_idx < input.len() {
                        output[out_idx] = 255;
                        output[out_idx + 1] = 255;
                        output[out_idx + 2] = 255;
                        output[out_idx + 3] = input[in_idx];
                    }
                }
                _ => {
                    // Default: treat as RGBA
                    let in_idx = y * pp + x * 4;
                    if in_idx + 3 < input.len() {
                        output[out_idx] = input[in_idx];
                        output[out_idx + 1] = input[in_idx + 1];
                        output[out_idx + 2] = input[in_idx + 2];
                        output[out_idx + 3] = input[in_idx + 3];
                    }
                }
            }
        }
    }
}
