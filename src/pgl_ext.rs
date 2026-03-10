//! PortableGL extension functions -- PGL-specific functionality beyond standard OpenGL.
//!
//! These functions provide direct framebuffer access, format conversion utilities,
//! full-screen fragment shader execution, and simple drawing primitives that bypass
//! the GL pipeline.

#![allow(
    non_upper_case_globals,
    non_snake_case,
    dead_code,
    clippy::too_many_arguments
)]

use crate::gl_context::*;
use crate::gl_types::*;
use crate::math::*;

use std::ffi::c_void;

// ---------------------------------------------------------------------------
// Extension data types
// ---------------------------------------------------------------------------

/// A simple vertex with position and color, for use with PGL extension drawing.
#[derive(Clone, Copy, Debug, Default)]
pub struct PglVertex {
    /// Position (x, y, z, w).
    pub pos: Vec4,
    /// Color (r, g, b, a) as floats in [0, 1].
    pub color: Vec4,
}

impl PglVertex {
    /// Create a new PglVertex with the given position and color.
    #[inline]
    pub fn new(pos: Vec4, color: Vec4) -> Self {
        Self { pos, color }
    }
}

// ---------------------------------------------------------------------------
// Default pass-through vertex shader (used by pgl_create_frag_program)
// ---------------------------------------------------------------------------

/// Pass-through vertex shader that copies vertex_attribs[0] to gl_Position
/// and forwards all attributes as shader outputs.
pub fn pgl_default_vs(
    vs_output: &mut [f32],
    vertex_attribs: &[Vec4],
    builtins: &mut ShaderBuiltins,
    _uniforms: *mut c_void,
) {
    builtins.gl_Position = vertex_attribs[0];

    // Pass through all vertex attributes as float outputs (4 floats per attrib)
    let num_attribs = vertex_attribs.len().min(GL_MAX_VERTEX_ATTRIBS);
    let mut out_idx = 0;
    for i in 0..num_attribs {
        if out_idx + 3 < vs_output.len() {
            vs_output[out_idx] = vertex_attribs[i].x;
            vs_output[out_idx + 1] = vertex_attribs[i].y;
            vs_output[out_idx + 2] = vertex_attribs[i].z;
            vs_output[out_idx + 3] = vertex_attribs[i].w;
            out_idx += 4;
        }
    }
}

/// Simple pass-through fragment shader that reads color from fs_input[0..4].
pub fn pgl_default_fs(
    fs_input: &[f32],
    builtins: &mut ShaderBuiltins,
    _uniforms: *mut c_void,
) {
    if fs_input.len() >= 4 {
        builtins.gl_FragColor = Vec4::new(fs_input[0], fs_input[1], fs_input[2], fs_input[3]);
    } else {
        builtins.gl_FragColor = Vec4::new(1.0, 1.0, 1.0, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Format conversion utilities (free functions)
// ---------------------------------------------------------------------------

/// Converts pixel data from various GL formats to packed RGBA (4 bytes per pixel).
///
/// Supported formats: `GL_RED`, `GL_RG`, `GL_RGB`, `GL_RGBA`, `GL_BGR`, `GL_BGRA`,
/// `GL_LUMINANCE`, `GL_LUMINANCE_ALPHA`, `GL_ALPHA`, `PGL_ONE_ALPHA`.
///
/// Returns a `Vec<u8>` of size `w * h * 4` containing packed RGBA data.
/// Returns an empty Vec for unsupported formats.
pub fn convert_format_to_packed_rgba(data: &[u8], format: GLenum, w: i32, h: i32) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let total_pixels = w * h;
    let mut output = vec![0u8; total_pixels * 4];

    match format {
        PGL_ONE_ALPHA => {
            // 1 byte per pixel: r=255, g=255, b=255, a=input
            for i in 0..total_pixels {
                if i >= data.len() {
                    break;
                }
                let dst = i * 4;
                output[dst] = 255;
                output[dst + 1] = 255;
                output[dst + 2] = 255;
                output[dst + 3] = data[i];
            }
            output
        }
        GL_ALPHA => {
            // 1 byte per pixel: r=0, g=0, b=0, a=input
            for i in 0..total_pixels {
                if i >= data.len() {
                    break;
                }
                let dst = i * 4;
                output[dst] = 0;
                output[dst + 1] = 0;
                output[dst + 2] = 0;
                output[dst + 3] = data[i];
            }
            output
        }
        GL_LUMINANCE => {
            // 1 byte per pixel: r=g=b=lum, a=255
            for i in 0..total_pixels {
                if i >= data.len() {
                    break;
                }
                let dst = i * 4;
                let lum = data[i];
                output[dst] = lum;
                output[dst + 1] = lum;
                output[dst + 2] = lum;
                output[dst + 3] = 255;
            }
            output
        }
        GL_RED => {
            // 1 byte per pixel: r=input, g=0, b=0, a=255
            for i in 0..total_pixels {
                if i >= data.len() {
                    break;
                }
                let dst = i * 4;
                output[dst] = data[i];
                output[dst + 1] = 0;
                output[dst + 2] = 0;
                output[dst + 3] = 255;
            }
            output
        }
        GL_LUMINANCE_ALPHA => {
            // 2 bytes per pixel: r=g=b=lum, a=alpha
            for i in 0..total_pixels {
                let src = i * 2;
                if src + 1 >= data.len() {
                    break;
                }
                let dst = i * 4;
                let lum = data[src];
                let alpha = data[src + 1];
                output[dst] = lum;
                output[dst + 1] = lum;
                output[dst + 2] = lum;
                output[dst + 3] = alpha;
            }
            output
        }
        GL_RG => {
            // 2 bytes per pixel: r=first, g=second, b=0, a=255
            for i in 0..total_pixels {
                let src = i * 2;
                if src + 1 >= data.len() {
                    break;
                }
                let dst = i * 4;
                output[dst] = data[src];
                output[dst + 1] = data[src + 1];
                output[dst + 2] = 0;
                output[dst + 3] = 255;
            }
            output
        }
        GL_RGB => {
            // 3 bytes per pixel: r,g,b from input, a=255
            for i in 0..total_pixels {
                let src = i * 3;
                if src + 2 >= data.len() {
                    break;
                }
                let dst = i * 4;
                output[dst] = data[src];
                output[dst + 1] = data[src + 1];
                output[dst + 2] = data[src + 2];
                output[dst + 3] = 255;
            }
            output
        }
        GL_BGR => {
            // 3 bytes per pixel: b,g,r from input (swapped), a=255
            for i in 0..total_pixels {
                let src = i * 3;
                if src + 2 >= data.len() {
                    break;
                }
                let dst = i * 4;
                output[dst] = data[src + 2]; // r from third byte
                output[dst + 1] = data[src + 1]; // g stays
                output[dst + 2] = data[src]; // b from first byte
                output[dst + 3] = 255;
            }
            output
        }
        GL_BGRA => {
            // 4 bytes per pixel: b,g,r,a from input (r/b swapped)
            for i in 0..total_pixels {
                let src = i * 4;
                if src + 3 >= data.len() {
                    break;
                }
                let dst = i * 4;
                output[dst] = data[src + 2]; // r from third byte
                output[dst + 1] = data[src + 1]; // g stays
                output[dst + 2] = data[src]; // b from first byte
                output[dst + 3] = data[src + 3]; // a stays
            }
            output
        }
        GL_RGBA => {
            // 4 bytes per pixel: direct copy
            let total_bytes = total_pixels * 4;
            let copy_len = total_bytes.min(data.len());
            output[..copy_len].copy_from_slice(&data[..copy_len]);
            output
        }
        _ => {
            // Unsupported format
            Vec::new()
        }
    }
}

/// Converts a single-channel grayscale image to RGBA.
///
/// Each grayscale byte becomes (v, v, v, 255) in the output.
pub fn convert_grayscale_to_rgba(data: &[u8], w: i32, h: i32) -> Vec<u8> {
    let total_pixels = (w as usize) * (h as usize);
    let count = total_pixels.min(data.len());
    let mut output = Vec::with_capacity(count * 4);

    for i in 0..count {
        let v = data[i];
        output.push(v);
        output.push(v);
        output.push(v);
        output.push(255);
    }

    output
}

/// Converts a grayscale image to RGBA using two colors for interpolation.
///
/// Each input byte (0-255) is used as the interpolation factor `t` between
/// `bg_rgba` (background, at t=0) and `text_rgba` (text/foreground, at t=255).
///
/// The RGBA u32 values are packed as 0xRRGGBBAA.
pub fn convert_grayscale_to_rgba_blend(
    input: &[u8],
    bg_rgba: u32,
    text_rgba: u32,
) -> Vec<u8> {
    let bg_r = ((bg_rgba >> 24) & 0xFF) as f32;
    let bg_g = ((bg_rgba >> 16) & 0xFF) as f32;
    let bg_b = ((bg_rgba >> 8) & 0xFF) as f32;
    let bg_a = (bg_rgba & 0xFF) as f32;

    let text_r = ((text_rgba >> 24) & 0xFF) as f32;
    let text_g = ((text_rgba >> 16) & 0xFF) as f32;
    let text_b = ((text_rgba >> 8) & 0xFF) as f32;
    let text_a = (text_rgba & 0xFF) as f32;

    let mut output = Vec::with_capacity(input.len() * 4);

    for &byte in input {
        let t = byte as f32 / 255.0;
        let r = (bg_r + (text_r - bg_r) * t + 0.5) as u8;
        let g = (bg_g + (text_g - bg_g) * t + 0.5) as u8;
        let b = (bg_b + (text_b - bg_b) * t + 0.5) as u8;
        let a = (bg_a + (text_a - bg_a) * t + 0.5) as u8;
        output.push(r);
        output.push(g);
        output.push(b);
        output.push(a);
    }

    output
}

// ---------------------------------------------------------------------------
// GlContext extension methods
// ---------------------------------------------------------------------------

impl GlContext {
    // -----------------------------------------------------------------------
    // Screen clearing
    // -----------------------------------------------------------------------

    /// Clears the entire back buffer to black (all bytes set to zero).
    pub fn pgl_clear_screen(&mut self) {
        for byte in self.back_buffer.buf.iter_mut() {
            *byte = 0;
        }
    }

    // -----------------------------------------------------------------------
    // Interpolation control
    // -----------------------------------------------------------------------

    /// Sets the interpolation modes for the current program's vertex shader outputs.
    ///
    /// `num_attribs` is the number of float components in the vertex shader output.
    /// `interps` contains the interpolation mode for each component (PGL_SMOOTH,
    /// PGL_FLAT, or PGL_NOPERSPECTIVE).
    pub fn pgl_set_interp(&mut self, num_attribs: GLsizei, interps: &[GLenum]) {
        let prog_idx = self.cur_program as usize;
        if prog_idx >= self.programs.len() {
            return;
        }

        self.programs[prog_idx].vs_output_size = num_attribs;

        let count = (num_attribs as usize)
            .min(GL_MAX_VERTEX_OUTPUT_COMPONENTS)
            .min(interps.len());
        for i in 0..count {
            self.programs[prog_idx].interpolation[i] = interps[i];
        }

        self.vs_output.size = num_attribs;
        self.vs_output.interpolation = self.programs[prog_idx].interpolation.as_ptr();
    }

    // -----------------------------------------------------------------------
    // Fragment-only program creation
    // -----------------------------------------------------------------------

    /// Creates a program using the default pass-through vertex shader with the
    /// given fragment shader and interpolation modes.
    ///
    /// `fs` is the fragment shader function.
    /// `n` is the number of interpolated output components.
    /// `interp` contains the interpolation mode for each component.
    /// `fragdepth_or_discard` indicates whether the fragment shader writes
    /// `gl_FragDepth` or uses `discard`.
    ///
    /// Returns the program ID (1-based index into the programs array).
    pub fn pgl_create_frag_program(
        &mut self,
        fs: FragFunc,
        n: GLsizei,
        interp: &[GLenum],
        fragdepth_or_discard: bool,
    ) -> GLuint {
        let mut interpolation = [PGL_SMOOTH; GL_MAX_VERTEX_OUTPUT_COMPONENTS];
        let count = (n as usize).min(GL_MAX_VERTEX_OUTPUT_COMPONENTS).min(interp.len());
        for i in 0..count {
            interpolation[i] = interp[i];
        }

        let program = GlProgram {
            vertex_shader: pgl_default_vs,
            fragment_shader: fs,
            uniform: std::ptr::null_mut(),
            vs_output_size: n,
            interpolation,
            fragdepth_or_discard,
            deleted: false,
        };

        // Find a deleted slot or push a new one (slot 0 is reserved)
        let mut id = 0u32;
        for (i, p) in self.programs.iter().enumerate() {
            if i == 0 {
                continue;
            }
            if p.deleted {
                id = i as GLuint;
                break;
            }
        }

        if id == 0 {
            self.programs.push(program);
            id = (self.programs.len() - 1) as GLuint;
        } else {
            self.programs[id as usize] = program;
        }

        id
    }

    // -----------------------------------------------------------------------
    // Frame drawing helpers
    // -----------------------------------------------------------------------

    /// Runs the current program's fragment shader for every pixel in the back buffer.
    ///
    /// For each pixel (x, y): sets `gl_FragCoord` to `(x + 0.5, y + 0.5, 0, 1)`,
    /// calls the fragment shader, and writes the resulting color to the back buffer.
    /// No depth or stencil testing is performed.
    ///
    /// In a hardware GL implementation this would present the back buffer to the
    /// screen. In a software renderer like PGL it runs the fragment shader across
    /// every pixel -- useful for full-screen post-processing effects.
    pub fn pgl_draw_frame(&mut self) {
        let prog_idx = self.cur_program as usize;
        if prog_idx >= self.programs.len() {
            return;
        }

        let frag_shader = self.programs[prog_idx].fragment_shader;
        let uniforms = self.programs[prog_idx].uniform;
        let w = self.back_buffer.w;
        let h = self.back_buffer.h;

        for y in 0..h {
            for x in 0..w {
                self.builtins.gl_FragCoord = Vec4::new(
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                    0.0,
                    1.0,
                );
                self.builtins.discard = false;

                (frag_shader)(&self.fs_input, &mut self.builtins, uniforms);

                if !self.builtins.discard {
                    let color = Color::from_vec4(self.builtins.gl_FragColor);
                    let idx = ((y * w + x) * 4) as usize;
                    if idx + 3 < self.back_buffer.buf.len() {
                        self.back_buffer.buf[idx] = color.r;
                        self.back_buffer.buf[idx + 1] = color.g;
                        self.back_buffer.buf[idx + 2] = color.b;
                        self.back_buffer.buf[idx + 3] = color.a;
                    }
                }
            }
        }
    }

    /// Same as `pgl_draw_frame` but uses the provided shader and uniforms
    /// instead of the current program.
    pub fn pgl_draw_frame2(&mut self, frag_shader: FragFunc, uniforms: *mut c_void) {
        let w = self.back_buffer.w;
        let h = self.back_buffer.h;

        for y in 0..h {
            for x in 0..w {
                self.builtins.gl_FragCoord = Vec4::new(
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                    0.0,
                    1.0,
                );
                self.builtins.discard = false;

                (frag_shader)(&self.fs_input, &mut self.builtins, uniforms);

                if !self.builtins.discard {
                    let color = Color::from_vec4(self.builtins.gl_FragColor);
                    let idx = ((y * w + x) * 4) as usize;
                    if idx + 3 < self.back_buffer.buf.len() {
                        self.back_buffer.buf[idx] = color.r;
                        self.back_buffer.buf[idx + 1] = color.g;
                        self.back_buffer.buf[idx + 2] = color.b;
                        self.back_buffer.buf[idx + 3] = color.a;
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Zero-copy buffer management
    // -----------------------------------------------------------------------

    /// Sets buffer data without copying, similar to the C version where the user
    /// retains ownership of the data.
    ///
    /// `target` is the buffer binding target (e.g. `GL_ARRAY_BUFFER`).
    /// `size` is the size of the data in bytes.
    /// `data` is a raw pointer to the user's data.
    /// `own` indicates whether the GL context takes ownership of the data.
    ///
    /// # Safety
    /// The caller must ensure `data` points to valid memory of at least `size` bytes
    /// and remains valid for the lifetime of the buffer (unless `own` is true).
    pub fn pgl_buffer_data(
        &mut self,
        target: GLenum,
        size: GLsizeiptr,
        data: *mut u8,
        own: bool,
    ) {
        let buf_type = target as usize;
        if buf_type < GL_ARRAY_BUFFER as usize || buf_type >= GL_NUM_BUFFER_TYPES as usize {
            self.error = GL_INVALID_ENUM;
            return;
        }

        let idx = buf_type - GL_ARRAY_BUFFER as usize;
        if idx >= self.bound_buffers.len() {
            self.error = GL_INVALID_OPERATION;
            return;
        }

        let buf_id = self.bound_buffers[idx] as usize;
        if buf_id == 0 || buf_id >= self.buffers.len() {
            self.error = GL_INVALID_OPERATION;
            return;
        }

        // In the Rust port, we copy the data into a Vec for safety,
        // but mark user_owned so the caller knows the semantics.
        if !data.is_null() && size > 0 {
            let slice = unsafe { std::slice::from_raw_parts(data, size as usize) };
            self.buffers[buf_id].data = slice.to_vec();
        } else {
            self.buffers[buf_id].data = Vec::new();
        }
        self.buffers[buf_id].size = size as GLsizei;
        self.buffers[buf_id].user_owned = !own;
    }

    /// Sets buffer data from a Rust slice (safe version of `pgl_buffer_data`).
    pub fn pgl_buffer_data_slice(&mut self, target: GLenum, data: &[u8], usage: GLenum) {
        let buf_type = target as usize;
        if buf_type < GL_ARRAY_BUFFER as usize || buf_type >= GL_NUM_BUFFER_TYPES as usize {
            self.error = GL_INVALID_ENUM;
            return;
        }

        let idx = buf_type - GL_ARRAY_BUFFER as usize;
        if idx >= self.bound_buffers.len() {
            self.error = GL_INVALID_OPERATION;
            return;
        }

        let buf_id = self.bound_buffers[idx] as usize;
        if buf_id == 0 || buf_id >= self.buffers.len() {
            self.error = GL_INVALID_OPERATION;
            return;
        }

        self.buffers[buf_id].data = data.to_vec();
        self.buffers[buf_id].size = data.len() as GLsizei;
        self.buffers[buf_id].type_ = usage;
        self.buffers[buf_id].user_owned = true;
    }

    // -----------------------------------------------------------------------
    // Zero-copy texture data
    // -----------------------------------------------------------------------

    /// Creates a 1D texture from user-provided data without copying (in the C sense).
    ///
    /// In the Rust port the data is copied for safety, but `user_owned` is set.
    ///
    /// # Safety
    /// When using raw pointers, the caller must ensure `data` is valid.
    pub fn pgl_tex_image_1d(
        &mut self,
        target: GLenum,
        _level: GLint,
        _internalformat: GLint,
        width: GLsizei,
        _border: GLint,
        format: GLenum,
        _type: GLenum,
        data: *mut u8,
    ) {
        let _ = format;

        let tex_idx = self
            .bound_textures
            .get((target as usize).wrapping_sub(GL_TEXTURE_UNBOUND as usize + 1))
            .copied()
            .unwrap_or(0) as usize;

        if tex_idx == 0 || tex_idx >= self.textures.len() {
            self.error = GL_INVALID_OPERATION;
            return;
        }

        let expected_size = (width as usize) * 4;

        self.textures[tex_idx].w = width;
        self.textures[tex_idx].h = 1;
        self.textures[tex_idx].d = 1;
        self.textures[tex_idx].type_ = target;
        self.textures[tex_idx].format = GL_RGBA;
        self.textures[tex_idx].user_owned = true;

        if !data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(data, expected_size) };
            self.textures[tex_idx].data = slice.to_vec();
        } else {
            self.textures[tex_idx].data = vec![0u8; expected_size];
        }
    }

    /// Creates a 2D texture from user-provided data without copying (in the C sense).
    ///
    /// In the Rust port the data is copied for safety, but `user_owned` is set.
    ///
    /// # Safety
    /// When using raw pointers, the caller must ensure `data` is valid.
    pub fn pgl_tex_image_2d(
        &mut self,
        target: GLenum,
        _level: GLint,
        _internalformat: GLint,
        width: GLsizei,
        height: GLsizei,
        _border: GLint,
        format: GLenum,
        _type: GLenum,
        data: *mut u8,
    ) {
        let _ = format;

        let tex_idx = self
            .bound_textures
            .get((target as usize).wrapping_sub(GL_TEXTURE_UNBOUND as usize + 1))
            .copied()
            .unwrap_or(0) as usize;

        if tex_idx == 0 || tex_idx >= self.textures.len() {
            self.error = GL_INVALID_OPERATION;
            return;
        }

        let expected_size = (width as usize) * (height as usize) * 4;

        self.textures[tex_idx].w = width;
        self.textures[tex_idx].h = height;
        self.textures[tex_idx].d = 1;
        self.textures[tex_idx].type_ = target;
        self.textures[tex_idx].format = GL_RGBA;
        self.textures[tex_idx].user_owned = true;

        if !data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(data, expected_size) };
            self.textures[tex_idx].data = slice.to_vec();
        } else {
            self.textures[tex_idx].data = vec![0u8; expected_size];
        }
    }

    /// Creates a 3D texture from user-provided data without copying (in the C sense).
    ///
    /// In the Rust port the data is copied for safety, but `user_owned` is set.
    ///
    /// # Safety
    /// When using raw pointers, the caller must ensure `data` is valid.
    pub fn pgl_tex_image_3d(
        &mut self,
        target: GLenum,
        _level: GLint,
        _internalformat: GLint,
        width: GLsizei,
        height: GLsizei,
        depth: GLsizei,
        _border: GLint,
        format: GLenum,
        _type: GLenum,
        data: *mut u8,
    ) {
        let _ = format;

        let tex_idx = self
            .bound_textures
            .get((target as usize).wrapping_sub(GL_TEXTURE_UNBOUND as usize + 1))
            .copied()
            .unwrap_or(0) as usize;

        if tex_idx == 0 || tex_idx >= self.textures.len() {
            self.error = GL_INVALID_OPERATION;
            return;
        }

        let expected_size = (width as usize) * (height as usize) * (depth as usize) * 4;

        self.textures[tex_idx].w = width;
        self.textures[tex_idx].h = height;
        self.textures[tex_idx].d = depth;
        self.textures[tex_idx].type_ = target;
        self.textures[tex_idx].format = GL_RGBA;
        self.textures[tex_idx].user_owned = true;

        if !data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(data, expected_size) };
            self.textures[tex_idx].data = slice.to_vec();
        } else {
            self.textures[tex_idx].data = vec![0u8; expected_size];
        }
    }

    // -----------------------------------------------------------------------
    // Data access
    // -----------------------------------------------------------------------

    /// Returns a raw pointer to the buffer's data.
    ///
    /// Returns null if the buffer ID is invalid or deleted.
    /// This mirrors the C API which returns `u8*`.
    pub fn pgl_get_buffer_data(&self, buffer: GLuint) -> *mut u8 {
        let idx = buffer as usize;
        if idx == 0 || idx >= self.buffers.len() {
            return std::ptr::null_mut();
        }
        if self.buffers[idx].deleted {
            return std::ptr::null_mut();
        }
        self.buffers[idx].data.as_ptr() as *mut u8
    }

    /// Returns a safe reference to the buffer's data, or `None` if invalid.
    pub fn pgl_get_buffer_data_ref(&self, buffer: GLuint) -> Option<&[u8]> {
        let idx = buffer as usize;
        if idx == 0 || idx >= self.buffers.len() {
            return None;
        }
        if self.buffers[idx].deleted {
            return None;
        }
        Some(&self.buffers[idx].data)
    }

    /// Returns a raw pointer to the texture's data.
    ///
    /// Returns null if the texture ID is invalid or deleted.
    /// This mirrors the C API which returns `u8*`.
    pub fn pgl_get_texture_data(&self, texture: GLuint) -> *mut u8 {
        let idx = texture as usize;
        if idx == 0 || idx >= self.textures.len() {
            return std::ptr::null_mut();
        }
        if self.textures[idx].deleted {
            return std::ptr::null_mut();
        }
        self.textures[idx].data.as_ptr() as *mut u8
    }

    /// Returns a safe reference to the texture's data, or `None` if invalid.
    pub fn pgl_get_texture_data_ref(&self, texture: GLuint) -> Option<&[u8]> {
        let idx = texture as usize;
        if idx == 0 || idx >= self.textures.len() {
            return None;
        }
        if self.textures[idx].deleted {
            return None;
        }
        Some(&self.textures[idx].data)
    }

    // -----------------------------------------------------------------------
    // Back buffer control
    // -----------------------------------------------------------------------

    /// Returns a reference to the back buffer framebuffer structure.
    pub fn pgl_get_back_buffer(&self) -> &GlFramebuffer {
        &self.back_buffer
    }

    /// Returns a mutable reference to the back buffer framebuffer structure.
    pub fn pgl_get_back_buffer_mut(&mut self) -> &mut GlFramebuffer {
        &mut self.back_buffer
    }

    /// Sets an external back buffer.
    ///
    /// `data` is a raw pointer to the pixel data (RGBA, 4 bytes per pixel).
    /// `w` and `h` are the dimensions in pixels.
    /// `own` indicates whether the GL context takes ownership.
    ///
    /// # Safety
    /// The caller must ensure `data` points to valid memory of at least `w * h * 4` bytes.
    pub fn pgl_set_back_buffer(&mut self, data: *mut u8, w: i32, h: i32, own: bool) {
        let size = (w as usize) * (h as usize) * 4;

        if !data.is_null() && size > 0 {
            let slice = unsafe { std::slice::from_raw_parts(data, size) };
            self.back_buffer.buf = slice.to_vec();
        } else {
            self.back_buffer.buf = vec![0u8; size];
        }

        self.back_buffer.w = w;
        self.back_buffer.h = h;
        self.user_alloced_backbuf = !own;
    }

    /// Sets the back buffer from a Rust Vec (safe version).
    pub fn pgl_set_back_buffer_vec(
        &mut self,
        width: GLsizei,
        height: GLsizei,
        buf: Vec<u8>,
    ) {
        self.back_buffer.w = width;
        self.back_buffer.h = height;
        self.back_buffer.buf = buf;
        self.user_alloced_backbuf = true;
    }

    // -----------------------------------------------------------------------
    // Simple drawing utilities (direct framebuffer, bypass GL pipeline)
    // -----------------------------------------------------------------------

    /// Draws a single pixel directly to the back buffer at (x, y).
    pub fn put_pixel(&mut self, color: Color, x: i32, y: i32) {
        let w = self.back_buffer.w;
        let h = self.back_buffer.h;

        if x < 0 || y < 0 || x >= w || y >= h {
            return;
        }

        let idx = ((y * w + x) * 4) as usize;
        if idx + 3 < self.back_buffer.buf.len() {
            self.back_buffer.buf[idx] = color.r;
            self.back_buffer.buf[idx + 1] = color.g;
            self.back_buffer.buf[idx + 2] = color.b;
            self.back_buffer.buf[idx + 3] = color.a;
        }
    }

    /// Draws a 1-pixel wide line using Bresenham's algorithm directly to the back buffer.
    pub fn put_line(&mut self, color: Color, x1: f32, y1: f32, x2: f32, y2: f32) {
        let mut x1 = x1 as i32;
        let mut y1 = y1 as i32;
        let x2 = x2 as i32;
        let y2 = y2 as i32;

        let dx = (x2 - x1).abs();
        let dy = -(y2 - y1).abs();
        let sx: i32 = if x1 < x2 { 1 } else { -1 };
        let sy: i32 = if y1 < y2 { 1 } else { -1 };
        let mut err = dx + dy;

        loop {
            self.put_pixel(color, x1, y1);

            if x1 == x2 && y1 == y2 {
                break;
            }

            let e2 = 2 * err;

            if e2 >= dy {
                if x1 == x2 {
                    break;
                }
                err += dy;
                x1 += sx;
            }

            if e2 <= dx {
                if y1 == y2 {
                    break;
                }
                err += dx;
                y1 += sy;
            }
        }
    }

    /// Draws a filled triangle with per-vertex colors using barycentric interpolation,
    /// directly to the back buffer.
    pub fn put_triangle(
        &mut self,
        c1: Color,
        c2: Color,
        c3: Color,
        p1: Vec2,
        p2: Vec2,
        p3: Vec2,
    ) {
        // Compute bounding box
        let min_x = p1.x.min(p2.x).min(p3.x).floor() as i32;
        let min_y = p1.y.min(p2.y).min(p3.y).floor() as i32;
        let max_x = p1.x.max(p2.x).max(p3.x).ceil() as i32;
        let max_y = p1.y.max(p2.y).max(p3.y).ceil() as i32;

        let w = self.back_buffer.w;
        let h = self.back_buffer.h;

        // Clamp to screen bounds
        let min_x = min_x.max(0);
        let min_y = min_y.max(0);
        let max_x = max_x.min(w - 1);
        let max_y = max_y.min(h - 1);

        // Triangle area * 2 (using cross product)
        let area = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y);

        if area.abs() < f32::EPSILON {
            return; // degenerate triangle
        }

        let inv_area = 1.0 / area;

        // Convert colors to floats for interpolation
        let c1v = c1.to_vec4();
        let c2v = c2.to_vec4();
        let c3v = c3.to_vec4();

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;

                // Barycentric coordinates
                let w1 =
                    ((p2.x - px) * (p3.y - py) - (p3.x - px) * (p2.y - py)) * inv_area;
                let w2 =
                    ((p3.x - px) * (p1.y - py) - (p1.x - px) * (p3.y - py)) * inv_area;
                let w3 = 1.0 - w1 - w2;

                // Check if point is inside triangle
                if w1 >= 0.0 && w2 >= 0.0 && w3 >= 0.0 {
                    // Interpolate color
                    let r = clamp_01(c1v.x * w1 + c2v.x * w2 + c3v.x * w3);
                    let g = clamp_01(c1v.y * w1 + c2v.y * w2 + c3v.y * w3);
                    let b = clamp_01(c1v.z * w1 + c2v.z * w2 + c3v.z * w3);
                    let a = clamp_01(c1v.w * w1 + c2v.w * w2 + c3v.w * w3);

                    let color = Color::from_vec4(Vec4::new(r, g, b, a));
                    self.put_pixel(color, x, y);
                }
            }
        }
    }
}
