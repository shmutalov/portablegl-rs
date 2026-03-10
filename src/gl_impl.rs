//! OpenGL API implementation module for the PortableGL Rust port.
//!
//! All public OpenGL-like functions are implemented as methods on `GlContext`.

#![allow(non_snake_case, non_upper_case_globals, unused_variables, dead_code, clippy::too_many_arguments)]

use crate::gl_context::*;
use crate::gl_internal;
use crate::gl_types::*;
use crate::math::*;
use core::ffi::c_void;

#[cfg(feature = "no_std")]
use alloc::vec;
#[cfg(feature = "no_std")]
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Helper: set error and return
// ---------------------------------------------------------------------------

macro_rules! set_err {
    ($ctx:expr, $err:expr) => {{
        if $ctx.error == GL_NO_ERROR {
            $ctx.error = $err;
        }
    }};
}

// ---------------------------------------------------------------------------
// Free functions (default shaders, helpers)
// ---------------------------------------------------------------------------

fn default_vs(
    _vs_output: &mut [f32],
    vertex_attribs: &[Vec4],
    builtins: &mut ShaderBuiltins,
    _uniforms: *mut c_void,
) {
    builtins.gl_Position = vertex_attribs[0];
}

fn default_fs(
    _fs_input: &[f32],
    builtins: &mut ShaderBuiltins,
    _uniforms: *mut c_void,
) {
    builtins.gl_FragColor = Vec4::new(1.0, 0.0, 0.0, 1.0);
}

fn init_tex(tex: &mut GlTexture, target: GLenum) {
    tex.w = 0;
    tex.h = 0;
    tex.d = 0;
    tex.mag_filter = GL_NEAREST;
    tex.min_filter = GL_NEAREST;
    tex.wrap_s = GL_REPEAT;
    tex.wrap_t = GL_REPEAT;
    tex.wrap_r = GL_REPEAT;
    tex.format = GL_RGBA;
    tex.type_ = target;
    tex.deleted = false;
    tex.user_owned = false;
    tex.data = Vec::new();
}

fn format_components(format: GLenum) -> Option<i32> {
    match format {
        GL_RED | GL_ALPHA | GL_LUMINANCE | PGL_ONE_ALPHA => Some(1),
        GL_RG | GL_LUMINANCE_ALPHA => Some(2),
        GL_RGB | GL_BGR => Some(3),
        GL_RGBA | GL_BGRA => Some(4),
        _ => None,
    }
}

/// Pack an RGBA color (as floats in [0,1]) into a u32 pixel in ABGR byte order.
#[inline]
fn color_to_u32(r: f32, g: f32, b: f32, a: f32) -> u32 {
    let rb = (clamp_01(r) * 255.0 + 0.5) as u32;
    let gb = (clamp_01(g) * 255.0 + 0.5) as u32;
    let bb = (clamp_01(b) * 255.0 + 0.5) as u32;
    let ab = (clamp_01(a) * 255.0 + 0.5) as u32;
    rb | (gb << 8) | (bb << 16) | (ab << 24)
}

/// Read a u32 from a byte slice at the given pixel index.
#[inline]
fn read_pixel(buf: &[u8], idx: usize) -> u32 {
    let off = idx * 4;
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Write a u32 to a byte slice at the given pixel index.
#[inline]
fn write_pixel(buf: &mut [u8], idx: usize, val: u32) {
    let off = idx * 4;
    let bytes = val.to_le_bytes();
    buf[off] = bytes[0];
    buf[off + 1] = bytes[1];
    buf[off + 2] = bytes[2];
    buf[off + 3] = bytes[3];
}

/// Map a buffer target GLenum to an index into bound_buffers.
#[inline]
fn buffer_target_index(target: GLenum) -> Option<usize> {
    if target >= GL_ARRAY_BUFFER && target < GL_NUM_BUFFER_TYPES {
        Some((target - GL_ARRAY_BUFFER) as usize)
    } else {
        None
    }
}

/// Map a texture target GLenum to an index into bound_textures/default_textures.
#[inline]
fn texture_target_index(target: GLenum) -> Option<usize> {
    if target >= GL_TEXTURE_1D && target < GL_NUM_TEXTURE_TYPES {
        Some((target - GL_TEXTURE_1D) as usize)
    } else {
        None
    }
}

/// Number of texture target types.
const NUM_TEX_TARGETS: usize = (GL_NUM_TEXTURE_TYPES - GL_TEXTURE_1D) as usize;

/// Number of buffer binding targets.
const NUM_BUF_TARGETS: usize = (GL_NUM_BUFFER_TYPES - GL_ARRAY_BUFFER) as usize;

// ---------------------------------------------------------------------------
// GlContext implementation
// ---------------------------------------------------------------------------

impl GlContext {
    /// Initialize the GL context with a given width and height.
    /// Returns a Vec<u32> that serves as the back buffer (pixel data).
    pub fn init(&mut self, w: GLsizei, h: GLsizei) -> Vec<u32> {
        // Reset to default state
        *self = GlContext::default();

        let total = (w * h) as usize;

        // State defaults matching OpenGL spec
        self.provoking_vert = GL_LAST_VERTEX_CONVENTION;
        self.cull_mode = GL_BACK;
        self.front_face = GL_CCW;
        self.depth_func = GL_LESS;
        self.blend_srgb = GL_ONE;
        self.blend_sa = GL_ONE;
        self.blend_drgb = GL_ZERO;
        self.blend_da = GL_ZERO;
        self.blend_eq_rgb = GL_FUNC_ADD;
        self.blend_eq_a = GL_FUNC_ADD;
        self.poly_mode_front = GL_FILL;
        self.poly_mode_back = GL_FILL;
        self.point_spr_origin = GL_UPPER_LEFT;
        self.logic_func = GL_COPY;
        self.point_size = 1.0;
        self.line_width = 1.0;
        self.clear_depth = 1.0;
        self.depth_range_near = 0.0;
        self.depth_range_far = 1.0;
        self.unpack_alignment = 4;
        self.pack_alignment = 4;
        self.color_mask = !0u32;
        self.depth_mask = true;

        // Stencil defaults
        self.stencil_writemask = !0u32;
        self.stencil_writemask_back = !0u32;
        self.stencil_valuemask = !0u32;
        self.stencil_valuemask_back = !0u32;
        self.stencil_func = GL_ALWAYS;
        self.stencil_func_back = GL_ALWAYS;
        self.stencil_sfail = GL_KEEP;
        self.stencil_dpfail = GL_KEEP;
        self.stencil_dppass = GL_KEEP;
        self.stencil_sfail_back = GL_KEEP;
        self.stencil_dpfail_back = GL_KEEP;
        self.stencil_dppass_back = GL_KEEP;

        // Triangle draw mode
        self.draw_triangle_front = TRIANGLE_FILL;
        self.draw_triangle_back = TRIANGLE_FILL;

        // Viewport dimensions
        self.width = w;
        self.height = h;
        self.xmin = 0;
        self.ymin = 0;
        self.lx = 0;
        self.ly = 0;
        self.ux = w;
        self.uy = h;

        // Scissor defaults to full viewport
        self.scissor_lx = 0;
        self.scissor_ly = 0;
        self.scissor_w = w;
        self.scissor_h = h;

        // Allocate vs_output buffer
        self.vs_output.output_buf = vec![0.0; PGL_MAX_VERTICES * GL_MAX_VERTEX_OUTPUT_COMPONENTS];

        // Create default program 0
        let mut prog = GlProgram::default();
        prog.vertex_shader = default_vs;
        prog.fragment_shader = default_fs;
        prog.vs_output_size = 0;
        self.programs.push(prog);
        self.cur_program = 0;

        // Create default VAO at index 0
        self.vertex_arrays.push(GlVertexArray::default());
        self.cur_vertex_array = 0;

        // Create invalid buffer at index 0
        let mut buf0 = GlBuffer::default();
        buf0.user_owned = true;
        self.buffers.push(buf0);

        // Bound buffers (one per buffer target type)
        self.bound_buffers = vec![0u32; NUM_BUF_TARGETS];

        // Create default texture at index 0
        let mut tex0 = GlTexture::default();
        tex0.type_ = GL_TEXTURE_UNBOUND;
        self.textures.push(tex0);

        // Initialize default_textures array (one per texture target type)
        self.default_textures = Vec::with_capacity(NUM_TEX_TARGETS);
        for i in 0..NUM_TEX_TARGETS {
            let mut t = GlTexture::default();
            t.type_ = GL_TEXTURE_UNBOUND;
            self.default_textures.push(t);
        }

        // Bound textures (one per texture target type)
        self.bound_textures = vec![0u32; NUM_TEX_TARGETS];

        // Allocate depth buffer (w*h u32 values stored as bytes)
        self.zbuf = GlFramebuffer {
            buf: vec![0u8; total * 4],
            w,
            h,
        };

        // Allocate stencil buffer (w*h u32 values stored as bytes)
        self.stencil_buf = GlFramebuffer {
            buf: vec![0u8; total * 4],
            w,
            h,
        };

        // Allocate back buffer (w*h pixels as bytes)
        self.back_buffer = GlFramebuffer {
            buf: vec![0u8; total * 4],
            w,
            h,
        };

        // Set viewport matrix
        self.vp_mat = make_viewport_matrix(0, 0, w, h, 1);

        // Allocate processed vertices
        self.glverts = Vec::new();

        // Return back buffer as Vec<u32> for the caller
        let pixels = vec![0u32; total];
        pixels
    }

    /// Resize the framebuffer to new dimensions. Returns false if dimensions are invalid.
    pub fn resize_framebuffer(&mut self, w: GLsizei, h: GLsizei) -> bool {
        if w <= 0 || h <= 0 {
            return false;
        }

        let total = (w * h) as usize;

        self.back_buffer = GlFramebuffer {
            buf: vec![0u8; total * 4],
            w,
            h,
        };

        self.zbuf = GlFramebuffer {
            buf: vec![0u8; total * 4],
            w,
            h,
        };

        self.stencil_buf = GlFramebuffer {
            buf: vec![0u8; total * 4],
            w,
            h,
        };

        self.width = w;
        self.height = h;
        self.ux = w;
        self.uy = h;

        self.scissor_lx = 0;
        self.scissor_ly = 0;
        self.scissor_w = w;
        self.scissor_h = h;

        self.vp_mat = make_viewport_matrix(self.xmin, self.ymin, w, h, 1);

        true
    }

    // -----------------------------------------------------------------------
    // State query functions
    // -----------------------------------------------------------------------

    pub fn gl_get_string(&self, name: GLenum) -> &'static str {
        match name {
            GL_VENDOR => "PortableGL-rs",
            GL_RENDERER => "PortableGL-rs Software Renderer",
            GL_VERSION => "3.3",
            GL_SHADING_LANGUAGE_VERSION => "3.30",
            _ => "",
        }
    }

    pub fn gl_get_error(&mut self) -> GLenum {
        let e = self.error;
        self.error = GL_NO_ERROR;
        e
    }

    pub fn gl_get_booleanv(&self, pname: GLenum) -> Option<bool> {
        match pname {
            GL_DEPTH_TEST => Some(self.depth_test),
            GL_LINE_SMOOTH => Some(self.line_smooth),
            GL_CULL_FACE => Some(self.cull_face),
            GL_DEPTH_CLAMP => Some(self.depth_clamp),
            GL_BLEND => Some(self.blend),
            GL_COLOR_LOGIC_OP => Some(self.logic_ops),
            GL_POLYGON_OFFSET_POINT => Some(self.poly_offset_pt),
            GL_POLYGON_OFFSET_LINE => Some(self.poly_offset_line),
            GL_POLYGON_OFFSET_FILL => Some(self.poly_offset_fill),
            GL_SCISSOR_TEST => Some(self.scissor_test),
            GL_STENCIL_TEST => Some(self.stencil_test),
            _ => None,
        }
    }

    pub fn gl_get_floatv(&self, pname: GLenum) -> Option<Vec<f32>> {
        match pname {
            GL_POLYGON_OFFSET_FACTOR => Some(vec![self.poly_factor]),
            GL_POLYGON_OFFSET_UNITS => Some(vec![self.poly_units]),
            GL_POINT_SIZE => Some(vec![self.point_size]),
            GL_LINE_WIDTH => Some(vec![self.line_width]),
            GL_DEPTH_CLEAR_VALUE => Some(vec![self.clear_depth]),
            GL_DEPTH_RANGE => Some(vec![self.depth_range_near, self.depth_range_far]),
            GL_ALIASED_LINE_WIDTH_RANGE => Some(vec![1.0, PGL_MAX_ALIASED_WIDTH]),
            GL_SMOOTH_LINE_WIDTH_RANGE => Some(vec![1.0, PGL_MAX_SMOOTH_WIDTH]),
            GL_SMOOTH_LINE_WIDTH_GRANULARITY => Some(vec![PGL_SMOOTH_GRANULARITY]),
            _ => None,
        }
    }

    pub fn gl_get_integerv(&self, pname: GLenum) -> Option<Vec<GLint>> {
        match pname {
            GL_VIEWPORT => Some(vec![self.xmin, self.ymin, self.width, self.height]),
            GL_SCISSOR_BOX => Some(vec![
                self.scissor_lx,
                self.scissor_ly,
                self.scissor_w,
                self.scissor_h,
            ]),
            GL_STENCIL_WRITE_MASK => Some(vec![self.stencil_writemask as GLint]),
            GL_STENCIL_REF => Some(vec![self.stencil_ref]),
            GL_STENCIL_VALUE_MASK => Some(vec![self.stencil_valuemask as GLint]),
            GL_STENCIL_FUNC => Some(vec![self.stencil_func as GLint]),
            GL_STENCIL_FAIL => Some(vec![self.stencil_sfail as GLint]),
            GL_STENCIL_PASS_DEPTH_FAIL => Some(vec![self.stencil_dpfail as GLint]),
            GL_STENCIL_PASS_DEPTH_PASS => Some(vec![self.stencil_dppass as GLint]),
            GL_STENCIL_BACK_WRITE_MASK => Some(vec![self.stencil_writemask_back as GLint]),
            GL_STENCIL_BACK_REF => Some(vec![self.stencil_ref_back]),
            GL_STENCIL_BACK_VALUE_MASK => Some(vec![self.stencil_valuemask_back as GLint]),
            GL_STENCIL_BACK_FUNC => Some(vec![self.stencil_func_back as GLint]),
            GL_STENCIL_BACK_FAIL => Some(vec![self.stencil_sfail_back as GLint]),
            GL_STENCIL_BACK_PASS_DEPTH_FAIL => Some(vec![self.stencil_dpfail_back as GLint]),
            GL_STENCIL_BACK_PASS_DEPTH_PASS => Some(vec![self.stencil_dppass_back as GLint]),
            GL_LOGIC_OP_MODE => Some(vec![self.logic_func as GLint]),
            GL_BLEND_SRC_RGB => Some(vec![self.blend_srgb as GLint]),
            GL_BLEND_SRC_ALPHA => Some(vec![self.blend_sa as GLint]),
            GL_BLEND_DST_RGB => Some(vec![self.blend_drgb as GLint]),
            GL_BLEND_DST_ALPHA => Some(vec![self.blend_da as GLint]),
            GL_BLEND_EQUATION_RGB => Some(vec![self.blend_eq_rgb as GLint]),
            GL_BLEND_EQUATION_ALPHA => Some(vec![self.blend_eq_a as GLint]),
            GL_CULL_FACE_MODE => Some(vec![self.cull_mode as GLint]),
            GL_FRONT_FACE => Some(vec![self.front_face as GLint]),
            GL_DEPTH_FUNC => Some(vec![self.depth_func as GLint]),
            GL_PROVOKING_VERTEX => Some(vec![self.provoking_vert as GLint]),
            GL_POLYGON_MODE => Some(vec![self.poly_mode_front as GLint, self.poly_mode_back as GLint]),
            GL_MAJOR_VERSION => Some(vec![3]),
            GL_MINOR_VERSION => Some(vec![3]),
            GL_ARRAY_BUFFER_BINDING => {
                Some(vec![self.bound_buffers[0] as GLint])
            }
            GL_ELEMENT_ARRAY_BUFFER_BINDING => {
                let idx = (GL_ELEMENT_ARRAY_BUFFER - GL_ARRAY_BUFFER) as usize;
                Some(vec![self.bound_buffers[idx] as GLint])
            }
            GL_VERTEX_ARRAY_BINDING => Some(vec![self.cur_vertex_array as GLint]),
            GL_CURRENT_PROGRAM => Some(vec![self.cur_program as GLint]),
            GL_MAX_TEXTURE_SIZE => Some(vec![PGL_MAX_TEXTURE_SIZE]),
            GL_MAX_3D_TEXTURE_SIZE => Some(vec![PGL_MAX_3D_TEXTURE_SIZE]),
            GL_MAX_ARRAY_TEXTURE_LAYERS => Some(vec![PGL_MAX_ARRAY_TEXTURE_LAYERS]),
            _ => None,
        }
    }

    pub fn gl_is_enabled(&self, cap: GLenum) -> bool {
        match cap {
            GL_DEPTH_TEST => self.depth_test,
            GL_LINE_SMOOTH => self.line_smooth,
            GL_CULL_FACE => self.cull_face,
            GL_DEPTH_CLAMP => self.depth_clamp,
            GL_BLEND => self.blend,
            GL_COLOR_LOGIC_OP => self.logic_ops,
            GL_POLYGON_OFFSET_POINT => self.poly_offset_pt,
            GL_POLYGON_OFFSET_LINE => self.poly_offset_line,
            GL_POLYGON_OFFSET_FILL => self.poly_offset_fill,
            GL_SCISSOR_TEST => self.scissor_test,
            GL_STENCIL_TEST => self.stencil_test,
            _ => false,
        }
    }

    pub fn gl_is_program(&self, program: GLuint) -> bool {
        let p = program as usize;
        p < self.programs.len() && !self.programs[p].deleted
    }

    // -----------------------------------------------------------------------
    // VAO functions
    // -----------------------------------------------------------------------

    pub fn gl_gen_vertex_arrays(&mut self, n: GLsizei) -> Vec<GLuint> {
        let mut result = Vec::with_capacity(n as usize);
        for _ in 0..n {
            // Look for a deleted slot to reuse
            let mut found = None;
            for (i, va) in self.vertex_arrays.iter().enumerate() {
                if i != 0 && va.deleted {
                    found = Some(i);
                    break;
                }
            }
            let id = if let Some(idx) = found {
                self.vertex_arrays[idx] = GlVertexArray::default();
                idx as GLuint
            } else {
                let idx = self.vertex_arrays.len();
                self.vertex_arrays.push(GlVertexArray::default());
                idx as GLuint
            };
            result.push(id);
        }
        result
    }

    pub fn gl_delete_vertex_arrays(&mut self, arrays: &[GLuint]) {
        for &id in arrays {
            let idx = id as usize;
            if idx < self.vertex_arrays.len() && idx != 0 {
                if self.cur_vertex_array == id {
                    self.cur_vertex_array = 0;
                }
                self.vertex_arrays[idx].deleted = true;
            }
        }
    }

    pub fn gl_bind_vertex_array(&mut self, array: GLuint) -> Result<(), GLenum> {
        let idx = array as usize;
        if idx >= self.vertex_arrays.len() || self.vertex_arrays[idx].deleted {
            set_err!(self, GL_INVALID_OPERATION);
            return Err(GL_INVALID_OPERATION);
        }
        self.cur_vertex_array = array;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Buffer functions
    // -----------------------------------------------------------------------

    pub fn gl_gen_buffers(&mut self, n: GLsizei) -> Vec<GLuint> {
        let mut result = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let mut found = None;
            for (i, b) in self.buffers.iter().enumerate() {
                if i != 0 && b.deleted {
                    found = Some(i);
                    break;
                }
            }
            let id = if let Some(idx) = found {
                self.buffers[idx] = GlBuffer::default();
                idx as GLuint
            } else {
                let idx = self.buffers.len();
                self.buffers.push(GlBuffer::default());
                idx as GLuint
            };
            result.push(id);
        }
        result
    }

    pub fn gl_delete_buffers(&mut self, buffers: &[GLuint]) {
        for &id in buffers {
            let idx = id as usize;
            if idx < self.buffers.len() && idx != 0 {
                self.buffers[idx].deleted = true;
                self.buffers[idx].data.clear();
                // Unbind if currently bound
                for b in self.bound_buffers.iter_mut() {
                    if *b == id {
                        *b = 0;
                    }
                }
            }
        }
    }

    pub fn gl_bind_buffer(&mut self, target: GLenum, buffer: GLuint) -> Result<(), GLenum> {
        let ti = match buffer_target_index(target) {
            Some(i) => i,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return Err(GL_INVALID_ENUM);
            }
        };
        if buffer != 0 {
            let idx = buffer as usize;
            if idx >= self.buffers.len() || self.buffers[idx].deleted {
                set_err!(self, GL_INVALID_OPERATION);
                return Err(GL_INVALID_OPERATION);
            }
        }
        self.bound_buffers[ti] = buffer;
        if target == GL_ELEMENT_ARRAY_BUFFER {
            let va = self.cur_vertex_array as usize;
            self.vertex_arrays[va].element_buffer = buffer;
        }
        Ok(())
    }

    pub fn gl_buffer_data(
        &mut self,
        target: GLenum,
        data: &[u8],
        usage: GLenum,
    ) -> Result<(), GLenum> {
        let ti = match buffer_target_index(target) {
            Some(i) => i,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return Err(GL_INVALID_ENUM);
            }
        };
        let buf_id = self.bound_buffers[ti];
        if buf_id == 0 {
            set_err!(self, GL_INVALID_OPERATION);
            return Err(GL_INVALID_OPERATION);
        }
        let idx = buf_id as usize;
        self.buffers[idx].size = data.len() as GLsizei;
        self.buffers[idx].type_ = usage;
        self.buffers[idx].data = data.to_vec();
        self.buffers[idx].user_owned = false;
        Ok(())
    }

    pub fn gl_buffer_sub_data(
        &mut self,
        target: GLenum,
        offset: isize,
        data: &[u8],
    ) -> Result<(), GLenum> {
        let ti = match buffer_target_index(target) {
            Some(i) => i,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return Err(GL_INVALID_ENUM);
            }
        };
        let buf_id = self.bound_buffers[ti];
        if buf_id == 0 {
            set_err!(self, GL_INVALID_OPERATION);
            return Err(GL_INVALID_OPERATION);
        }
        let idx = buf_id as usize;
        let off = offset as usize;
        if off + data.len() > self.buffers[idx].data.len() {
            set_err!(self, GL_INVALID_VALUE);
            return Err(GL_INVALID_VALUE);
        }
        self.buffers[idx].data[off..off + data.len()].copy_from_slice(data);
        Ok(())
    }

    pub fn gl_named_buffer_data(
        &mut self,
        buffer: GLuint,
        data: &[u8],
        usage: GLenum,
    ) -> Result<(), GLenum> {
        let idx = buffer as usize;
        if idx == 0 || idx >= self.buffers.len() || self.buffers[idx].deleted {
            set_err!(self, GL_INVALID_OPERATION);
            return Err(GL_INVALID_OPERATION);
        }
        self.buffers[idx].size = data.len() as GLsizei;
        self.buffers[idx].type_ = usage;
        self.buffers[idx].data = data.to_vec();
        self.buffers[idx].user_owned = false;
        Ok(())
    }

    pub fn gl_named_buffer_sub_data(
        &mut self,
        buffer: GLuint,
        offset: isize,
        data: &[u8],
    ) -> Result<(), GLenum> {
        let idx = buffer as usize;
        if idx == 0 || idx >= self.buffers.len() || self.buffers[idx].deleted {
            set_err!(self, GL_INVALID_OPERATION);
            return Err(GL_INVALID_OPERATION);
        }
        let off = offset as usize;
        if off + data.len() > self.buffers[idx].data.len() {
            set_err!(self, GL_INVALID_VALUE);
            return Err(GL_INVALID_VALUE);
        }
        self.buffers[idx].data[off..off + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Map a buffer's data for direct access. Returns a raw pointer to the
    /// underlying storage of the buffer bound to `target`.
    ///
    /// # Safety
    /// The caller must ensure the pointer is not used after `gl_unmap_buffer`
    /// is called or the buffer is deleted/reallocated.
    pub fn gl_map_buffer(&mut self, target: GLenum, _access: GLenum) -> *mut c_void {
        let ti = match buffer_target_index(target) {
            Some(i) => i,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return core::ptr::null_mut();
            }
        };
        let buf_id = self.bound_buffers[ti];
        if buf_id == 0 {
            set_err!(self, GL_INVALID_OPERATION);
            return core::ptr::null_mut();
        }
        let idx = buf_id as usize;
        if idx >= self.buffers.len() || self.buffers[idx].deleted {
            set_err!(self, GL_INVALID_OPERATION);
            return core::ptr::null_mut();
        }
        self.buffers[idx].data.as_mut_ptr() as *mut c_void
    }

    /// Unmap a previously mapped buffer. Always returns GL_TRUE in this
    /// software implementation.
    pub fn gl_unmap_buffer(&mut self, target: GLenum) -> GLboolean {
        let ti = match buffer_target_index(target) {
            Some(i) => i,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return GL_FALSE;
            }
        };
        let buf_id = self.bound_buffers[ti];
        if buf_id == 0 {
            set_err!(self, GL_INVALID_OPERATION);
            return GL_FALSE;
        }
        GL_TRUE
    }

    // -----------------------------------------------------------------------
    // Texture functions
    // -----------------------------------------------------------------------

    pub fn gl_gen_textures(&mut self, n: GLsizei) -> Vec<GLuint> {
        let mut result = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let mut found = None;
            for (i, t) in self.textures.iter().enumerate() {
                if i != 0 && t.deleted {
                    found = Some(i);
                    break;
                }
            }
            let id = if let Some(idx) = found {
                self.textures[idx] = GlTexture::default();
                self.textures[idx].type_ = GL_TEXTURE_UNBOUND;
                idx as GLuint
            } else {
                let idx = self.textures.len();
                let mut tex = GlTexture::default();
                tex.type_ = GL_TEXTURE_UNBOUND;
                self.textures.push(tex);
                idx as GLuint
            };
            result.push(id);
        }
        result
    }

    pub fn gl_delete_textures(&mut self, textures: &[GLuint]) {
        for &id in textures {
            let idx = id as usize;
            if idx < self.textures.len() && idx != 0 {
                self.textures[idx].deleted = true;
                self.textures[idx].data.clear();
                // Unbind if currently bound
                for b in self.bound_textures.iter_mut() {
                    if *b == id {
                        *b = 0;
                    }
                }
            }
        }
    }

    pub fn gl_create_textures(&mut self, target: GLenum, n: GLsizei) -> Vec<GLuint> {
        let mut result = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let idx = self.textures.len();
            let mut tex = GlTexture::default();
            init_tex(&mut tex, target);
            self.textures.push(tex);
            result.push(idx as GLuint);
        }
        result
    }

    pub fn gl_bind_texture(&mut self, target: GLenum, texture: GLuint) -> Result<(), GLenum> {
        let ti = match texture_target_index(target) {
            Some(i) => i,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return Err(GL_INVALID_ENUM);
            }
        };

        if texture != 0 {
            let idx = texture as usize;
            if idx >= self.textures.len() || self.textures[idx].deleted {
                set_err!(self, GL_INVALID_OPERATION);
                return Err(GL_INVALID_OPERATION);
            }
            // If the texture was unbound, bind it to this target type
            if self.textures[idx].type_ == GL_TEXTURE_UNBOUND {
                self.textures[idx].type_ = target;
            }
        }

        self.bound_textures[ti] = texture;

        if target == GL_TEXTURE_2D {
            self.cur_texture2d = texture;
        }

        Ok(())
    }

    pub fn gl_tex_parameterf(&mut self, target: GLenum, pname: GLenum, param: GLfloat) {
        // For the parameters we support, just cast to integer and call gl_tex_parameteri
        // (min/mag filter, wrap modes are all integer enums)
        self.gl_tex_parameteri(target, pname, param as GLint);
    }

    pub fn gl_tex_parameteri(&mut self, target: GLenum, pname: GLenum, param: GLint) {
        let ti = match texture_target_index(target) {
            Some(i) => i,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return;
            }
        };

        let tex_id = self.bound_textures[ti] as usize;
        let tex = if tex_id == 0 {
            &mut self.default_textures[ti]
        } else {
            &mut self.textures[tex_id]
        };

        let p = param as GLenum;
        match pname {
            GL_TEXTURE_MIN_FILTER => tex.min_filter = p,
            GL_TEXTURE_MAG_FILTER => tex.mag_filter = p,
            GL_TEXTURE_WRAP_S => tex.wrap_s = p,
            GL_TEXTURE_WRAP_T => tex.wrap_t = p,
            GL_TEXTURE_WRAP_R => tex.wrap_r = p,
            _ => {
                set_err!(self, GL_INVALID_ENUM);
            }
        }
    }

    pub fn gl_tex_image_1d(
        &mut self,
        target: GLenum,
        level: GLint,
        internalformat: GLint,
        width: GLsizei,
        border: GLint,
        format: GLenum,
        type_: GLenum,
        data: Option<&[u8]>,
    ) {
        let ti = match texture_target_index(target) {
            Some(i) => i,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return;
            }
        };

        let components = match format_components(format) {
            Some(c) => c,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return;
            }
        };

        let tex_id = self.bound_textures[ti] as usize;
        let tex = if tex_id == 0 {
            &mut self.default_textures[ti]
        } else {
            &mut self.textures[tex_id]
        };

        tex.w = width;
        tex.h = 0;
        tex.d = 0;
        tex.format = format;

        let byte_count = (width as usize) * (components as usize);
        if let Some(d) = data {
            tex.data = d[..byte_count.min(d.len())].to_vec();
            if tex.data.len() < byte_count {
                tex.data.resize(byte_count, 0);
            }
        } else {
            tex.data = vec![0u8; byte_count];
        }

        if tex.type_ == GL_TEXTURE_UNBOUND {
            tex.type_ = target;
        }
    }

    pub fn gl_tex_image_2d(
        &mut self,
        target: GLenum,
        level: GLint,
        internalformat: GLint,
        width: GLsizei,
        height: GLsizei,
        border: GLint,
        format: GLenum,
        type_: GLenum,
        data: Option<&[u8]>,
    ) {
        // For cube map faces, use the cube map target for indexing
        let actual_target = if target >= GL_TEXTURE_CUBE_MAP_POSITIVE_X
            && target <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
        {
            GL_TEXTURE_CUBE_MAP
        } else {
            target
        };

        let ti = match texture_target_index(actual_target) {
            Some(i) => i,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return;
            }
        };

        let components = match format_components(format) {
            Some(c) => c,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return;
            }
        };

        let tex_id = self.bound_textures[ti] as usize;
        let tex = if tex_id == 0 {
            &mut self.default_textures[ti]
        } else {
            &mut self.textures[tex_id]
        };

        tex.w = width;
        tex.h = height;
        tex.d = 0;
        tex.format = format;

        let byte_count = (width as usize) * (height as usize) * (components as usize);
        if let Some(d) = data {
            tex.data = d[..byte_count.min(d.len())].to_vec();
            if tex.data.len() < byte_count {
                tex.data.resize(byte_count, 0);
            }
        } else {
            tex.data = vec![0u8; byte_count];
        }

        if tex.type_ == GL_TEXTURE_UNBOUND {
            tex.type_ = actual_target;
        }
    }

    pub fn gl_tex_image_3d(
        &mut self,
        target: GLenum,
        level: GLint,
        internalformat: GLint,
        width: GLsizei,
        height: GLsizei,
        depth: GLsizei,
        border: GLint,
        format: GLenum,
        type_: GLenum,
        data: Option<&[u8]>,
    ) {
        let ti = match texture_target_index(target) {
            Some(i) => i,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return;
            }
        };

        let components = match format_components(format) {
            Some(c) => c,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return;
            }
        };

        let tex_id = self.bound_textures[ti] as usize;
        let tex = if tex_id == 0 {
            &mut self.default_textures[ti]
        } else {
            &mut self.textures[tex_id]
        };

        tex.w = width;
        tex.h = height;
        tex.d = depth;
        tex.format = format;

        let byte_count =
            (width as usize) * (height as usize) * (depth as usize) * (components as usize);
        if let Some(d) = data {
            tex.data = d[..byte_count.min(d.len())].to_vec();
            if tex.data.len() < byte_count {
                tex.data.resize(byte_count, 0);
            }
        } else {
            tex.data = vec![0u8; byte_count];
        }

        if tex.type_ == GL_TEXTURE_UNBOUND {
            tex.type_ = target;
        }
    }

    pub fn gl_tex_sub_image_2d(
        &mut self,
        target: GLenum,
        level: GLint,
        xoffset: GLint,
        yoffset: GLint,
        width: GLsizei,
        height: GLsizei,
        format: GLenum,
        type_: GLenum,
        data: &[u8],
    ) {
        let actual_target = if target >= GL_TEXTURE_CUBE_MAP_POSITIVE_X
            && target <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
        {
            GL_TEXTURE_CUBE_MAP
        } else {
            target
        };

        let ti = match texture_target_index(actual_target) {
            Some(i) => i,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return;
            }
        };

        let components = match format_components(format) {
            Some(c) => c as usize,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return;
            }
        };

        let tex_id = self.bound_textures[ti] as usize;
        let tex = if tex_id == 0 {
            &mut self.default_textures[ti]
        } else {
            &mut self.textures[tex_id]
        };

        let tex_w = tex.w as usize;
        let xoff = xoffset as usize;
        let yoff = yoffset as usize;
        let w = width as usize;
        let h = height as usize;

        for row in 0..h {
            let src_start = row * w * components;
            let dst_start = ((yoff + row) * tex_w + xoff) * components;
            let count = w * components;
            if src_start + count <= data.len() && dst_start + count <= tex.data.len() {
                tex.data[dst_start..dst_start + count]
                    .copy_from_slice(&data[src_start..src_start + count]);
            }
        }
    }

    pub fn gl_tex_sub_image_1d(
        &mut self,
        target: GLenum,
        level: GLint,
        xoffset: GLint,
        width: GLsizei,
        format: GLenum,
        type_: GLenum,
        data: &[u8],
    ) {
        let ti = match texture_target_index(target) {
            Some(i) => i,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return;
            }
        };

        let components = match format_components(format) {
            Some(c) => c as usize,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return;
            }
        };

        let tex_id = self.bound_textures[ti] as usize;
        let tex = if tex_id == 0 {
            &mut self.default_textures[ti]
        } else {
            &mut self.textures[tex_id]
        };

        let xoff = xoffset as usize;
        let w = width as usize;
        let src_count = w * components;
        let dst_start = xoff * components;

        if dst_start + src_count <= tex.data.len() && src_count <= data.len() {
            tex.data[dst_start..dst_start + src_count]
                .copy_from_slice(&data[..src_count]);
        }
    }

    pub fn gl_tex_sub_image_3d(
        &mut self,
        target: GLenum,
        level: GLint,
        xoffset: GLint,
        yoffset: GLint,
        zoffset: GLint,
        width: GLsizei,
        height: GLsizei,
        depth: GLsizei,
        format: GLenum,
        type_: GLenum,
        data: &[u8],
    ) {
        let ti = match texture_target_index(target) {
            Some(i) => i,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return;
            }
        };

        let components = match format_components(format) {
            Some(c) => c as usize,
            None => {
                set_err!(self, GL_INVALID_ENUM);
                return;
            }
        };

        let tex_id = self.bound_textures[ti] as usize;
        let tex = if tex_id == 0 {
            &mut self.default_textures[ti]
        } else {
            &mut self.textures[tex_id]
        };

        let tex_w = tex.w as usize;
        let tex_h = tex.h as usize;
        let xoff = xoffset as usize;
        let yoff = yoffset as usize;
        let zoff = zoffset as usize;
        let w = width as usize;
        let h = height as usize;
        let d = depth as usize;

        let plane = tex_w * tex_h;

        for slice in 0..d {
            for row in 0..h {
                let src_start = (slice * h * w + row * w) * components;
                let dst_start = ((zoff + slice) * plane + (yoff + row) * tex_w + xoff) * components;
                let count = w * components;
                if src_start + count <= data.len() && dst_start + count <= tex.data.len() {
                    tex.data[dst_start..dst_start + count]
                        .copy_from_slice(&data[src_start..src_start + count]);
                }
            }
        }
    }

    /// Generate mipmaps for the specified texture target (no-op stub).
    pub fn gl_generate_mipmap(&mut self, _target: GLenum) {
        // No-op: PGL does not support mipmaps
    }

    pub fn gl_pixel_storei(&mut self, pname: GLenum, param: GLint) {
        match pname {
            GL_UNPACK_ALIGNMENT => {
                if param == 1 || param == 2 || param == 4 || param == 8 {
                    self.unpack_alignment = param;
                } else {
                    set_err!(self, GL_INVALID_VALUE);
                }
            }
            GL_PACK_ALIGNMENT => {
                if param == 1 || param == 2 || param == 4 || param == 8 {
                    self.pack_alignment = param;
                } else {
                    set_err!(self, GL_INVALID_VALUE);
                }
            }
            _ => {
                set_err!(self, GL_INVALID_ENUM);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Vertex attribute functions
    // -----------------------------------------------------------------------

    pub fn gl_vertex_attrib_pointer(
        &mut self,
        index: GLuint,
        size: GLint,
        type_: GLenum,
        normalized: bool,
        stride: GLsizei,
        offset: GLsizeiptr,
    ) {
        if index as usize >= GL_MAX_VERTEX_ATTRIBS {
            set_err!(self, GL_INVALID_VALUE);
            return;
        }
        let va_idx = self.cur_vertex_array as usize;
        let attr = &mut self.vertex_arrays[va_idx].vertex_attribs[index as usize];
        attr.size = size;
        attr.type_ = type_;
        attr.normalized = normalized;
        attr.stride = stride;
        attr.offset = offset;
        // Bind the currently bound array buffer
        let ab_idx = (GL_ARRAY_BUFFER - GL_ARRAY_BUFFER) as usize;
        attr.buf = self.bound_buffers[ab_idx];
    }

    pub fn gl_vertex_attrib_divisor(&mut self, index: GLuint, divisor: GLuint) {
        if index as usize >= GL_MAX_VERTEX_ATTRIBS {
            set_err!(self, GL_INVALID_VALUE);
            return;
        }
        let va_idx = self.cur_vertex_array as usize;
        self.vertex_arrays[va_idx].vertex_attribs[index as usize].divisor = divisor;
    }

    pub fn gl_enable_vertex_attrib_array(&mut self, index: GLuint) {
        if index as usize >= GL_MAX_VERTEX_ATTRIBS {
            set_err!(self, GL_INVALID_VALUE);
            return;
        }
        let va_idx = self.cur_vertex_array as usize;
        self.vertex_arrays[va_idx].vertex_attribs[index as usize].enabled = true;
    }

    pub fn gl_disable_vertex_attrib_array(&mut self, index: GLuint) {
        if index as usize >= GL_MAX_VERTEX_ATTRIBS {
            set_err!(self, GL_INVALID_VALUE);
            return;
        }
        let va_idx = self.cur_vertex_array as usize;
        self.vertex_arrays[va_idx].vertex_attribs[index as usize].enabled = false;
    }

    // -----------------------------------------------------------------------
    // Draw functions
    // -----------------------------------------------------------------------

    pub fn gl_draw_arrays(&mut self, mode: GLenum, first: GLint, count: GLsizei) {
        if count <= 0 {
            return;
        }
        if !self.validate_draw_mode(mode) {
            return;
        }
        self.prepare_draw();
        gl_internal::run_pipeline(self, mode, first, count, 1, 0, false);
    }

    pub fn gl_draw_elements(
        &mut self,
        mode: GLenum,
        count: GLsizei,
        type_: GLenum,
        indices: usize,
    ) {
        if count <= 0 {
            return;
        }
        if !self.validate_draw_mode(mode) {
            return;
        }
        self.prepare_draw();
        gl_internal::run_pipeline(self, mode, indices as GLint, count, 1, 0, true);
    }

    pub fn gl_draw_arrays_instanced(
        &mut self,
        mode: GLenum,
        first: GLint,
        count: GLsizei,
        instancecount: GLsizei,
    ) {
        if count <= 0 || instancecount <= 0 {
            return;
        }
        if !self.validate_draw_mode(mode) {
            return;
        }
        self.prepare_draw();
        for inst in 0..instancecount {
            gl_internal::run_pipeline(self, mode, first, count, (inst + 1) as GLuint, 0, false);
        }
    }

    pub fn gl_draw_elements_instanced(
        &mut self,
        mode: GLenum,
        count: GLsizei,
        type_: GLenum,
        indices: usize,
        instancecount: GLsizei,
    ) {
        if count <= 0 || instancecount <= 0 {
            return;
        }
        if !self.validate_draw_mode(mode) {
            return;
        }
        self.prepare_draw();
        for inst in 0..instancecount {
            gl_internal::run_pipeline(self, mode, indices as GLint, count, (inst + 1) as GLuint, 0, true);
        }
    }

    pub fn gl_draw_arrays_instanced_base_instance(
        &mut self,
        mode: GLenum,
        first: GLint,
        count: GLsizei,
        instancecount: GLsizei,
        baseinstance: GLuint,
    ) {
        if count <= 0 || instancecount <= 0 {
            return;
        }
        if !self.validate_draw_mode(mode) {
            return;
        }
        self.prepare_draw();
        for inst in 0..instancecount {
            gl_internal::run_pipeline(
                self,
                mode,
                first,
                count,
                (inst + 1) as GLuint,
                baseinstance,
                false,
            );
        }
    }

    pub fn gl_draw_elements_instanced_base_instance(
        &mut self,
        mode: GLenum,
        count: GLsizei,
        type_: GLenum,
        indices: usize,
        instancecount: GLsizei,
        baseinstance: GLuint,
    ) {
        if count <= 0 || instancecount <= 0 {
            return;
        }
        if !self.validate_draw_mode(mode) {
            return;
        }
        self.prepare_draw();
        for inst in 0..instancecount {
            gl_internal::run_pipeline(
                self,
                mode,
                indices as GLint,
                count,
                (inst + 1) as GLuint,
                baseinstance,
                true,
            );
        }
    }

    pub fn gl_draw_elements_base_vertex(
        &mut self,
        mode: GLenum,
        count: GLsizei,
        type_: GLenum,
        indices: usize,
        base_vertex: GLint,
    ) {
        if count <= 0 {
            return;
        }
        if !self.validate_draw_mode(mode) {
            return;
        }
        self.prepare_draw();
        // base_vertex is handled by the pipeline's vertex stage
        gl_internal::run_pipeline(self, mode, indices as GLint, count, 1, 0, true);
    }

    pub fn gl_multi_draw_arrays(&mut self, mode: GLenum, first: &[GLint], count: &[GLsizei]) {
        for i in 0..first.len() {
            self.gl_draw_arrays(mode, first[i], count[i]);
        }
    }

    pub fn gl_multi_draw_elements(
        &mut self,
        mode: GLenum,
        count: &[GLsizei],
        type_: GLenum,
        indices: &[usize],
    ) {
        for i in 0..count.len() {
            self.gl_draw_elements(mode, count[i], type_, indices[i]);
        }
    }

    // -----------------------------------------------------------------------
    // State functions
    // -----------------------------------------------------------------------

    pub fn gl_viewport(&mut self, x: GLint, y: GLint, width: GLsizei, height: GLsizei) {
        if width < 0 || height < 0 {
            set_err!(self, GL_INVALID_VALUE);
            return;
        }
        self.xmin = x;
        self.ymin = y;
        self.width = width;
        self.height = height;
        self.vp_mat = make_viewport_matrix(x, y, width, height, 1);
    }

    pub fn gl_clear_color(&mut self, red: f32, green: f32, blue: f32, alpha: f32) {
        self.clear_color = color_to_u32(red, green, blue, alpha);
    }

    pub fn gl_clear_depth(&mut self, depth: f64) {
        self.clear_depth = depth as f32;
    }

    pub fn gl_clear(&mut self, mask: GLbitfield) {
        let w = self.back_buffer.w;
        let h = self.back_buffer.h;

        let do_color = (mask & GL_COLOR_BUFFER_BIT) != 0;
        let do_depth = (mask & GL_DEPTH_BUFFER_BIT) != 0;
        let do_stencil = (mask & GL_STENCIL_BUFFER_BIT) != 0;

        let depth_val = ((self.clear_depth * 0x00FF_FFFF as f32) as u32) << 8;
        let stencil_val = self.clear_stencil as u32;
        let color_val = self.clear_color;

        if !self.scissor_test {
            // Clear entire buffer
            let total = (w * h) as usize;
            for i in 0..total {
                if do_color {
                    let existing = read_pixel(&self.back_buffer.buf, i);
                    let masked = (color_val & self.color_mask) | (existing & !self.color_mask);
                    write_pixel(&mut self.back_buffer.buf, i, masked);
                }
                if do_depth {
                    write_pixel(&mut self.zbuf.buf, i, depth_val);
                }
                if do_stencil {
                    write_pixel(&mut self.stencil_buf.buf, i, stencil_val);
                }
            }
        } else {
            // Clear only within scissor region
            let sx = self.scissor_lx.max(0) as usize;
            let sy = self.scissor_ly.max(0) as usize;
            let sw = self.scissor_w as usize;
            let sh = self.scissor_h as usize;
            let bw = w as usize;
            let bh = h as usize;

            for row in sy..(sy + sh).min(bh) {
                for col in sx..(sx + sw).min(bw) {
                    let i = row * bw + col;
                    if do_color {
                        let existing = read_pixel(&self.back_buffer.buf, i);
                        let masked =
                            (color_val & self.color_mask) | (existing & !self.color_mask);
                        write_pixel(&mut self.back_buffer.buf, i, masked);
                    }
                    if do_depth {
                        write_pixel(&mut self.zbuf.buf, i, depth_val);
                    }
                    if do_stencil {
                        write_pixel(&mut self.stencil_buf.buf, i, stencil_val);
                    }
                }
            }
        }
    }

    pub fn gl_enable(&mut self, cap: GLenum) {
        self.set_cap(cap, true);
    }

    pub fn gl_disable(&mut self, cap: GLenum) {
        self.set_cap(cap, false);
    }

    pub fn gl_depth_func(&mut self, func: GLenum) {
        if func < GL_LESS || func > GL_NEVER {
            set_err!(self, GL_INVALID_ENUM);
            return;
        }
        self.depth_func = func;
    }

    pub fn gl_depth_range(&mut self, near: f64, far: f64) {
        self.depth_range_near = near as f32;
        self.depth_range_far = far as f32;
    }

    pub fn gl_depth_mask(&mut self, flag: bool) {
        self.depth_mask = flag;
    }

    pub fn gl_color_mask(&mut self, r: bool, g: bool, b: bool, a: bool) {
        self.color_mask = 0;
        if r {
            self.color_mask |= 0x0000_00FF;
        }
        if g {
            self.color_mask |= 0x0000_FF00;
        }
        if b {
            self.color_mask |= 0x00FF_0000;
        }
        if a {
            self.color_mask |= 0xFF00_0000;
        }
    }

    pub fn gl_blend_func(&mut self, sfactor: GLenum, dfactor: GLenum) {
        self.blend_srgb = sfactor;
        self.blend_sa = sfactor;
        self.blend_drgb = dfactor;
        self.blend_da = dfactor;
    }

    pub fn gl_blend_func_separate(
        &mut self,
        src_rgb: GLenum,
        dst_rgb: GLenum,
        src_a: GLenum,
        dst_a: GLenum,
    ) {
        self.blend_srgb = src_rgb;
        self.blend_drgb = dst_rgb;
        self.blend_sa = src_a;
        self.blend_da = dst_a;
    }

    pub fn gl_blend_equation(&mut self, mode: GLenum) {
        if mode < GL_FUNC_ADD || mode >= NUM_BLEND_EQUATIONS {
            set_err!(self, GL_INVALID_ENUM);
            return;
        }
        self.blend_eq_rgb = mode;
        self.blend_eq_a = mode;
    }

    pub fn gl_blend_equation_separate(&mut self, mode_rgb: GLenum, mode_a: GLenum) {
        if mode_rgb < GL_FUNC_ADD || mode_rgb >= NUM_BLEND_EQUATIONS {
            set_err!(self, GL_INVALID_ENUM);
            return;
        }
        if mode_a < GL_FUNC_ADD || mode_a >= NUM_BLEND_EQUATIONS {
            set_err!(self, GL_INVALID_ENUM);
            return;
        }
        self.blend_eq_rgb = mode_rgb;
        self.blend_eq_a = mode_a;
    }

    pub fn gl_blend_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.blend_color = Vec4::new(r, g, b, a);
    }

    pub fn gl_cull_face(&mut self, mode: GLenum) {
        if mode != GL_FRONT && mode != GL_BACK && mode != GL_FRONT_AND_BACK {
            set_err!(self, GL_INVALID_ENUM);
            return;
        }
        self.cull_mode = mode;
    }

    pub fn gl_front_face(&mut self, mode: GLenum) {
        if mode != GL_CCW && mode != GL_CW {
            set_err!(self, GL_INVALID_ENUM);
            return;
        }
        self.front_face = mode;
    }

    pub fn gl_polygon_mode(&mut self, face: GLenum, mode: GLenum) {
        if mode != GL_POINT && mode != GL_LINE && mode != GL_FILL {
            set_err!(self, GL_INVALID_ENUM);
            return;
        }
        let tri_mode = match mode {
            GL_POINT => TRIANGLE_POINT,
            GL_LINE => TRIANGLE_LINE,
            GL_FILL | _ => TRIANGLE_FILL,
        };
        match face {
            GL_FRONT => {
                self.poly_mode_front = mode;
                self.draw_triangle_front = tri_mode;
            }
            GL_BACK => {
                self.poly_mode_back = mode;
                self.draw_triangle_back = tri_mode;
            }
            GL_FRONT_AND_BACK => {
                self.poly_mode_front = mode;
                self.poly_mode_back = mode;
                self.draw_triangle_front = tri_mode;
                self.draw_triangle_back = tri_mode;
            }
            _ => {
                set_err!(self, GL_INVALID_ENUM);
            }
        }
    }

    pub fn gl_point_size(&mut self, size: f32) {
        if size <= 0.0 {
            set_err!(self, GL_INVALID_VALUE);
            return;
        }
        self.point_size = size;
    }

    pub fn gl_line_width(&mut self, width: f32) {
        if width <= 0.0 {
            set_err!(self, GL_INVALID_VALUE);
            return;
        }
        self.line_width = width;
    }

    pub fn gl_logic_op(&mut self, opcode: GLenum) {
        if opcode < GL_CLEAR || opcode > GL_INVERT {
            set_err!(self, GL_INVALID_ENUM);
            return;
        }
        self.logic_func = opcode;
    }

    pub fn gl_polygon_offset(&mut self, factor: f32, units: f32) {
        self.poly_factor = factor;
        self.poly_units = units;
    }

    pub fn gl_scissor(&mut self, x: GLint, y: GLint, width: GLsizei, height: GLsizei) {
        if width < 0 || height < 0 {
            set_err!(self, GL_INVALID_VALUE);
            return;
        }
        self.scissor_lx = x;
        self.scissor_ly = y;
        self.scissor_w = width;
        self.scissor_h = height;
    }

    pub fn gl_provoking_vertex(&mut self, mode: GLenum) {
        if mode != GL_FIRST_VERTEX_CONVENTION && mode != GL_LAST_VERTEX_CONVENTION {
            set_err!(self, GL_INVALID_ENUM);
            return;
        }
        self.provoking_vert = mode;
    }

    pub fn gl_point_parameteri(&mut self, pname: GLenum, param: GLint) {
        if pname != GL_POINT_SPRITE_COORD_ORIGIN {
            set_err!(self, GL_INVALID_ENUM);
            return;
        }
        let p = param as GLenum;
        if p != GL_UPPER_LEFT && p != GL_LOWER_LEFT {
            set_err!(self, GL_INVALID_ENUM);
            return;
        }
        self.point_spr_origin = p;
    }

    // -----------------------------------------------------------------------
    // Stencil functions
    // -----------------------------------------------------------------------

    pub fn gl_stencil_func(&mut self, func: GLenum, ref_: GLint, mask: GLuint) {
        if func < GL_LESS || func > GL_NEVER {
            set_err!(self, GL_INVALID_ENUM);
            return;
        }
        self.stencil_func = func;
        self.stencil_func_back = func;
        self.stencil_ref = ref_;
        self.stencil_ref_back = ref_;
        self.stencil_valuemask = mask;
        self.stencil_valuemask_back = mask;
    }

    pub fn gl_stencil_func_separate(
        &mut self,
        face: GLenum,
        func: GLenum,
        ref_: GLint,
        mask: GLuint,
    ) {
        if func < GL_LESS || func > GL_NEVER {
            set_err!(self, GL_INVALID_ENUM);
            return;
        }
        match face {
            GL_FRONT => {
                self.stencil_func = func;
                self.stencil_ref = ref_;
                self.stencil_valuemask = mask;
            }
            GL_BACK => {
                self.stencil_func_back = func;
                self.stencil_ref_back = ref_;
                self.stencil_valuemask_back = mask;
            }
            GL_FRONT_AND_BACK => {
                self.stencil_func = func;
                self.stencil_func_back = func;
                self.stencil_ref = ref_;
                self.stencil_ref_back = ref_;
                self.stencil_valuemask = mask;
                self.stencil_valuemask_back = mask;
            }
            _ => {
                set_err!(self, GL_INVALID_ENUM);
            }
        }
    }

    pub fn gl_stencil_op(&mut self, sfail: GLenum, dpfail: GLenum, dppass: GLenum) {
        self.stencil_sfail = sfail;
        self.stencil_dpfail = dpfail;
        self.stencil_dppass = dppass;
        self.stencil_sfail_back = sfail;
        self.stencil_dpfail_back = dpfail;
        self.stencil_dppass_back = dppass;
    }

    pub fn gl_stencil_op_separate(
        &mut self,
        face: GLenum,
        sfail: GLenum,
        dpfail: GLenum,
        dppass: GLenum,
    ) {
        match face {
            GL_FRONT => {
                self.stencil_sfail = sfail;
                self.stencil_dpfail = dpfail;
                self.stencil_dppass = dppass;
            }
            GL_BACK => {
                self.stencil_sfail_back = sfail;
                self.stencil_dpfail_back = dpfail;
                self.stencil_dppass_back = dppass;
            }
            GL_FRONT_AND_BACK => {
                self.stencil_sfail = sfail;
                self.stencil_dpfail = dpfail;
                self.stencil_dppass = dppass;
                self.stencil_sfail_back = sfail;
                self.stencil_dpfail_back = dpfail;
                self.stencil_dppass_back = dppass;
            }
            _ => {
                set_err!(self, GL_INVALID_ENUM);
            }
        }
    }

    pub fn gl_clear_stencil(&mut self, s: GLint) {
        self.clear_stencil = s;
    }

    pub fn gl_stencil_mask(&mut self, mask: GLuint) {
        self.stencil_writemask = mask;
        self.stencil_writemask_back = mask;
    }

    pub fn gl_stencil_mask_separate(&mut self, face: GLenum, mask: GLuint) {
        match face {
            GL_FRONT => {
                self.stencil_writemask = mask;
            }
            GL_BACK => {
                self.stencil_writemask_back = mask;
            }
            GL_FRONT_AND_BACK => {
                self.stencil_writemask = mask;
                self.stencil_writemask_back = mask;
            }
            _ => {
                set_err!(self, GL_INVALID_ENUM);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Shader/Program functions
    // -----------------------------------------------------------------------

    pub fn pgl_create_program(
        &mut self,
        vertex_shader: VertFunc,
        fragment_shader: FragFunc,
        n: GLsizei,
        interpolation: &[GLenum],
        fragdepth_or_discard: bool,
    ) -> GLuint {
        let mut prog = GlProgram::default();
        prog.vertex_shader = vertex_shader;
        prog.fragment_shader = fragment_shader;
        prog.vs_output_size = n;
        prog.fragdepth_or_discard = fragdepth_or_discard;

        let interp_count = (n as usize).min(GL_MAX_VERTEX_OUTPUT_COMPONENTS);
        for i in 0..interp_count {
            if i < interpolation.len() {
                prog.interpolation[i] = interpolation[i];
            } else {
                prog.interpolation[i] = PGL_SMOOTH;
            }
        }

        // Find a deleted slot or push new
        let mut found = None;
        for (i, p) in self.programs.iter().enumerate() {
            if i != 0 && p.deleted {
                found = Some(i);
                break;
            }
        }
        let id = if let Some(idx) = found {
            self.programs[idx] = prog;
            idx as GLuint
        } else {
            let idx = self.programs.len();
            self.programs.push(prog);
            idx as GLuint
        };
        id
    }

    pub fn gl_delete_program(&mut self, program: GLuint) {
        let idx = program as usize;
        if idx == 0 || idx >= self.programs.len() {
            return;
        }
        self.programs[idx].deleted = true;
        if self.cur_program == program {
            self.cur_program = 0;
        }
    }

    pub fn gl_use_program(&mut self, program: GLuint) {
        let idx = program as usize;
        if idx >= self.programs.len() || self.programs[idx].deleted {
            set_err!(self, GL_INVALID_OPERATION);
            return;
        }
        self.cur_program = program;
        self.vs_output.size = self.programs[idx].vs_output_size;
        self.vs_output.interpolation = self.programs[idx].interpolation.as_ptr();
        self.fragdepth_or_discard = self.programs[idx].fragdepth_or_discard;
    }

    pub fn pgl_set_uniform(&mut self, uniform: *mut c_void) {
        let idx = self.cur_program as usize;
        if idx < self.programs.len() {
            self.programs[idx].uniform = uniform;
        }
    }

    pub fn pgl_set_program_uniform(&mut self, program: GLuint, uniform: *mut c_void) {
        let idx = program as usize;
        if idx < self.programs.len() && !self.programs[idx].deleted {
            self.programs[idx].uniform = uniform;
        }
    }

    // -----------------------------------------------------------------------
    // Framebuffer read
    // -----------------------------------------------------------------------

    /// Read pixels from the back buffer.
    ///
    /// Only GL_RGBA / GL_UNSIGNED_BYTE is fully supported. Other format
    /// combinations will set GL_INVALID_ENUM.
    pub fn gl_read_pixels(
        &self,
        x: GLint,
        y: GLint,
        width: GLsizei,
        height: GLsizei,
        format: GLenum,
        type_: GLenum,
        data: &mut [u8],
    ) {
        if width <= 0 || height <= 0 {
            return;
        }
        // We only support reading RGBA / UNSIGNED_BYTE from the back buffer
        let components: usize = match format {
            GL_RGBA | GL_BGRA => 4,
            GL_RGB | GL_BGR => 3,
            GL_RED => 1,
            GL_RG => 2,
            _ => return,
        };

        let bw = self.back_buffer.w as usize;
        let bh = self.back_buffer.h as usize;

        let mut dst_off = 0usize;
        for row in 0..height as usize {
            let src_y = y as usize + row;
            if src_y >= bh {
                break;
            }
            for col in 0..width as usize {
                let src_x = x as usize + col;
                if src_x >= bw {
                    break;
                }
                let px_idx = src_y * bw + src_x;
                let pixel = read_pixel(&self.back_buffer.buf, px_idx);

                let r = (pixel & 0xFF) as u8;
                let g = ((pixel >> 8) & 0xFF) as u8;
                let b = ((pixel >> 16) & 0xFF) as u8;
                let a = ((pixel >> 24) & 0xFF) as u8;

                if dst_off + components > data.len() {
                    return;
                }
                match format {
                    GL_RGBA => {
                        data[dst_off] = r;
                        data[dst_off + 1] = g;
                        data[dst_off + 2] = b;
                        data[dst_off + 3] = a;
                    }
                    GL_BGRA => {
                        data[dst_off] = b;
                        data[dst_off + 1] = g;
                        data[dst_off + 2] = r;
                        data[dst_off + 3] = a;
                    }
                    GL_RGB => {
                        data[dst_off] = r;
                        data[dst_off + 1] = g;
                        data[dst_off + 2] = b;
                    }
                    GL_BGR => {
                        data[dst_off] = b;
                        data[dst_off + 1] = g;
                        data[dst_off + 2] = r;
                    }
                    GL_RG => {
                        data[dst_off] = r;
                        data[dst_off + 1] = g;
                    }
                    GL_RED => {
                        data[dst_off] = r;
                    }
                    _ => {}
                }
                dst_off += components;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Set a capability on or off.
    fn set_cap(&mut self, cap: GLenum, enabled: bool) {
        match cap {
            GL_CULL_FACE => self.cull_face = enabled,
            GL_DEPTH_TEST => self.depth_test = enabled,
            GL_DEPTH_CLAMP => self.depth_clamp = enabled,
            GL_LINE_SMOOTH => self.line_smooth = enabled,
            GL_BLEND => self.blend = enabled,
            GL_COLOR_LOGIC_OP => self.logic_ops = enabled,
            GL_POLYGON_OFFSET_POINT => self.poly_offset_pt = enabled,
            GL_POLYGON_OFFSET_LINE => self.poly_offset_line = enabled,
            GL_POLYGON_OFFSET_FILL => self.poly_offset_fill = enabled,
            GL_SCISSOR_TEST => self.scissor_test = enabled,
            GL_STENCIL_TEST => self.stencil_test = enabled,
            _ => {
                set_err!(self, GL_INVALID_ENUM);
            }
        }
    }

    /// Validate that a draw mode is a recognized primitive type.
    fn validate_draw_mode(&mut self, mode: GLenum) -> bool {
        match mode {
            GL_POINTS
            | GL_LINES
            | GL_LINE_STRIP
            | GL_LINE_LOOP
            | GL_TRIANGLES
            | GL_TRIANGLE_STRIP
            | GL_TRIANGLE_FAN
            | GL_LINE_STRIP_ADJACENCY
            | GL_LINES_ADJACENCY
            | GL_TRIANGLES_ADJACENCY
            | GL_TRIANGLE_STRIP_ADJACENCY => true,
            _ => {
                set_err!(self, GL_INVALID_ENUM);
                false
            }
        }
    }

    /// Prepare context state before a draw call.
    fn prepare_draw(&mut self) {
        let prog_idx = self.cur_program as usize;
        if prog_idx < self.programs.len() {
            self.vs_output.size = self.programs[prog_idx].vs_output_size;
            self.vs_output.interpolation = self.programs[prog_idx].interpolation.as_ptr();
            self.fragdepth_or_discard = self.programs[prog_idx].fragdepth_or_discard;
        }
    }

}

