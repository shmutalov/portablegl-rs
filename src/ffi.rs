//! C FFI wrapper module for the PortableGL Rust library.
//!
//! This module provides `#[no_mangle] pub unsafe extern "C" fn` wrappers for
//! every C API function in the original PortableGL library, making the Rust
//! implementation a drop-in replacement when compiled as a C-compatible library.
//!
//! A global mutable context pointer mirrors the C library's `static glContext* c`.
//! Call `init_glContext` / `set_glContext` before using any other function.

#![allow(
    non_snake_case,
    non_upper_case_globals,
    unused_variables,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use crate::gl_context::*;
use crate::gl_types::*;
use crate::math::*;
use core::ffi::c_void;

// ---------------------------------------------------------------------------
// Global context (mirrors C library's `static glContext* c`)
// ---------------------------------------------------------------------------

static mut CONTEXT: *mut GlContext = core::ptr::null_mut();

/// Helper: return `&mut *CONTEXT`. Caller must ensure CONTEXT is non-null.
macro_rules! ctx {
    () => {
        &mut *CONTEXT
    };
}

// ---------------------------------------------------------------------------
// Context management
// ---------------------------------------------------------------------------

/// Allocate and initialize a new GlContext.
///
/// On success the context pointer is written to `*context`, the back-buffer
/// pointer is written to `*back_buffer`, and GL_TRUE is returned.
#[no_mangle]
pub unsafe extern "C" fn init_glContext(
    context: *mut *mut GlContext,
    back_buffer: *mut *mut u32,
    width: GLsizei,
    height: GLsizei,
) -> GLboolean {
    if context.is_null() {
        return GL_FALSE;
    }

    let mut ctx = Box::new(GlContext::new());
    let _pixels = ctx.init(width, height);

    // The back buffer lives inside ctx.back_buffer.buf.
    // Give the caller a pointer into it (as *mut u32).
    if !back_buffer.is_null() {
        *back_buffer = ctx.back_buffer.buf.as_mut_ptr() as *mut u32;
    }

    let raw = Box::into_raw(ctx);
    *context = raw;

    // Also set as the active context
    CONTEXT = raw;

    GL_TRUE
}

/// Free a previously allocated GlContext.
#[no_mangle]
pub unsafe extern "C" fn free_glContext(context: *mut GlContext) {
    if !context.is_null() {
        if CONTEXT == context {
            CONTEXT = core::ptr::null_mut();
        }
        drop(Box::from_raw(context));
    }
}

/// Set the active global context (like the C library's `set_glContext`).
#[no_mangle]
pub unsafe extern "C" fn set_glContext(context: *mut GlContext) {
    CONTEXT = context;
}

/// Resize the framebuffer of the active context.
#[no_mangle]
pub unsafe extern "C" fn pglResizeFramebuffer(width: GLsizei, height: GLsizei) -> GLboolean {
    if ctx!().resize_framebuffer(width, height) {
        GL_TRUE
    } else {
        GL_FALSE
    }
}

// ---------------------------------------------------------------------------
// State queries
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn glGetString(name: GLenum) -> *mut GLubyte {
    let s = ctx!().gl_get_string(name);
    s.as_ptr() as *mut GLubyte
}

#[no_mangle]
pub unsafe extern "C" fn glGetError() -> GLenum {
    ctx!().gl_get_error()
}

#[no_mangle]
pub unsafe extern "C" fn glGetBooleanv(pname: GLenum, data: *mut GLboolean) {
    if data.is_null() {
        return;
    }
    if let Some(val) = ctx!().gl_get_booleanv(pname) {
        *data = if val { GL_TRUE } else { GL_FALSE };
    }
}

#[no_mangle]
pub unsafe extern "C" fn glGetFloatv(pname: GLenum, data: *mut GLfloat) {
    if data.is_null() {
        return;
    }
    if let Some(vals) = ctx!().gl_get_floatv(pname) {
        for (i, v) in vals.iter().enumerate() {
            *data.add(i) = *v;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn glGetIntegerv(pname: GLenum, data: *mut GLint) {
    if data.is_null() {
        return;
    }
    if let Some(vals) = ctx!().gl_get_integerv(pname) {
        for (i, v) in vals.iter().enumerate() {
            *data.add(i) = *v;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn glIsEnabled(cap: GLenum) -> GLboolean {
    if ctx!().gl_is_enabled(cap) {
        GL_TRUE
    } else {
        GL_FALSE
    }
}

#[no_mangle]
pub unsafe extern "C" fn glIsProgram(program: GLuint) -> GLboolean {
    if ctx!().gl_is_program(program) {
        GL_TRUE
    } else {
        GL_FALSE
    }
}

// ---------------------------------------------------------------------------
// State setting
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn glColorMask(
    red: GLboolean,
    green: GLboolean,
    blue: GLboolean,
    alpha: GLboolean,
) {
    ctx!().gl_color_mask(red != 0, green != 0, blue != 0, alpha != 0);
}

#[no_mangle]
pub unsafe extern "C" fn glClearColor(red: GLfloat, green: GLfloat, blue: GLfloat, alpha: GLfloat) {
    ctx!().gl_clear_color(red, green, blue, alpha);
}

#[no_mangle]
pub unsafe extern "C" fn glClearDepthf(depth: GLfloat) {
    ctx!().gl_clear_depth(depth as f64);
}

#[no_mangle]
pub unsafe extern "C" fn glClearDepth(depth: GLdouble) {
    ctx!().gl_clear_depth(depth);
}

#[no_mangle]
pub unsafe extern "C" fn glDepthFunc(func: GLenum) {
    ctx!().gl_depth_func(func);
}

#[no_mangle]
pub unsafe extern "C" fn glDepthRangef(nearVal: GLfloat, farVal: GLfloat) {
    ctx!().gl_depth_range(nearVal as f64, farVal as f64);
}

#[no_mangle]
pub unsafe extern "C" fn glDepthRange(nearVal: GLdouble, farVal: GLdouble) {
    ctx!().gl_depth_range(nearVal, farVal);
}

#[no_mangle]
pub unsafe extern "C" fn glDepthMask(flag: GLboolean) {
    ctx!().gl_depth_mask(flag != 0);
}

#[no_mangle]
pub unsafe extern "C" fn glBlendFunc(sfactor: GLenum, dfactor: GLenum) {
    ctx!().gl_blend_func(sfactor, dfactor);
}

#[no_mangle]
pub unsafe extern "C" fn glBlendEquation(mode: GLenum) {
    ctx!().gl_blend_equation(mode);
}

#[no_mangle]
pub unsafe extern "C" fn glBlendFuncSeparate(
    srcRGB: GLenum,
    dstRGB: GLenum,
    srcAlpha: GLenum,
    dstAlpha: GLenum,
) {
    ctx!().gl_blend_func_separate(srcRGB, dstRGB, srcAlpha, dstAlpha);
}

#[no_mangle]
pub unsafe extern "C" fn glBlendEquationSeparate(modeRGB: GLenum, modeAlpha: GLenum) {
    ctx!().gl_blend_equation_separate(modeRGB, modeAlpha);
}

#[no_mangle]
pub unsafe extern "C" fn glBlendColor(
    red: GLfloat,
    green: GLfloat,
    blue: GLfloat,
    alpha: GLfloat,
) {
    ctx!().gl_blend_color(red, green, blue, alpha);
}

#[no_mangle]
pub unsafe extern "C" fn glClear(mask: GLbitfield) {
    ctx!().gl_clear(mask);
}

#[no_mangle]
pub unsafe extern "C" fn glProvokingVertex(provokeMode: GLenum) {
    ctx!().gl_provoking_vertex(provokeMode);
}

#[no_mangle]
pub unsafe extern "C" fn glEnable(cap: GLenum) {
    ctx!().gl_enable(cap);
}

#[no_mangle]
pub unsafe extern "C" fn glDisable(cap: GLenum) {
    ctx!().gl_disable(cap);
}

#[no_mangle]
pub unsafe extern "C" fn glCullFace(mode: GLenum) {
    ctx!().gl_cull_face(mode);
}

#[no_mangle]
pub unsafe extern "C" fn glFrontFace(mode: GLenum) {
    ctx!().gl_front_face(mode);
}

#[no_mangle]
pub unsafe extern "C" fn glPolygonMode(face: GLenum, mode: GLenum) {
    ctx!().gl_polygon_mode(face, mode);
}

#[no_mangle]
pub unsafe extern "C" fn glPointSize(size: GLfloat) {
    ctx!().gl_point_size(size);
}

#[no_mangle]
pub unsafe extern "C" fn glPointParameteri(pname: GLenum, param: GLint) {
    ctx!().gl_point_parameteri(pname, param);
}

#[no_mangle]
pub unsafe extern "C" fn glLineWidth(width: GLfloat) {
    ctx!().gl_line_width(width);
}

#[no_mangle]
pub unsafe extern "C" fn glLogicOp(opcode: GLenum) {
    ctx!().gl_logic_op(opcode);
}

#[no_mangle]
pub unsafe extern "C" fn glPolygonOffset(factor: GLfloat, units: GLfloat) {
    ctx!().gl_polygon_offset(factor, units);
}

#[no_mangle]
pub unsafe extern "C" fn glScissor(x: GLint, y: GLint, width: GLsizei, height: GLsizei) {
    ctx!().gl_scissor(x, y, width, height);
}

#[no_mangle]
pub unsafe extern "C" fn glViewport(x: GLint, y: GLint, width: GLsizei, height: GLsizei) {
    ctx!().gl_viewport(x, y, width, height);
}

// ---------------------------------------------------------------------------
// Stencil
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn glStencilFunc(func: GLenum, ref_: GLint, mask: GLuint) {
    ctx!().gl_stencil_func(func, ref_, mask);
}

#[no_mangle]
pub unsafe extern "C" fn glStencilFuncSeparate(
    face: GLenum,
    func: GLenum,
    ref_: GLint,
    mask: GLuint,
) {
    ctx!().gl_stencil_func_separate(face, func, ref_, mask);
}

#[no_mangle]
pub unsafe extern "C" fn glStencilOp(sfail: GLenum, dpfail: GLenum, dppass: GLenum) {
    ctx!().gl_stencil_op(sfail, dpfail, dppass);
}

#[no_mangle]
pub unsafe extern "C" fn glStencilOpSeparate(
    face: GLenum,
    sfail: GLenum,
    dpfail: GLenum,
    dppass: GLenum,
) {
    ctx!().gl_stencil_op_separate(face, sfail, dpfail, dppass);
}

#[no_mangle]
pub unsafe extern "C" fn glClearStencil(s: GLint) {
    ctx!().gl_clear_stencil(s);
}

#[no_mangle]
pub unsafe extern "C" fn glStencilMask(mask: GLuint) {
    ctx!().gl_stencil_mask(mask);
}

#[no_mangle]
pub unsafe extern "C" fn glStencilMaskSeparate(face: GLenum, mask: GLuint) {
    ctx!().gl_stencil_mask_separate(face, mask);
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn glGenTextures(n: GLsizei, textures: *mut GLuint) {
    if textures.is_null() || n <= 0 {
        return;
    }
    let ids = ctx!().gl_gen_textures(n);
    for (i, id) in ids.iter().enumerate() {
        *textures.add(i) = *id;
    }
}

#[no_mangle]
pub unsafe extern "C" fn glDeleteTextures(n: GLsizei, textures: *const GLuint) {
    if textures.is_null() || n <= 0 {
        return;
    }
    let slice = core::slice::from_raw_parts(textures, n as usize);
    ctx!().gl_delete_textures(slice);
}

#[no_mangle]
pub unsafe extern "C" fn glBindTexture(target: GLenum, texture: GLuint) {
    let _ = ctx!().gl_bind_texture(target, texture);
}

#[no_mangle]
pub unsafe extern "C" fn glTexParameteri(target: GLenum, pname: GLenum, param: GLint) {
    ctx!().gl_tex_parameteri(target, pname, param);
}

#[no_mangle]
pub unsafe extern "C" fn glPixelStorei(pname: GLenum, param: GLint) {
    ctx!().gl_pixel_storei(pname, param);
}

#[no_mangle]
pub unsafe extern "C" fn glTexImage1D(
    target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    border: GLint,
    format: GLenum,
    type_: GLenum,
    data: *const c_void,
) {
    let opt_data = if data.is_null() {
        None
    } else {
        // Compute a conservative upper bound for the slice size.
        // The actual component count is handled inside gl_tex_image_1d.
        let size = (width as usize) * 4;
        Some(core::slice::from_raw_parts(data as *const u8, size))
    };
    ctx!().gl_tex_image_1d(target, level, internalformat, width, border, format, type_, opt_data);
}

#[no_mangle]
pub unsafe extern "C" fn glTexImage2D(
    target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    height: GLsizei,
    border: GLint,
    format: GLenum,
    type_: GLenum,
    data: *const c_void,
) {
    let opt_data = if data.is_null() {
        None
    } else {
        let size = (width as usize) * (height as usize) * 4;
        Some(core::slice::from_raw_parts(data as *const u8, size))
    };
    ctx!().gl_tex_image_2d(
        target,
        level,
        internalformat,
        width,
        height,
        border,
        format,
        type_,
        opt_data,
    );
}

#[no_mangle]
pub unsafe extern "C" fn glTexImage3D(
    target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    height: GLsizei,
    depth: GLsizei,
    border: GLint,
    format: GLenum,
    type_: GLenum,
    data: *const c_void,
) {
    let opt_data = if data.is_null() {
        None
    } else {
        let size = (width as usize) * (height as usize) * (depth as usize) * 4;
        Some(core::slice::from_raw_parts(data as *const u8, size))
    };
    ctx!().gl_tex_image_3d(
        target,
        level,
        internalformat,
        width,
        height,
        depth,
        border,
        format,
        type_,
        opt_data,
    );
}

#[no_mangle]
pub unsafe extern "C" fn glTexSubImage1D(
    target: GLenum,
    level: GLint,
    xoffset: GLint,
    width: GLsizei,
    format: GLenum,
    type_: GLenum,
    data: *const c_void,
) {
    if data.is_null() {
        return;
    }
    let size = (width as usize) * 4;
    let slice = core::slice::from_raw_parts(data as *const u8, size);
    ctx!().gl_tex_sub_image_1d(target, level, xoffset, width, format, type_, slice);
}

#[no_mangle]
pub unsafe extern "C" fn glTexSubImage2D(
    target: GLenum,
    level: GLint,
    xoffset: GLint,
    yoffset: GLint,
    width: GLsizei,
    height: GLsizei,
    format: GLenum,
    type_: GLenum,
    data: *const c_void,
) {
    if data.is_null() {
        return;
    }
    let size = (width as usize) * (height as usize) * 4;
    let slice = core::slice::from_raw_parts(data as *const u8, size);
    ctx!().gl_tex_sub_image_2d(
        target, level, xoffset, yoffset, width, height, format, type_, slice,
    );
}

#[no_mangle]
pub unsafe extern "C" fn glTexSubImage3D(
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
    data: *const c_void,
) {
    if data.is_null() {
        return;
    }
    let size = (width as usize) * (height as usize) * (depth as usize) * 4;
    let slice = core::slice::from_raw_parts(data as *const u8, size);
    ctx!().gl_tex_sub_image_3d(
        target, level, xoffset, yoffset, zoffset, width, height, depth, format, type_, slice,
    );
}

// ---------------------------------------------------------------------------
// VAOs
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn glGenVertexArrays(n: GLsizei, arrays: *mut GLuint) {
    if arrays.is_null() || n <= 0 {
        return;
    }
    let ids = ctx!().gl_gen_vertex_arrays(n);
    for (i, id) in ids.iter().enumerate() {
        *arrays.add(i) = *id;
    }
}

#[no_mangle]
pub unsafe extern "C" fn glDeleteVertexArrays(n: GLsizei, arrays: *const GLuint) {
    if arrays.is_null() || n <= 0 {
        return;
    }
    let slice = core::slice::from_raw_parts(arrays, n as usize);
    ctx!().gl_delete_vertex_arrays(slice);
}

#[no_mangle]
pub unsafe extern "C" fn glBindVertexArray(array: GLuint) {
    let _ = ctx!().gl_bind_vertex_array(array);
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn glGenBuffers(n: GLsizei, buffers: *mut GLuint) {
    if buffers.is_null() || n <= 0 {
        return;
    }
    let ids = ctx!().gl_gen_buffers(n);
    for (i, id) in ids.iter().enumerate() {
        *buffers.add(i) = *id;
    }
}

#[no_mangle]
pub unsafe extern "C" fn glDeleteBuffers(n: GLsizei, buffers: *const GLuint) {
    if buffers.is_null() || n <= 0 {
        return;
    }
    let slice = core::slice::from_raw_parts(buffers, n as usize);
    ctx!().gl_delete_buffers(slice);
}

#[no_mangle]
pub unsafe extern "C" fn glBindBuffer(target: GLenum, buffer: GLuint) {
    let _ = ctx!().gl_bind_buffer(target, buffer);
}

#[no_mangle]
pub unsafe extern "C" fn glBufferData(
    target: GLenum,
    size: GLsizeiptr,
    data: *const c_void,
    usage: GLenum,
) {
    if data.is_null() || size <= 0 {
        // Even with null data, create the buffer storage
        let empty: &[u8] = &[];
        let _ = ctx!().gl_buffer_data(target, empty, usage);
        return;
    }
    let slice = core::slice::from_raw_parts(data as *const u8, size as usize);
    let _ = ctx!().gl_buffer_data(target, slice, usage);
}

#[no_mangle]
pub unsafe extern "C" fn glBufferSubData(
    target: GLenum,
    offset: GLintptr,
    size: GLsizeiptr,
    data: *const c_void,
) {
    if data.is_null() || size <= 0 {
        return;
    }
    let slice = core::slice::from_raw_parts(data as *const u8, size as usize);
    let _ = ctx!().gl_buffer_sub_data(target, offset, slice);
}

#[no_mangle]
pub unsafe extern "C" fn glMapBuffer(target: GLenum, access: GLenum) -> *mut c_void {
    ctx!().gl_map_buffer(target, access)
}

// ---------------------------------------------------------------------------
// Vertex attribs
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn glVertexAttribPointer(
    index: GLuint,
    size: GLint,
    type_: GLenum,
    normalized: GLboolean,
    stride: GLsizei,
    pointer: *const c_void,
) {
    ctx!().gl_vertex_attrib_pointer(
        index,
        size,
        type_,
        normalized != 0,
        stride,
        pointer as GLsizeiptr,
    );
}

#[no_mangle]
pub unsafe extern "C" fn glVertexAttribDivisor(index: GLuint, divisor: GLuint) {
    ctx!().gl_vertex_attrib_divisor(index, divisor);
}

#[no_mangle]
pub unsafe extern "C" fn glEnableVertexAttribArray(index: GLuint) {
    ctx!().gl_enable_vertex_attrib_array(index);
}

#[no_mangle]
pub unsafe extern "C" fn glDisableVertexAttribArray(index: GLuint) {
    ctx!().gl_disable_vertex_attrib_array(index);
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn glDrawArrays(mode: GLenum, first: GLint, count: GLsizei) {
    ctx!().gl_draw_arrays(mode, first, count);
}

#[no_mangle]
pub unsafe extern "C" fn glDrawElements(
    mode: GLenum,
    count: GLsizei,
    type_: GLenum,
    indices: *const c_void,
) {
    ctx!().gl_draw_elements(mode, count, type_, indices as usize);
}

#[no_mangle]
pub unsafe extern "C" fn glDrawArraysInstanced(
    mode: GLenum,
    first: GLint,
    count: GLsizei,
    primcount: GLsizei,
) {
    ctx!().gl_draw_arrays_instanced(mode, first, count, primcount);
}

#[no_mangle]
pub unsafe extern "C" fn glDrawElementsInstanced(
    mode: GLenum,
    count: GLsizei,
    type_: GLenum,
    indices: *const c_void,
    primcount: GLsizei,
) {
    ctx!().gl_draw_elements_instanced(mode, count, type_, indices as usize, primcount);
}

#[no_mangle]
pub unsafe extern "C" fn glDrawArraysInstancedBaseInstance(
    mode: GLenum,
    first: GLint,
    count: GLsizei,
    primcount: GLsizei,
    baseinstance: GLuint,
) {
    ctx!().gl_draw_arrays_instanced_base_instance(mode, first, count, primcount, baseinstance);
}

#[no_mangle]
pub unsafe extern "C" fn glDrawElementsInstancedBaseInstance(
    mode: GLenum,
    count: GLsizei,
    type_: GLenum,
    indices: *const c_void,
    primcount: GLsizei,
    baseinstance: GLuint,
) {
    ctx!().gl_draw_elements_instanced_base_instance(
        mode,
        count,
        type_,
        indices as usize,
        primcount,
        baseinstance,
    );
}

// ---------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn pglCreateProgram(
    vertex_shader: VertFunc,
    fragment_shader: FragFunc,
    n: GLsizei,
    interpolation: *mut GLenum,
    fragdepth_or_discard: GLboolean,
) -> GLuint {
    let interp_slice = if interpolation.is_null() || n <= 0 {
        &[]
    } else {
        core::slice::from_raw_parts(interpolation, n as usize)
    };
    ctx!().pgl_create_program(
        vertex_shader,
        fragment_shader,
        n,
        interp_slice,
        fragdepth_or_discard != 0,
    )
}

#[no_mangle]
pub unsafe extern "C" fn glDeleteProgram(program: GLuint) {
    ctx!().gl_delete_program(program);
}

#[no_mangle]
pub unsafe extern "C" fn glUseProgram(program: GLuint) {
    ctx!().gl_use_program(program);
}

#[no_mangle]
pub unsafe extern "C" fn pglSetUniform(uniform: *mut c_void) {
    ctx!().pgl_set_uniform(uniform);
}

// ---------------------------------------------------------------------------
// PGL extensions
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn pglClearScreen() {
    ctx!().pgl_clear_screen();
}

#[no_mangle]
pub unsafe extern "C" fn pglSetInterp(n: GLsizei, interpolation: *mut GLenum) {
    let interp_slice = if interpolation.is_null() || n <= 0 {
        &[]
    } else {
        core::slice::from_raw_parts(interpolation, n as usize)
    };
    ctx!().pgl_set_interp(n, interp_slice);
}

#[no_mangle]
pub unsafe extern "C" fn pglCreateFragProgram(
    fragment_shader: FragFunc,
    fragdepth_or_discard: GLboolean,
) -> GLuint {
    // pgl_create_frag_program expects interp slice; for the C API the interps
    // are set later via pglSetInterp, so we pass an empty slice with n=0.
    ctx!().pgl_create_frag_program(fragment_shader, 0, &[], fragdepth_or_discard != 0)
}

#[no_mangle]
pub unsafe extern "C" fn pglDrawFrame() {
    ctx!().pgl_draw_frame();
}

#[no_mangle]
pub unsafe extern "C" fn pglDrawFrame2(frag_shader: FragFunc, uniforms: *mut c_void) {
    ctx!().pgl_draw_frame2(frag_shader, uniforms);
}

#[no_mangle]
pub unsafe extern "C" fn pglBufferData(
    target: GLenum,
    size: GLsizei,
    data: *const c_void,
    usage: GLenum,
) {
    ctx!().pgl_buffer_data(target, size as GLsizeiptr, data as *mut u8, false);
}

#[no_mangle]
pub unsafe extern "C" fn pglTexImage1D(
    target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    border: GLint,
    format: GLenum,
    type_: GLenum,
    data: *const c_void,
) {
    ctx!().pgl_tex_image_1d(
        target,
        level,
        internalformat,
        width,
        border,
        format,
        type_,
        data as *mut u8,
    );
}

#[no_mangle]
pub unsafe extern "C" fn pglTexImage2D(
    target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    height: GLsizei,
    border: GLint,
    format: GLenum,
    type_: GLenum,
    data: *const c_void,
) {
    ctx!().pgl_tex_image_2d(
        target,
        level,
        internalformat,
        width,
        height,
        border,
        format,
        type_,
        data as *mut u8,
    );
}

#[no_mangle]
pub unsafe extern "C" fn pglTexImage3D(
    target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    height: GLsizei,
    depth: GLsizei,
    border: GLint,
    format: GLenum,
    type_: GLenum,
    data: *const c_void,
) {
    ctx!().pgl_tex_image_3d(
        target,
        level,
        internalformat,
        width,
        height,
        depth,
        border,
        format,
        type_,
        data as *mut u8,
    );
}

#[no_mangle]
pub unsafe extern "C" fn pglGetBufferData(buffer: GLuint, data: *mut *mut c_void) {
    if data.is_null() {
        return;
    }
    *data = ctx!().pgl_get_buffer_data(buffer) as *mut c_void;
}

#[no_mangle]
pub unsafe extern "C" fn pglGetTextureData(texture: GLuint, data: *mut *mut c_void) {
    if data.is_null() {
        return;
    }
    *data = ctx!().pgl_get_texture_data(texture) as *mut c_void;
}

#[no_mangle]
pub unsafe extern "C" fn pglGetBackBuffer() -> *mut c_void {
    ctx!().back_buffer.buf.as_mut_ptr() as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn pglSetBackBuffer(backbuf: *mut c_void, width: GLsizei, height: GLsizei) {
    ctx!().pgl_set_back_buffer(backbuf as *mut u8, width, height, false);
}

// ---------------------------------------------------------------------------
// Direct drawing primitives
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn put_pixel(color: Color, x: i32, y: i32) {
    ctx!().put_pixel(color, x, y);
}

#[no_mangle]
pub unsafe extern "C" fn put_line(the_color: Color, x1: f32, y1: f32, x2: f32, y2: f32) {
    ctx!().put_line(the_color, x1, y1, x2, y2);
}

#[no_mangle]
pub unsafe extern "C" fn put_triangle(c1: Color, c2: Color, c3: Color, p1: Vec2, p2: Vec2, p3: Vec2) {
    ctx!().put_triangle(c1, c2, c3, p1, p2, p3);
}

// ---------------------------------------------------------------------------
// Read pixels
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn glReadPixels(
    x: GLint,
    y: GLint,
    width: GLsizei,
    height: GLsizei,
    format: GLenum,
    type_: GLenum,
    data: *mut c_void,
) {
    if data.is_null() || width <= 0 || height <= 0 {
        return;
    }
    let size = (width as usize) * (height as usize) * 4;
    let slice = core::slice::from_raw_parts_mut(data as *mut u8, size);
    ctx!().gl_read_pixels(x, y, width, height, format, type_, slice);
}
