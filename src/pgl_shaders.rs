//! Standard shaders and uniform types for common rendering use cases.
//!
//! These provide ready-made vertex and fragment shaders covering identity transforms,
//! smooth shading, directional/point lighting, texture replacement/modulation, and
//! texture-rectangle sampling.  Users with standard rendering needs can use these
//! directly instead of writing their own shader functions.

#![allow(
    non_upper_case_globals,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]

use crate::gl_context::GlContext;
use crate::gl_types::*;
use crate::math::*;

use core::ffi::c_void;

// ---------------------------------------------------------------------------
// Standard uniform struct
// ---------------------------------------------------------------------------

/// Standard uniform block matching the C `pgl_uniforms` struct.
///
/// Covers the most common uniform needs: MVP/MV/P matrices, a normal matrix,
/// a color, a texture unit, a light position, and a context pointer (needed
/// because the Rust port has no global GL context).
#[repr(C)]
pub struct PglUniforms {
    pub mvp_mat: Mat4,
    pub mv_mat: Mat4,
    pub p_mat: Mat4,
    pub normal_mat: Mat3,
    pub color: Vec4,
    pub tex0: GLuint,
    pub light_pos: Vec3,
    /// Context pointer for texture sampling in shaders.
    /// The C version uses a global context; in Rust we pass it through uniforms.
    pub ctx: *const GlContext,
}

impl Default for PglUniforms {
    fn default() -> Self {
        Self {
            mvp_mat: Mat4::identity(),
            mv_mat: Mat4::identity(),
            p_mat: Mat4::identity(),
            normal_mat: Mat3::identity(),
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            tex0: 0,
            light_pos: Vec3::new(0.0, 0.0, 0.0),
            ctx: core::ptr::null(),
        }
    }
}

// ---------------------------------------------------------------------------
// Standard attribute indices
// ---------------------------------------------------------------------------

pub const PGL_ATTR_VERT: GLuint = 0;
pub const PGL_ATTR_COLOR: GLuint = 1;
pub const PGL_ATTR_NORMAL: GLuint = 2;
pub const PGL_ATTR_TEXCOORD0: GLuint = 3;
pub const PGL_ATTR_TEXCOORD1: GLuint = 4;

// ---------------------------------------------------------------------------
// Standard shader indices (for use with pgl_init_std_shaders)
// ---------------------------------------------------------------------------

pub const PGL_SHADER_IDENTITY: usize = 0;
pub const PGL_SHADER_FLAT: usize = 1;
pub const PGL_SHADER_SHADED: usize = 2;
pub const PGL_SHADER_DFLT_LIGHT: usize = 3;
pub const PGL_SHADER_POINT_LIGHT_DIFF: usize = 4;
pub const PGL_SHADER_TEX_REPLACE: usize = 5;
pub const PGL_SHADER_TEX_MODULATE: usize = 6;
pub const PGL_SHADER_TEX_POINT_LIGHT_DIFF: usize = 7;
pub const PGL_SHADER_TEX_RECT_REPLACE: usize = 8;
pub const PGL_NUM_SHADERS: usize = 9;

// ---------------------------------------------------------------------------
// Standard shaders
// ---------------------------------------------------------------------------

/// Identity vertex shader — passes vertex position straight through, no transform.
pub unsafe extern "C" fn pgl_identity_vs(
    _vs_output: *mut f32, vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins, _uniforms: *mut c_void,
) {
    (*builtins).gl_Position = *vertex_attribs.add(PGL_ATTR_VERT as usize);
}

/// Identity fragment shader — outputs `PglUniforms::color` as the fragment color.
pub unsafe extern "C" fn pgl_identity_fs(
    _fs_input: *mut f32, builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    (*builtins).gl_FragColor = (*(uniforms as *const PglUniforms)).color;
}

/// Flat vertex shader — transforms position by `uniforms` interpreted as a raw `Mat4`.
/// Produces no varyings.
pub unsafe extern "C" fn pgl_flat_vs(
    _vs_output: *mut f32, vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let mvp = *(uniforms as *const Mat4);
    (*builtins).gl_Position = mult_m4_v4(mvp, *vertex_attribs.add(PGL_ATTR_VERT as usize));
}

/// Smooth-shaded vertex shader — transforms by MVP, outputs per-vertex color.
pub unsafe extern "C" fn pgl_shaded_vs(
    vs_output: *mut f32, vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let color = *vertex_attribs.add(PGL_ATTR_COLOR as usize);
    *(vs_output as *mut Vec4) = color;
    let mvp = *(uniforms as *const Mat4);
    (*builtins).gl_Position = mult_m4_v4(mvp, *vertex_attribs.add(PGL_ATTR_VERT as usize));
}

/// Smooth-shaded fragment shader — reads interpolated color from `fs_input`.
pub unsafe extern "C" fn pgl_shaded_fs(
    fs_input: *mut f32, builtins: *mut ShaderBuiltins, _uniforms: *mut c_void,
) {
    (*builtins).gl_FragColor = *(fs_input as *const Vec4);
}

/// Default directional-light vertex shader — lights with a fixed (0,0,1) direction.
pub unsafe extern "C" fn pgl_dflt_light_vs(
    vs_output: *mut f32, vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let u = &*(uniforms as *const PglUniforms);
    let normal_raw = *vertex_attribs.add(PGL_ATTR_NORMAL as usize);
    let n3 = Vec3::new(normal_raw.x, normal_raw.y, normal_raw.z);
    let norm = norm_v3(mult_m3_v3(u.normal_mat, n3));
    let light_dir = Vec3::new(0.0, 0.0, 1.0);
    let fdot = dot_v3s(norm, light_dir).max(0.0);
    let c = u.color;
    *(vs_output as *mut Vec4) = Vec4::new(c.x * fdot, c.y * fdot, c.z * fdot, c.w);
    (*builtins).gl_Position = mult_m4_v4(u.mvp_mat, *vertex_attribs.add(PGL_ATTR_VERT as usize));
}

/// Point-light diffuse vertex shader — computes diffuse lighting from `light_pos`.
pub unsafe extern "C" fn pgl_pnt_light_diff_vs(
    vs_output: *mut f32, vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let u = &*(uniforms as *const PglUniforms);
    let normal_raw = *vertex_attribs.add(PGL_ATTR_NORMAL as usize);
    let n3 = Vec3::new(normal_raw.x, normal_raw.y, normal_raw.z);
    let norm = norm_v3(mult_m3_v3(u.normal_mat, n3));
    let ec_pos = mult_m4_v4(u.mv_mat, *vertex_attribs.add(PGL_ATTR_VERT as usize));
    let ec_pos3 = ec_pos.to_vec3h();
    let light_dir = norm_v3(sub_v3s(u.light_pos, ec_pos3));
    let fdot = dot_v3s(norm, light_dir).max(0.0);
    let c = u.color;
    *(vs_output as *mut Vec4) = Vec4::new(c.x * fdot, c.y * fdot, c.z * fdot, c.w);
    (*builtins).gl_Position = mult_m4_v4(u.mvp_mat, *vertex_attribs.add(PGL_ATTR_VERT as usize));
}

/// Texture-replace vertex shader — transforms by MVP and passes through tex coords.
pub unsafe extern "C" fn pgl_tex_rplc_vs(
    vs_output: *mut f32, vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let u = &*(uniforms as *const PglUniforms);
    let tc = *vertex_attribs.add(PGL_ATTR_TEXCOORD0 as usize);
    *(vs_output as *mut Vec2) = Vec2::new(tc.x, tc.y);
    (*builtins).gl_Position = mult_m4_v4(u.mvp_mat, *vertex_attribs.add(PGL_ATTR_VERT as usize));
}

/// Texture-replace fragment shader — samples `tex0` and outputs the texel color.
pub unsafe extern "C" fn pgl_tex_rplc_fs(
    fs_input: *mut f32, builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let tex_coords = *(fs_input as *const Vec2);
    let u = &*(uniforms as *const PglUniforms);
    (*builtins).gl_FragColor = (*u.ctx).texture2d(u.tex0, tex_coords.x, tex_coords.y);
}

/// Texture-rectangle replace fragment shader — same as `pgl_tex_rplc_fs` but uses
/// `texture_rect` instead of `texture2d`.
pub unsafe extern "C" fn pgl_tex_rect_rplc_fs(
    fs_input: *mut f32, builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let tex_coords = *(fs_input as *const Vec2);
    let u = &*(uniforms as *const PglUniforms);
    (*builtins).gl_FragColor = (*u.ctx).texture_rect(u.tex0, tex_coords.x, tex_coords.y);
}

/// Texture-modulate fragment shader — multiplies texel color by `PglUniforms::color`.
pub unsafe extern "C" fn pgl_tex_modulate_fs(
    fs_input: *mut f32, builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let tex_coords = *(fs_input as *const Vec2);
    let u = &*(uniforms as *const PglUniforms);
    let tex_color = (*u.ctx).texture2d(u.tex0, tex_coords.x, tex_coords.y);
    (*builtins).gl_FragColor = Vec4::new(
        tex_color.x * u.color.x,
        tex_color.y * u.color.y,
        tex_color.z * u.color.z,
        tex_color.w * u.color.w,
    );
}

/// Texture + point-light diffuse vertex shader — combines lighting with texture coords.
pub unsafe extern "C" fn pgl_tex_pnt_light_diff_vs(
    vs_output: *mut f32, vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let u = &*(uniforms as *const PglUniforms);
    let normal_raw = *vertex_attribs.add(PGL_ATTR_NORMAL as usize);
    let n3 = Vec3::new(normal_raw.x, normal_raw.y, normal_raw.z);
    let norm = norm_v3(mult_m3_v3(u.normal_mat, n3));
    let ec_pos = mult_m4_v4(u.mv_mat, *vertex_attribs.add(PGL_ATTR_VERT as usize));
    let ec_pos3 = ec_pos.to_vec3h();
    let light_dir = norm_v3(sub_v3s(u.light_pos, ec_pos3));
    let fdot = dot_v3s(norm, light_dir).max(0.0);
    let c = u.color;
    *(vs_output as *mut Vec4) = Vec4::new(c.x * fdot, c.y * fdot, c.z * fdot, c.w);
    let tc = *vertex_attribs.add(PGL_ATTR_TEXCOORD0 as usize);
    *((vs_output as *mut Vec2).add(2)) = Vec2::new(tc.x, tc.y);
    (*builtins).gl_Position = mult_m4_v4(u.mvp_mat, *vertex_attribs.add(PGL_ATTR_VERT as usize));
}

/// Texture + point-light diffuse fragment shader — multiplies lighting color by texel.
pub unsafe extern "C" fn pgl_tex_pnt_light_diff_fs(
    fs_input: *mut f32, builtins: *mut ShaderBuiltins, uniforms: *mut c_void,
) {
    let u = &*(uniforms as *const PglUniforms);
    let light_color = *(fs_input as *const Vec4);
    let tex_coords = *((fs_input as *const Vec2).add(2));
    let tex_color = (*u.ctx).texture2d(u.tex0, tex_coords.x, tex_coords.y);
    (*builtins).gl_FragColor = Vec4::new(
        light_color.x * tex_color.x,
        light_color.y * tex_color.y,
        light_color.z * tex_color.z,
        light_color.w * tex_color.w,
    );
}

// ---------------------------------------------------------------------------
// Convenience: initialise all standard shader programs at once
// ---------------------------------------------------------------------------

/// Creates all standard shader programs and returns their program IDs as an array
/// indexed by `PGL_SHADER_*` constants.
pub fn pgl_init_std_shaders(ctx: &mut GlContext) -> [GLuint; PGL_NUM_SHADERS] {
    let mut programs = [0u32; PGL_NUM_SHADERS];

    let smooth4: [GLenum; 4] = [PGL_SMOOTH; 4];
    let smooth2: [GLenum; 2] = [PGL_SMOOTH; 2];
    let smooth4_2: [GLenum; 6] = [PGL_SMOOTH; 6];
    let empty: [GLenum; 0] = [];

    programs[PGL_SHADER_IDENTITY] = ctx.pgl_create_program(pgl_identity_vs, pgl_identity_fs, 0, &empty, false);
    programs[PGL_SHADER_FLAT] = ctx.pgl_create_program(pgl_flat_vs, pgl_identity_fs, 0, &empty, false);
    programs[PGL_SHADER_SHADED] = ctx.pgl_create_program(pgl_shaded_vs, pgl_shaded_fs, 4, &smooth4, false);
    programs[PGL_SHADER_DFLT_LIGHT] = ctx.pgl_create_program(pgl_dflt_light_vs, pgl_shaded_fs, 4, &smooth4, false);
    programs[PGL_SHADER_POINT_LIGHT_DIFF] = ctx.pgl_create_program(pgl_pnt_light_diff_vs, pgl_shaded_fs, 4, &smooth4, false);
    programs[PGL_SHADER_TEX_REPLACE] = ctx.pgl_create_program(pgl_tex_rplc_vs, pgl_tex_rplc_fs, 2, &smooth2, false);
    programs[PGL_SHADER_TEX_MODULATE] = ctx.pgl_create_program(pgl_tex_rplc_vs, pgl_tex_modulate_fs, 2, &smooth2, false);
    programs[PGL_SHADER_TEX_POINT_LIGHT_DIFF] = ctx.pgl_create_program(pgl_tex_pnt_light_diff_vs, pgl_tex_pnt_light_diff_fs, 6, &smooth4_2, false);
    programs[PGL_SHADER_TEX_RECT_REPLACE] = ctx.pgl_create_program(pgl_tex_rplc_vs, pgl_tex_rect_rplc_fs, 2, &smooth2, false);

    programs
}
