//! GLSL-like texture sampling functions for the PortableGL Rust port.
//!
//! This module implements texture sampling functions that users call from their
//! fragment/vertex shaders. All public functions are methods on `GlContext`.

#![allow(non_upper_case_globals, non_snake_case, dead_code, clippy::too_many_arguments)]

use crate::math::*;
use crate::gl_types::*;
use crate::gl_context::*;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 0.000001;

// ---------------------------------------------------------------------------
// Texture wrapping helper (private)
// ---------------------------------------------------------------------------

/// Wrap a texel coordinate `i` into the range `[0, size-1]` according to `mode`.
fn wrap(i: i32, size: i32, mode: GLenum) -> i32 {
    match mode {
        GL_REPEAT => ((i % size) + size) % size,
        GL_CLAMP_TO_EDGE | GL_CLAMP_TO_BORDER => i.clamp(0, size - 1),
        GL_MIRRORED_REPEAT => {
            let sz2 = 2 * size;
            let mut v = ((i % sz2) + sz2) % sz2;
            v -= size;
            v = if v >= 0 { v } else { -(1 + v) };
            size - 1 - v
        }
        _ => i.clamp(0, size - 1),
    }
}

// ---------------------------------------------------------------------------
// Color conversion helper
// ---------------------------------------------------------------------------

/// Unpack an ABGR32 pixel (stored as u32, little-endian RGBA layout) into a Vec4.
///
/// Memory layout: byte 0 = R, byte 1 = G, byte 2 = B, byte 3 = A.
#[inline]
fn color_from_u32(pixel: u32) -> Vec4 {
    let r = (pixel & 0xFF) as f32 / 255.0;
    let g = ((pixel >> 8) & 0xFF) as f32 / 255.0;
    let b = ((pixel >> 16) & 0xFF) as f32 / 255.0;
    let a = ((pixel >> 24) & 0xFF) as f32 / 255.0;
    Vec4::new(r, g, b, a)
}

// ---------------------------------------------------------------------------
// Private helper to reinterpret texture data as a &[u32] slice
// ---------------------------------------------------------------------------

/// Reinterpret the texture's raw byte data as a slice of packed RGBA pixels.
///
/// # Safety
/// The texture data must be aligned and sized as a multiple of 4 bytes.
#[inline]
fn tex_data_as_u32(tex: &GlTexture) -> &[u32] {
    let len = tex.data.len() / 4;
    if len == 0 {
        return &[];
    }
    let ptr = tex.data.as_ptr() as *const u32;
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

// ---------------------------------------------------------------------------
// Fractional part helper (matches C modf behavior)
// ---------------------------------------------------------------------------

/// Return the fractional part of `x`. Always in [0, 1) for positive values.
#[inline]
fn fract(x: f32) -> f32 {
    x - x.floor()
}

// ---------------------------------------------------------------------------
// GlContext texture sampling methods
// ---------------------------------------------------------------------------

impl GlContext {
    // -----------------------------------------------------------------------
    // Private helper: look up a texture by handle
    // -----------------------------------------------------------------------

    /// Retrieve a reference to the texture for a given handle.
    ///
    /// If `tex` is nonzero and valid, returns `&self.textures[tex]`.
    /// Otherwise falls back to the default texture for the given target.
    fn get_texture(&self, tex: GLuint, target: GLenum) -> &GlTexture {
        if tex != 0 && (tex as usize) < self.textures.len() {
            &self.textures[tex as usize]
        } else {
            let idx = (target - GL_TEXTURE_UNBOUND) as usize;
            if idx < self.default_textures.len() {
                &self.default_textures[idx]
            } else {
                // Fallback: return the first default texture (should not happen
                // in well-formed programs).
                &self.default_textures[0]
            }
        }
    }

    // =======================================================================
    // texture1d
    // =======================================================================

    /// Sample a 1D texture at normalised coordinate `x`.
    pub fn texture1d(&self, tex: GLuint, x: f32) -> Vec4 {
        let t = self.get_texture(tex, GL_TEXTURE_1D);
        let texdata = tex_data_as_u32(t);
        if texdata.is_empty() || t.w <= 0 {
            return Vec4::default();
        }

        let w = t.w as f32 - EPSILON;
        let xw = x * w;

        if t.mag_filter == GL_NEAREST {
            let i0 = wrap(xw.floor() as i32, t.w, t.wrap_s);
            color_from_u32(texdata[i0 as usize])
        } else {
            // GL_LINEAR
            let i0 = wrap((xw - 0.5).floor() as i32, t.w, t.wrap_s);
            let i1 = wrap((xw + 0.499999).floor() as i32, t.w, t.wrap_s);
            let mut alpha = fract(xw + 0.5);
            if alpha < 0.0 {
                alpha += 1.0;
            }
            let ci = color_from_u32(texdata[i0 as usize]) * (1.0 - alpha);
            let ci1 = color_from_u32(texdata[i1 as usize]) * alpha;
            ci + ci1
        }
    }

    // =======================================================================
    // texture2d
    // =======================================================================

    /// Sample a 2D texture at normalised coordinates `(x, y)`.
    pub fn texture2d(&self, tex: GLuint, x: f32, y: f32) -> Vec4 {
        let t = self.get_texture(tex, GL_TEXTURE_2D);
        let texdata = tex_data_as_u32(t);
        if texdata.is_empty() || t.w <= 0 || t.h <= 0 {
            return Vec4::default();
        }

        let w = t.w as f32 - EPSILON;
        let h = t.h as f32 - EPSILON;
        let xw = x * w;
        let yh = y * h;

        let tw = t.w;

        if t.mag_filter == GL_NEAREST {
            let i0 = wrap(xw.floor() as i32, t.w, t.wrap_s);
            let j0 = wrap(yh.floor() as i32, t.h, t.wrap_t);
            color_from_u32(texdata[(j0 * tw + i0) as usize])
        } else {
            // GL_LINEAR — bilinear interpolation
            let i0 = wrap((xw - 0.5).floor() as i32, t.w, t.wrap_s);
            let i1 = wrap((xw + 0.499999).floor() as i32, t.w, t.wrap_s);
            let j0 = wrap((yh - 0.5).floor() as i32, t.h, t.wrap_t);
            let j1 = wrap((yh + 0.499999).floor() as i32, t.h, t.wrap_t);

            let mut alpha = fract(xw + 0.5);
            if alpha < 0.0 {
                alpha += 1.0;
            }
            let mut beta = fract(yh + 0.5);
            if beta < 0.0 {
                beta += 1.0;
            }

            let cij = color_from_u32(texdata[(j0 * tw + i0) as usize]);
            let ci1j = color_from_u32(texdata[(j0 * tw + i1) as usize]);
            let cij1 = color_from_u32(texdata[(j1 * tw + i0) as usize]);
            let ci1j1 = color_from_u32(texdata[(j1 * tw + i1) as usize]);

            cij * ((1.0 - alpha) * (1.0 - beta))
                + ci1j * (alpha * (1.0 - beta))
                + cij1 * ((1.0 - alpha) * beta)
                + ci1j1 * (alpha * beta)
        }
    }

    // =======================================================================
    // texture3d
    // =======================================================================

    /// Sample a 3D texture at normalised coordinates `(x, y, z)`.
    pub fn texture3d(&self, tex: GLuint, x: f32, y: f32, z: f32) -> Vec4 {
        let t = self.get_texture(tex, GL_TEXTURE_3D);
        let texdata = tex_data_as_u32(t);
        if texdata.is_empty() || t.w <= 0 || t.h <= 0 || t.d <= 0 {
            return Vec4::default();
        }

        let w = t.w as f32 - EPSILON;
        let h = t.h as f32 - EPSILON;
        let d = t.d as f32 - EPSILON;
        let xw = x * w;
        let yh = y * h;
        let zd = z * d;

        let tw = t.w;
        let th = t.h;
        let plane = tw * th; // pixels per depth slice

        if t.mag_filter == GL_NEAREST {
            let i0 = wrap(xw.floor() as i32, t.w, t.wrap_s);
            let j0 = wrap(yh.floor() as i32, t.h, t.wrap_t);
            let k0 = wrap(zd.floor() as i32, t.d, t.wrap_r);
            color_from_u32(texdata[(k0 * plane + j0 * tw + i0) as usize])
        } else {
            // GL_LINEAR — trilinear interpolation (8 samples)
            let i0 = wrap((xw - 0.5).floor() as i32, t.w, t.wrap_s);
            let i1 = wrap((xw + 0.499999).floor() as i32, t.w, t.wrap_s);
            let j0 = wrap((yh - 0.5).floor() as i32, t.h, t.wrap_t);
            let j1 = wrap((yh + 0.499999).floor() as i32, t.h, t.wrap_t);
            let k0 = wrap((zd - 0.5).floor() as i32, t.d, t.wrap_r);
            let k1 = wrap((zd + 0.499999).floor() as i32, t.d, t.wrap_r);

            let mut alpha = fract(xw + 0.5);
            if alpha < 0.0 {
                alpha += 1.0;
            }
            let mut beta = fract(yh + 0.5);
            if beta < 0.0 {
                beta += 1.0;
            }
            let mut gamma = fract(zd + 0.5);
            if gamma < 0.0 {
                gamma += 1.0;
            }

            // Front face (k0)
            let c000 = color_from_u32(texdata[(k0 * plane + j0 * tw + i0) as usize]);
            let c100 = color_from_u32(texdata[(k0 * plane + j0 * tw + i1) as usize]);
            let c010 = color_from_u32(texdata[(k0 * plane + j1 * tw + i0) as usize]);
            let c110 = color_from_u32(texdata[(k0 * plane + j1 * tw + i1) as usize]);

            // Back face (k1)
            let c001 = color_from_u32(texdata[(k1 * plane + j0 * tw + i0) as usize]);
            let c101 = color_from_u32(texdata[(k1 * plane + j0 * tw + i1) as usize]);
            let c011 = color_from_u32(texdata[(k1 * plane + j1 * tw + i0) as usize]);
            let c111 = color_from_u32(texdata[(k1 * plane + j1 * tw + i1) as usize]);

            let a0 = 1.0 - alpha;
            let b0 = 1.0 - beta;
            let g0 = 1.0 - gamma;

            c000 * (a0 * b0 * g0)
                + c100 * (alpha * b0 * g0)
                + c010 * (a0 * beta * g0)
                + c110 * (alpha * beta * g0)
                + c001 * (a0 * b0 * gamma)
                + c101 * (alpha * b0 * gamma)
                + c011 * (a0 * beta * gamma)
                + c111 * (alpha * beta * gamma)
        }
    }

    // =======================================================================
    // texture2d_array
    // =======================================================================

    /// Sample a 2D texture array at normalised coordinates `(x, y)` and
    /// integer layer index `z`.
    pub fn texture2d_array(&self, tex: GLuint, x: f32, y: f32, z: i32) -> Vec4 {
        let t = self.get_texture(tex, GL_TEXTURE_2D_ARRAY);
        let texdata = tex_data_as_u32(t);
        if texdata.is_empty() || t.w <= 0 || t.h <= 0 || t.d <= 0 {
            return Vec4::default();
        }

        let w = t.w as f32 - EPSILON;
        let h = t.h as f32 - EPSILON;
        let xw = x * w;
        let yh = y * h;

        let tw = t.w;
        let th = t.h;
        let plane = tw * th;
        let k = z.clamp(0, t.d - 1);

        if t.mag_filter == GL_NEAREST {
            let i0 = wrap(xw.floor() as i32, t.w, t.wrap_s);
            let j0 = wrap(yh.floor() as i32, t.h, t.wrap_t);
            color_from_u32(texdata[(k * plane + j0 * tw + i0) as usize])
        } else {
            // GL_LINEAR — bilinear within the layer
            let i0 = wrap((xw - 0.5).floor() as i32, t.w, t.wrap_s);
            let i1 = wrap((xw + 0.499999).floor() as i32, t.w, t.wrap_s);
            let j0 = wrap((yh - 0.5).floor() as i32, t.h, t.wrap_t);
            let j1 = wrap((yh + 0.499999).floor() as i32, t.h, t.wrap_t);

            let mut alpha = fract(xw + 0.5);
            if alpha < 0.0 {
                alpha += 1.0;
            }
            let mut beta = fract(yh + 0.5);
            if beta < 0.0 {
                beta += 1.0;
            }

            let base = k * plane;
            let cij = color_from_u32(texdata[(base + j0 * tw + i0) as usize]);
            let ci1j = color_from_u32(texdata[(base + j0 * tw + i1) as usize]);
            let cij1 = color_from_u32(texdata[(base + j1 * tw + i0) as usize]);
            let ci1j1 = color_from_u32(texdata[(base + j1 * tw + i1) as usize]);

            cij * ((1.0 - alpha) * (1.0 - beta))
                + ci1j * (alpha * (1.0 - beta))
                + cij1 * ((1.0 - alpha) * beta)
                + ci1j1 * (alpha * beta)
        }
    }

    // =======================================================================
    // texture_rect
    // =======================================================================

    /// Sample a rectangle texture at texel-space coordinates `(x, y)`.
    ///
    /// Unlike `texture2d`, coordinates are NOT normalised -- they are already
    /// in texel space and are not multiplied by the texture dimensions.
    pub fn texture_rect(&self, tex: GLuint, x: f32, y: f32) -> Vec4 {
        let t = self.get_texture(tex, GL_TEXTURE_RECTANGLE);
        let texdata = tex_data_as_u32(t);
        if texdata.is_empty() || t.w <= 0 || t.h <= 0 {
            return Vec4::default();
        }

        let tw = t.w;

        if t.mag_filter == GL_NEAREST {
            let i0 = wrap(x.floor() as i32, t.w, t.wrap_s);
            let j0 = wrap(y.floor() as i32, t.h, t.wrap_t);
            color_from_u32(texdata[(j0 * tw + i0) as usize])
        } else {
            // GL_LINEAR — bilinear interpolation
            let i0 = wrap((x - 0.5).floor() as i32, t.w, t.wrap_s);
            let i1 = wrap((x + 0.499999).floor() as i32, t.w, t.wrap_s);
            let j0 = wrap((y - 0.5).floor() as i32, t.h, t.wrap_t);
            let j1 = wrap((y + 0.499999).floor() as i32, t.h, t.wrap_t);

            let mut alpha = fract(x + 0.5);
            if alpha < 0.0 {
                alpha += 1.0;
            }
            let mut beta = fract(y + 0.5);
            if beta < 0.0 {
                beta += 1.0;
            }

            let cij = color_from_u32(texdata[(j0 * tw + i0) as usize]);
            let ci1j = color_from_u32(texdata[(j0 * tw + i1) as usize]);
            let cij1 = color_from_u32(texdata[(j1 * tw + i0) as usize]);
            let ci1j1 = color_from_u32(texdata[(j1 * tw + i1) as usize]);

            cij * ((1.0 - alpha) * (1.0 - beta))
                + ci1j * (alpha * (1.0 - beta))
                + cij1 * ((1.0 - alpha) * beta)
                + ci1j1 * (alpha * beta)
        }
    }

    // =======================================================================
    // texture_cubemap
    // =======================================================================

    /// Sample a cube-map texture using a direction vector `(x, y, z)`.
    ///
    /// The six faces are stored sequentially in the texture data, each face
    /// being `w * w` pixels. The face order matches the OpenGL convention:
    ///   0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z
    pub fn texture_cubemap(&self, tex: GLuint, x: f32, y: f32, z: f32) -> Vec4 {
        let t = self.get_texture(tex, GL_TEXTURE_CUBE_MAP);
        let texdata = tex_data_as_u32(t);
        if texdata.is_empty() || t.w <= 0 {
            return Vec4::default();
        }

        let ax = x.abs();
        let ay = y.abs();
        let az = z.abs();

        // Determine dominant axis and compute face index + (s, t) coordinates.
        let (face, mut s, mut t_coord): (i32, f32, f32);

        if ax >= ay && ax >= az {
            // +/- X is dominant
            let max = ax;
            if x > 0.0 {
                // +X face (index 0)
                face = 0;
                s = -z;
                t_coord = -y;
            } else {
                // -X face (index 1)
                face = 1;
                s = z;
                t_coord = -y;
            }
            s = (s / max + 1.0) / 2.0;
            t_coord = (t_coord / max + 1.0) / 2.0;
        } else if ay >= ax && ay >= az {
            // +/- Y is dominant
            let max = ay;
            if y > 0.0 {
                // +Y face (index 2)
                face = 2;
                s = x;
                t_coord = z;
            } else {
                // -Y face (index 3)
                face = 3;
                s = x;
                t_coord = -z;
            }
            s = (s / max + 1.0) / 2.0;
            t_coord = (t_coord / max + 1.0) / 2.0;
        } else {
            // +/- Z is dominant
            let max = az;
            if z > 0.0 {
                // +Z face (index 4)
                face = 4;
                s = x;
                t_coord = -y;
            } else {
                // -Z face (index 5)
                face = 5;
                s = -x;
                t_coord = -y;
            }
            s = (s / max + 1.0) / 2.0;
            t_coord = (t_coord / max + 1.0) / 2.0;
        }

        // Each face is w * w pixels, stored sequentially.
        let face_size = t.w * t.w;
        let face_offset = face * face_size;
        let tw = t.w;
        let w = tw as f32 - EPSILON;

        let xw = s * w;
        let yh = t_coord * w;

        if t.mag_filter == GL_NEAREST {
            let i0 = wrap(xw.floor() as i32, tw, t.wrap_s);
            let j0 = wrap(yh.floor() as i32, tw, t.wrap_t);
            color_from_u32(texdata[(face_offset + j0 * tw + i0) as usize])
        } else {
            // GL_LINEAR — bilinear interpolation on the face
            let i0 = wrap((xw - 0.5).floor() as i32, tw, t.wrap_s);
            let i1 = wrap((xw + 0.499999).floor() as i32, tw, t.wrap_s);
            let j0 = wrap((yh - 0.5).floor() as i32, tw, t.wrap_t);
            let j1 = wrap((yh + 0.499999).floor() as i32, tw, t.wrap_t);

            let mut alpha = fract(xw + 0.5);
            if alpha < 0.0 {
                alpha += 1.0;
            }
            let mut beta = fract(yh + 0.5);
            if beta < 0.0 {
                beta += 1.0;
            }

            let base = face_offset;
            let cij = color_from_u32(texdata[(base + j0 * tw + i0) as usize]);
            let ci1j = color_from_u32(texdata[(base + j0 * tw + i1) as usize]);
            let cij1 = color_from_u32(texdata[(base + j1 * tw + i0) as usize]);
            let ci1j1 = color_from_u32(texdata[(base + j1 * tw + i1) as usize]);

            cij * ((1.0 - alpha) * (1.0 - beta))
                + ci1j * (alpha * (1.0 - beta))
                + cij1 * ((1.0 - alpha) * beta)
                + ci1j1 * (alpha * beta)
        }
    }

    // =======================================================================
    // texelFetch functions — direct integer coordinate access, no filtering
    // =======================================================================

    /// Fetch a single texel from a 1D texture by integer coordinate.
    pub fn texel_fetch1d(&self, tex: GLuint, x: i32, _lod: i32) -> Vec4 {
        let t = self.get_texture(tex, GL_TEXTURE_1D);
        let texdata = tex_data_as_u32(t);
        if texdata.is_empty() || t.w <= 0 {
            return Vec4::default();
        }

        let i = x.clamp(0, t.w - 1) as usize;
        color_from_u32(texdata[i])
    }

    /// Fetch a single texel from a 2D texture by integer coordinates.
    pub fn texel_fetch2d(&self, tex: GLuint, x: i32, y: i32, _lod: i32) -> Vec4 {
        let t = self.get_texture(tex, GL_TEXTURE_2D);
        let texdata = tex_data_as_u32(t);
        if texdata.is_empty() || t.w <= 0 || t.h <= 0 {
            return Vec4::default();
        }

        let i = x.clamp(0, t.w - 1);
        let j = y.clamp(0, t.h - 1);
        color_from_u32(texdata[(j * t.w + i) as usize])
    }

    /// Fetch a single texel from a 3D texture by integer coordinates.
    pub fn texel_fetch3d(&self, tex: GLuint, x: i32, y: i32, z: i32, _lod: i32) -> Vec4 {
        let t = self.get_texture(tex, GL_TEXTURE_3D);
        let texdata = tex_data_as_u32(t);
        if texdata.is_empty() || t.w <= 0 || t.h <= 0 || t.d <= 0 {
            return Vec4::default();
        }

        let i = x.clamp(0, t.w - 1);
        let j = y.clamp(0, t.h - 1);
        let k = z.clamp(0, t.d - 1);
        let plane = t.w * t.h;
        color_from_u32(texdata[(k * plane + j * t.w + i) as usize])
    }

    // =======================================================================
    // texture_size
    // =======================================================================

    /// Return the dimensions of a texture as an `IVec3(w, h, d)`.
    ///
    /// For 1D textures, `h` and `d` will reflect the stored values (typically
    /// `h = 0, d = 0` or `h = 1, d = 1` depending on how the texture was
    /// created).
    pub fn texture_size(&self, tex: GLuint, _lod: GLint) -> IVec3 {
        // Use GL_TEXTURE_2D as a reasonable default target when looking up
        // the texture; the target only matters for the fallback to
        // default_textures, and most callers will pass a valid handle.
        let t = if tex != 0 && (tex as usize) < self.textures.len() {
            &self.textures[tex as usize]
        } else {
            // Fall back; pick 2D as default
            if !self.default_textures.is_empty() {
                let idx = (GL_TEXTURE_2D - GL_TEXTURE_UNBOUND) as usize;
                if idx < self.default_textures.len() {
                    &self.default_textures[idx]
                } else {
                    &self.default_textures[0]
                }
            } else {
                return IVec3::new(0, 0, 0);
            }
        };

        IVec3::new(t.w, t.h, t.d)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a small 2x2 RGBA texture (4 pixels, 16 bytes).
    /// Pixel layout (ABGR u32):
    ///   (0,0) = red,   (1,0) = green
    ///   (0,1) = blue,  (1,1) = white
    fn make_2x2_texture() -> GlTexture {
        let red: u32 = 0xFF0000FF;   // A=FF B=00 G=00 R=FF
        let green: u32 = 0xFF00FF00; // A=FF B=00 G=FF R=00
        let blue: u32 = 0xFFFF0000;  // A=FF B=FF G=00 R=00
        let white: u32 = 0xFFFFFFFF; // A=FF B=FF G=FF R=FF

        let mut data = Vec::with_capacity(16);
        for pixel in &[red, green, blue, white] {
            data.extend_from_slice(&pixel.to_ne_bytes());
        }

        GlTexture {
            w: 2,
            h: 2,
            d: 1,
            mag_filter: GL_NEAREST,
            min_filter: GL_NEAREST,
            wrap_s: GL_REPEAT,
            wrap_t: GL_REPEAT,
            wrap_r: GL_REPEAT,
            format: GL_RGBA,
            type_: GL_TEXTURE_2D,
            deleted: false,
            user_owned: false,
            data,
        }
    }

    fn make_context_with_texture(tex: GlTexture) -> GlContext {
        let mut ctx = GlContext::default();
        // Index 0 is the "null" texture slot
        ctx.textures.push(GlTexture::default());
        // Index 1 is our test texture
        ctx.textures.push(tex);
        // Set up a minimal default_textures array
        let num_targets = (GL_NUM_TEXTURE_TYPES - GL_TEXTURE_UNBOUND) as usize;
        ctx.default_textures = vec![GlTexture::default(); num_targets];
        ctx
    }

    #[test]
    fn test_wrap_repeat() {
        assert_eq!(wrap(5, 4, GL_REPEAT), 1);
        assert_eq!(wrap(-1, 4, GL_REPEAT), 3);
        assert_eq!(wrap(0, 4, GL_REPEAT), 0);
        assert_eq!(wrap(3, 4, GL_REPEAT), 3);
    }

    #[test]
    fn test_wrap_clamp_to_edge() {
        assert_eq!(wrap(-1, 4, GL_CLAMP_TO_EDGE), 0);
        assert_eq!(wrap(5, 4, GL_CLAMP_TO_EDGE), 3);
        assert_eq!(wrap(2, 4, GL_CLAMP_TO_EDGE), 2);
    }

    #[test]
    fn test_wrap_mirrored_repeat() {
        // i=0: sz2=8, v=0%8=0, v-=4 => -4, v=-(1+(-4))=3, result=3-3=0
        assert_eq!(wrap(0, 4, GL_MIRRORED_REPEAT), 0);
        // i=3: v=3%8=3, v-=4 => -1, v=-(1+(-1))=0, result=3-0=3
        assert_eq!(wrap(3, 4, GL_MIRRORED_REPEAT), 3);
        // i=4: v=4%8=4, v-=4 => 0, v=0, result=3-0=3
        assert_eq!(wrap(4, 4, GL_MIRRORED_REPEAT), 3);
        // i=7: v=7%8=7, v-=4 => 3, v=3, result=3-3=0
        assert_eq!(wrap(7, 4, GL_MIRRORED_REPEAT), 0);
    }

    #[test]
    fn test_color_from_u32() {
        let pixel: u32 = 0xFF804020; // A=FF, B=80, G=40, R=20
        let c = color_from_u32(pixel);
        assert!((c.x - 0x20 as f32 / 255.0).abs() < 1e-3); // R
        assert!((c.y - 0x40 as f32 / 255.0).abs() < 1e-3); // G
        assert!((c.z - 0x80 as f32 / 255.0).abs() < 1e-3); // B
        assert!((c.w - 0xFF as f32 / 255.0).abs() < 1e-3); // A
    }

    #[test]
    fn test_texture2d_nearest() {
        let tex = make_2x2_texture();
        let ctx = make_context_with_texture(tex);

        // Sample at (0.25, 0.25) should hit pixel (0,0) = red
        let c = ctx.texture2d(1, 0.25, 0.25);
        assert!((c.x - 1.0).abs() < 1e-3); // R = 1.0
        assert!((c.y - 0.0).abs() < 1e-3); // G = 0.0
        assert!((c.z - 0.0).abs() < 1e-3); // B = 0.0
        assert!((c.w - 1.0).abs() < 1e-3); // A = 1.0
    }

    #[test]
    fn test_texel_fetch2d() {
        let tex = make_2x2_texture();
        let ctx = make_context_with_texture(tex);

        // Fetch pixel (1, 0) = green
        let c = ctx.texel_fetch2d(1, 1, 0, 0);
        assert!((c.x - 0.0).abs() < 1e-3); // R = 0
        assert!((c.y - 1.0).abs() < 1e-3); // G = 1.0
        assert!((c.z - 0.0).abs() < 1e-3); // B = 0
        assert!((c.w - 1.0).abs() < 1e-3); // A = 1.0

        // Fetch pixel (0, 1) = blue
        let c = ctx.texel_fetch2d(1, 0, 1, 0);
        assert!((c.x - 0.0).abs() < 1e-3);
        assert!((c.y - 0.0).abs() < 1e-3);
        assert!((c.z - 1.0).abs() < 1e-3);
        assert!((c.w - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_texture_size() {
        let tex = make_2x2_texture();
        let ctx = make_context_with_texture(tex);

        let sz = ctx.texture_size(1, 0);
        assert_eq!(sz.x, 2);
        assert_eq!(sz.y, 2);
        assert_eq!(sz.z, 1);
    }

    #[test]
    fn test_texture1d_nearest() {
        // Create a 4-pixel 1D texture: red, green, blue, white
        let red: u32 = 0xFF0000FF;
        let green: u32 = 0xFF00FF00;
        let blue: u32 = 0xFFFF0000;
        let white: u32 = 0xFFFFFFFF;

        let mut data = Vec::with_capacity(16);
        for pixel in &[red, green, blue, white] {
            data.extend_from_slice(&pixel.to_ne_bytes());
        }

        let tex = GlTexture {
            w: 4,
            h: 1,
            d: 1,
            mag_filter: GL_NEAREST,
            min_filter: GL_NEAREST,
            wrap_s: GL_REPEAT,
            wrap_t: GL_REPEAT,
            wrap_r: GL_REPEAT,
            format: GL_RGBA,
            type_: GL_TEXTURE_1D,
            deleted: false,
            user_owned: false,
            data,
        };
        let ctx = make_context_with_texture(tex);

        // Sample near the start -> red
        let c = ctx.texture1d(1, 0.1);
        assert!((c.x - 1.0).abs() < 1e-3);
        assert!((c.y - 0.0).abs() < 1e-3);

        // Sample near 0.75 -> last pixel = white
        let c = ctx.texture1d(1, 0.9);
        assert!((c.x - 1.0).abs() < 1e-3);
        assert!((c.y - 1.0).abs() < 1e-3);
        assert!((c.z - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_texture2d_linear() {
        let mut tex = make_2x2_texture();
        tex.mag_filter = GL_LINEAR;
        let ctx = make_context_with_texture(tex);

        // Sample at the center (0.5, 0.5) — should blend all four pixels
        let c = ctx.texture2d(1, 0.5, 0.5);
        // All four contribute equally, so result should be the average
        // Red(1,0,0,1) + Green(0,1,0,1) + Blue(0,0,1,1) + White(1,1,1,1) / 4
        // = (0.5, 0.5, 0.5, 1.0)
        assert!((c.x - 0.5).abs() < 0.1);
        assert!((c.y - 0.5).abs() < 0.1);
        assert!((c.z - 0.5).abs() < 0.1);
        assert!((c.w - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_empty_texture_returns_default() {
        let ctx = make_context_with_texture(GlTexture::default());
        let c = ctx.texture2d(1, 0.5, 0.5);
        assert_eq!(c, Vec4::default());
    }
}
