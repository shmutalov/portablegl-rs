//! Math types and operations for the PortableGL-rs software renderer.
//!
//! All matrix types use column-major storage for OpenGL compatibility.
//! Types are `#[repr(C)]` for FFI compatibility with the original C library.

use std::ops;

// ---------------------------------------------------------------------------
// Utility free functions
// ---------------------------------------------------------------------------

/// Clamp a float to [0, 1].
#[inline]
pub fn clamp_01(f: f32) -> f32 {
    if f < 0.0 {
        0.0
    } else if f > 1.0 {
        1.0
    } else {
        f
    }
}

/// Clamp an integer to [min, max].
#[inline]
pub fn clampi(i: i32, min: i32, max: i32) -> i32 {
    if i < min {
        min
    } else if i > max {
        max
    } else {
        i
    }
}

/// Linearly map `x` from the range [oldmin, oldmax] to [newmin, newmax].
#[inline]
pub fn rsw_mapf(x: f32, oldmin: f32, oldmax: f32, newmin: f32, newmax: f32) -> f32 {
    newmin + (x - oldmin) / (oldmax - oldmin) * (newmax - newmin)
}

// ===========================================================================
// Vec2
// ===========================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn len(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let l = self.len();
        if l == 0.0 {
            return self;
        }
        Self {
            x: self.x / l,
            y: self.y / l,
        }
    }

    #[inline]
    pub fn dot(a: Self, b: Self) -> f32 {
        a.x * b.x + a.y * b.y
    }

    /// 2D cross product (returns scalar: the z-component of the 3D cross product).
    #[inline]
    pub fn cross(a: Self, b: Self) -> f32 {
        a.x * b.y - a.y * b.x
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }

    #[inline]
    pub fn add(a: Self, b: Self) -> Self {
        Self {
            x: a.x + b.x,
            y: a.y + b.y,
        }
    }

    #[inline]
    pub fn sub(a: Self, b: Self) -> Self {
        Self {
            x: a.x - b.x,
            y: a.y - b.y,
        }
    }

    /// Component-wise multiplication.
    #[inline]
    pub fn mul_comp(a: Self, b: Self) -> Self {
        Self {
            x: a.x * b.x,
            y: a.y * b.y,
        }
    }

    /// Component-wise division.
    #[inline]
    pub fn div_comp(a: Self, b: Self) -> Self {
        Self {
            x: a.x / b.x,
            y: a.y / b.y,
        }
    }
}

// --- std::ops for Vec2 ---

impl ops::Add for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Vec2::add(self, rhs)
    }
}

impl ops::Sub for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Vec2::sub(self, rhs)
    }
}

impl ops::Mul<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        self.scale(rhs)
    }
}

impl ops::Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline]
    fn mul(self, rhs: Vec2) -> Vec2 {
        rhs.scale(self)
    }
}

impl ops::Neg for Vec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

// ---------------------------------------------------------------------------
// Free-function aliases for Vec2
// ---------------------------------------------------------------------------

#[inline]
pub fn vec2(x: f32, y: f32) -> Vec2 {
    Vec2::new(x, y)
}

#[inline]
pub fn len_v2(v: Vec2) -> f32 {
    v.len()
}

#[inline]
pub fn norm_v2(v: Vec2) -> Vec2 {
    v.normalize()
}

#[inline]
pub fn dot_v2s(a: Vec2, b: Vec2) -> f32 {
    Vec2::dot(a, b)
}

#[inline]
pub fn cross_v2s(a: Vec2, b: Vec2) -> f32 {
    Vec2::cross(a, b)
}

#[inline]
pub fn scale_v2(v: Vec2, s: f32) -> Vec2 {
    v.scale(s)
}

#[inline]
pub fn add_v2s(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::add(a, b)
}

#[inline]
pub fn sub_v2s(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::sub(a, b)
}

#[inline]
pub fn mul_v2s(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::mul_comp(a, b)
}

#[inline]
pub fn div_v2s(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::div_comp(a, b)
}

// ===========================================================================
// Vec3
// ===========================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn len(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let l = self.len();
        if l == 0.0 {
            return self;
        }
        Self {
            x: self.x / l,
            y: self.y / l,
            z: self.z / l,
        }
    }

    #[inline]
    pub fn dot(a: Self, b: Self) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    #[inline]
    pub fn cross(a: Self, b: Self) -> Self {
        Self {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    #[inline]
    pub fn add(a: Self, b: Self) -> Self {
        Self {
            x: a.x + b.x,
            y: a.y + b.y,
            z: a.z + b.z,
        }
    }

    #[inline]
    pub fn sub(a: Self, b: Self) -> Self {
        Self {
            x: a.x - b.x,
            y: a.y - b.y,
            z: a.z - b.z,
        }
    }

    /// Component-wise multiplication.
    #[inline]
    pub fn mul_comp(a: Self, b: Self) -> Self {
        Self {
            x: a.x * b.x,
            y: a.y * b.y,
            z: a.z * b.z,
        }
    }

    /// Component-wise division.
    #[inline]
    pub fn div_comp(a: Self, b: Self) -> Self {
        Self {
            x: a.x / b.x,
            y: a.y / b.y,
            z: a.z / b.z,
        }
    }
}

// --- std::ops for Vec3 ---

impl ops::Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Vec3::add(self, rhs)
    }
}

impl ops::Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Vec3::sub(self, rhs)
    }
}

impl ops::Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        self.scale(rhs)
    }
}

impl ops::Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs.scale(self)
    }
}

impl ops::Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// ---------------------------------------------------------------------------
// Free-function aliases for Vec3
// ---------------------------------------------------------------------------

#[inline]
pub fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

#[inline]
pub fn len_v3(v: Vec3) -> f32 {
    v.len()
}

#[inline]
pub fn norm_v3(v: Vec3) -> Vec3 {
    v.normalize()
}

#[inline]
pub fn dot_v3s(a: Vec3, b: Vec3) -> f32 {
    Vec3::dot(a, b)
}

#[inline]
pub fn cross_v3s(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::cross(a, b)
}

#[inline]
pub fn scale_v3(v: Vec3, s: f32) -> Vec3 {
    v.scale(s)
}

#[inline]
pub fn add_v3s(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::add(a, b)
}

#[inline]
pub fn sub_v3s(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::sub(a, b)
}

#[inline]
pub fn mul_v3s(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::mul_comp(a, b)
}

#[inline]
pub fn div_v3s(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::div_comp(a, b)
}

// ===========================================================================
// Vec4
// ===========================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
            w: self.w * s,
        }
    }

    #[inline]
    pub fn add(a: Self, b: Self) -> Self {
        Self {
            x: a.x + b.x,
            y: a.y + b.y,
            z: a.z + b.z,
            w: a.w + b.w,
        }
    }

    #[inline]
    pub fn sub(a: Self, b: Self) -> Self {
        Self {
            x: a.x - b.x,
            y: a.y - b.y,
            z: a.z - b.z,
            w: a.w - b.w,
        }
    }

    /// Component-wise multiplication.
    #[inline]
    pub fn mul_comp(a: Self, b: Self) -> Self {
        Self {
            x: a.x * b.x,
            y: a.y * b.y,
            z: a.z * b.z,
            w: a.w * b.w,
        }
    }

    /// Component-wise division.
    #[inline]
    pub fn div_comp(a: Self, b: Self) -> Self {
        Self {
            x: a.x / b.x,
            y: a.y / b.y,
            z: a.z / b.z,
            w: a.w / b.w,
        }
    }

    /// Perspective divide: returns `Vec3(x/w, y/w, z/w)`.
    #[inline]
    pub fn to_vec3h(self) -> Vec3 {
        Vec3 {
            x: self.x / self.w,
            y: self.y / self.w,
            z: self.z / self.w,
        }
    }
}

// --- std::ops for Vec4 ---

impl ops::Add for Vec4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Vec4::add(self, rhs)
    }
}

impl ops::Sub for Vec4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Vec4::sub(self, rhs)
    }
}

impl ops::Mul<f32> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        self.scale(rhs)
    }
}

impl ops::Mul<Vec4> for f32 {
    type Output = Vec4;
    #[inline]
    fn mul(self, rhs: Vec4) -> Vec4 {
        rhs.scale(self)
    }
}

impl ops::Neg for Vec4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

// ---------------------------------------------------------------------------
// Free-function aliases for Vec4
// ---------------------------------------------------------------------------

#[inline]
pub fn vec4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    Vec4::new(x, y, z, w)
}

#[inline]
pub fn scale_v4(v: Vec4, s: f32) -> Vec4 {
    v.scale(s)
}

#[inline]
pub fn add_v4s(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::add(a, b)
}

#[inline]
pub fn sub_v4s(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::sub(a, b)
}

#[inline]
pub fn mul_v4s(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::mul_comp(a, b)
}

#[inline]
pub fn div_v4s(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::div_comp(a, b)
}

#[inline]
pub fn vec4_to_vec3h(v: Vec4) -> Vec3 {
    v.to_vec3h()
}

// ===========================================================================
// IVec3
// ===========================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct IVec3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl IVec3 {
    #[inline]
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

#[inline]
pub fn ivec3(x: i32, y: i32, z: i32) -> IVec3 {
    IVec3::new(x, y, z)
}

// ===========================================================================
// Mat3
// ===========================================================================

/// 3x3 matrix stored in column-major order.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Mat3(pub [f32; 9]);

impl Default for Mat3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Mat3 {
    /// Create a `Mat3` from raw column-major data.
    #[inline]
    pub fn new(data: [f32; 9]) -> Self {
        Self(data)
    }

    /// Return the 3x3 identity matrix.
    #[inline]
    pub fn identity() -> Self {
        #[rustfmt::skip]
        let m = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        Self(m)
    }
}

// ===========================================================================
// Mat4
// ===========================================================================

/// 4x4 matrix stored in column-major order.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Mat4(pub [f32; 16]);

impl Default for Mat4 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Mat4 {
    /// Create a `Mat4` from raw column-major data.
    #[inline]
    pub fn new(data: [f32; 16]) -> Self {
        Self(data)
    }

    /// Return the 4x4 identity matrix.
    #[inline]
    pub fn identity() -> Self {
        #[rustfmt::skip]
        let m = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        Self(m)
    }

    /// Multiply this matrix by a `Vec4` (column vector on the right).
    #[inline]
    pub fn mult_m4_v4(self, v: Vec4) -> Vec4 {
        let m = &self.0;
        Vec4 {
            x: m[0] * v.x + m[4] * v.y + m[8]  * v.z + m[12] * v.w,
            y: m[1] * v.x + m[5] * v.y + m[9]  * v.z + m[13] * v.w,
            z: m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
            w: m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w,
        }
    }
}

impl ops::Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, rhs: Vec4) -> Vec4 {
        self.mult_m4_v4(rhs)
    }
}

// ---------------------------------------------------------------------------
// Free-function aliases for Mat4
// ---------------------------------------------------------------------------

/// Multiply a `Mat4` by a `Vec4`.
#[inline]
pub fn mult_m4_v4(m: Mat4, v: Vec4) -> Vec4 {
    m.mult_m4_v4(v)
}

/// Build a viewport matrix that maps from NDC to screen coordinates.
///
/// * `x`, `y`   -- lower-left corner of the viewport in pixels
/// * `w`, `h`   -- width and height of the viewport in pixels
/// * `half_z`   -- if non-zero, map z from [-1,1] to [0,1]; otherwise keep [-1,1]
#[inline]
pub fn make_viewport_matrix(x: i32, y: i32, w: i32, h: i32, half_z: i32) -> Mat4 {
    let wf = w as f32;
    let hf = h as f32;
    let xf = x as f32;
    let yf = y as f32;

    let mut m = [0.0f32; 16];
    m[0]  = wf / 2.0;
    m[5]  = hf / 2.0;
    m[10] = if half_z != 0 { 0.5 } else { 1.0 };
    m[12] = xf + wf / 2.0;
    m[13] = yf + hf / 2.0;
    m[14] = if half_z != 0 { 0.5 } else { 0.0 };
    m[15] = 1.0;
    Mat4(m)
}

// ===========================================================================
// Color
// ===========================================================================

/// RGBA colour stored as four bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    #[inline]
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Convert to a `Vec4` with each component in [0.0, 1.0].
    #[inline]
    pub fn to_vec4(self) -> Vec4 {
        Vec4 {
            x: self.r as f32 / 255.0,
            y: self.g as f32 / 255.0,
            z: self.b as f32 / 255.0,
            w: self.a as f32 / 255.0,
        }
    }

    /// Create a `Color` from a `Vec4`, clamping each component to [0.0, 1.0]
    /// and scaling to [0, 255].
    #[inline]
    pub fn from_vec4(v: Vec4) -> Self {
        Self {
            r: (clamp_01(v.x) * 255.0 + 0.5) as u8,
            g: (clamp_01(v.y) * 255.0 + 0.5) as u8,
            b: (clamp_01(v.z) * 255.0 + 0.5) as u8,
            a: (clamp_01(v.w) * 255.0 + 0.5) as u8,
        }
    }
}

// ===========================================================================
// Line
// ===========================================================================

/// Implicit line equation `Ax + By + C = 0`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Line {
    pub a: f32,
    pub b: f32,
    pub c: f32,
}

impl Line {
    /// Construct a `Line` from two points `(x1,y1)` and `(x2,y2)`.
    ///
    /// The line is `A*x + B*y + C = 0` where:
    /// * `A = y1 - y2`
    /// * `B = x2 - x1`
    /// * `C = x1*y2 - x2*y1`
    #[inline]
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            a: y1 - y2,
            b: x2 - x1,
            c: x1 * y2 - x2 * y1,
        }
    }

    /// Evaluate `A*x + B*y + C`.
    ///
    /// The sign tells you which side of the line the point is on.
    #[inline]
    pub fn func(&self, x: f32, y: f32) -> f32 {
        self.a * x + self.b * y + self.c
    }

    /// Normalize the line equation so that `A*A + B*B == 1`.
    ///
    /// After normalisation, `func(x, y)` returns the signed distance from
    /// `(x, y)` to the line.
    #[inline]
    pub fn normalize(&mut self) {
        let l = (self.a * self.a + self.b * self.b).sqrt();
        if l != 0.0 {
            self.a /= l;
            self.b /= l;
            self.c /= l;
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_01() {
        assert_eq!(clamp_01(-0.5), 0.0);
        assert_eq!(clamp_01(0.5), 0.5);
        assert_eq!(clamp_01(1.5), 1.0);
    }

    #[test]
    fn test_clampi() {
        assert_eq!(clampi(-5, 0, 10), 0);
        assert_eq!(clampi(5, 0, 10), 5);
        assert_eq!(clampi(15, 0, 10), 10);
    }

    #[test]
    fn test_rsw_mapf() {
        let v = rsw_mapf(0.5, 0.0, 1.0, 0.0, 10.0);
        assert!((v - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = Vec3::cross(a, b);
        assert!((c.x).abs() < 1e-6);
        assert!((c.y).abs() < 1e-6);
        assert!((c.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mat4_identity_mul() {
        let m = Mat4::identity();
        let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let r = m * v;
        assert!((r.x - 1.0).abs() < 1e-6);
        assert!((r.y - 2.0).abs() < 1e-6);
        assert!((r.z - 3.0).abs() < 1e-6);
        assert!((r.w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_viewport_matrix() {
        let m = make_viewport_matrix(0, 0, 800, 600, 0);
        let ndc = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let screen = m * ndc;
        assert!((screen.x - 400.0).abs() < 1e-3);
        assert!((screen.y - 300.0).abs() < 1e-3);
    }

    #[test]
    fn test_color_roundtrip() {
        let c = Color::new(128, 64, 255, 0);
        let v = c.to_vec4();
        let c2 = Color::from_vec4(v);
        assert_eq!(c, c2);
    }

    #[test]
    fn test_line() {
        // Horizontal line y=1: points (0,1) and (2,1)
        let l = Line::new(0.0, 1.0, 2.0, 1.0);
        // Points on the line evaluate to 0
        assert!((l.func(1.0, 1.0)).abs() < 1e-6);
        // Point above the line
        assert!(l.func(1.0, 2.0) != 0.0);
    }

    #[test]
    fn test_vec4_to_vec3h() {
        let v = Vec4::new(2.0, 4.0, 6.0, 2.0);
        let v3 = v.to_vec3h();
        assert!((v3.x - 1.0).abs() < 1e-6);
        assert!((v3.y - 2.0).abs() < 1e-6);
        assert!((v3.z - 3.0).abs() < 1e-6);
    }
}
