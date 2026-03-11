//! Float math compatibility layer for no_std support.
//!
//! When `std` is available, these are simple wrappers around f32 methods.
//! When `no_std` is active, they delegate to `libm`.

#[cfg(not(feature = "no_std"))]
mod imp {
    #[inline(always)] pub fn floor(x: f32) -> f32 { x.floor() }
    #[inline(always)] pub fn ceil(x: f32) -> f32 { x.ceil() }
    #[inline(always)] pub fn sqrt(x: f32) -> f32 { x.sqrt() }
    #[inline(always)] pub fn abs(x: f32) -> f32 { x.abs() }
    #[inline(always)] pub fn min(a: f32, b: f32) -> f32 { a.min(b) }
    #[inline(always)] pub fn max(a: f32, b: f32) -> f32 { a.max(b) }
    #[inline(always)] pub fn clamp(x: f32, lo: f32, hi: f32) -> f32 { x.clamp(lo, hi) }
    #[inline(always)] pub fn sin(x: f32) -> f32 { x.sin() }
    #[inline(always)] pub fn cos(x: f32) -> f32 { x.cos() }
    #[inline(always)] pub fn tan(x: f32) -> f32 { x.tan() }
}

#[cfg(feature = "no_std")]
mod imp {
    #[inline(always)] pub fn floor(x: f32) -> f32 { libm::floorf(x) }
    #[inline(always)] pub fn ceil(x: f32) -> f32 { libm::ceilf(x) }
    #[inline(always)] pub fn sqrt(x: f32) -> f32 { libm::sqrtf(x) }
    #[inline(always)] pub fn abs(x: f32) -> f32 { libm::fabsf(x) }
    #[inline(always)]
    pub fn min(a: f32, b: f32) -> f32 {
        if a < b { a } else { b }
    }
    #[inline(always)]
    pub fn max(a: f32, b: f32) -> f32 {
        if a > b { a } else { b }
    }
    #[inline(always)]
    pub fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
        if x < lo { lo } else if x > hi { hi } else { x }
    }
    #[inline(always)] pub fn sin(x: f32) -> f32 { libm::sinf(x) }
    #[inline(always)] pub fn cos(x: f32) -> f32 { libm::cosf(x) }
    #[inline(always)] pub fn tan(x: f32) -> f32 { libm::tanf(x) }
}

pub use imp::*;

/// Extension trait to make f32 math methods available uniformly in both std and no_std.
pub trait F32Ext {
    fn floor_(self) -> f32;
    fn ceil_(self) -> f32;
    fn sqrt_(self) -> f32;
    fn abs_(self) -> f32;
    fn min_(self, other: f32) -> f32;
    fn max_(self, other: f32) -> f32;
    fn clamp_(self, lo: f32, hi: f32) -> f32;
    fn sin_(self) -> f32;
    fn cos_(self) -> f32;
    fn tan_(self) -> f32;
}

impl F32Ext for f32 {
    #[inline(always)] fn floor_(self) -> f32 { floor(self) }
    #[inline(always)] fn ceil_(self) -> f32 { ceil(self) }
    #[inline(always)] fn sqrt_(self) -> f32 { sqrt(self) }
    #[inline(always)] fn abs_(self) -> f32 { abs(self) }
    #[inline(always)] fn min_(self, other: f32) -> f32 { min(self, other) }
    #[inline(always)] fn max_(self, other: f32) -> f32 { max(self, other) }
    #[inline(always)] fn clamp_(self, lo: f32, hi: f32) -> f32 { clamp(self, lo, hi) }
    #[inline(always)] fn sin_(self) -> f32 { sin(self) }
    #[inline(always)] fn cos_(self) -> f32 { cos(self) }
    #[inline(always)] fn tan_(self) -> f32 { tan(self) }
}
