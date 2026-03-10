#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate alloc;

pub mod float_math;
pub mod math;
pub mod gl_types;
pub mod gl_context;
pub mod gl_impl;
pub mod gl_glsl;
pub mod gl_internal;
pub mod pgl_ext;
