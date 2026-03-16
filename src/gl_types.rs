//! OpenGL types, constants, enums, and core data structures for the PortableGL Rust port.

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code)]

use crate::math::{Vec2, Vec4};
use core::ffi::c_void;

#[cfg(feature = "no_std")]
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// GL Type Aliases
// ---------------------------------------------------------------------------

pub type GLboolean = u8;
pub type GLbyte = i8;
pub type GLubyte = u8;
pub type GLshort = i16;
pub type GLushort = u16;
pub type GLint = i32;
pub type GLuint = u32;
pub type GLint64 = i64;
pub type GLuint64 = u64;
pub type GLsizei = i32;
pub type GLenum = u32;
pub type GLbitfield = u32;
pub type GLintptr = isize;
pub type GLsizeiptr = isize;
pub type GLfloat = f32;
pub type GLclampf = f32;
pub type GLdouble = f64;

// ---------------------------------------------------------------------------
// GL Boolean constants
// ---------------------------------------------------------------------------

pub const GL_FALSE: GLboolean = 0;
pub const GL_TRUE: GLboolean = 1;

// ---------------------------------------------------------------------------
// Sequential GL constants (matching C enum, starting at 0)
// ---------------------------------------------------------------------------

// Errors: 0-5
pub const GL_NO_ERROR: GLenum = 0;
pub const GL_INVALID_ENUM: GLenum = 1;
pub const GL_INVALID_VALUE: GLenum = 2;
pub const GL_INVALID_OPERATION: GLenum = 3;
pub const GL_INVALID_FRAMEBUFFER_OPERATION: GLenum = 4;
pub const GL_OUT_OF_MEMORY: GLenum = 5;

// Buffer types: 6-15
pub const GL_ARRAY_BUFFER: GLenum = 6;
pub const GL_COPY_READ_BUFFER: GLenum = 7;
pub const GL_COPY_WRITE_BUFFER: GLenum = 8;
pub const GL_ELEMENT_ARRAY_BUFFER: GLenum = 9;
pub const GL_PIXEL_PACK_BUFFER: GLenum = 10;
pub const GL_PIXEL_UNPACK_BUFFER: GLenum = 11;
pub const GL_TEXTURE_BUFFER: GLenum = 12;
pub const GL_TRANSFORM_FEEDBACK_BUFFER: GLenum = 13;
pub const GL_UNIFORM_BUFFER: GLenum = 14;
pub const GL_NUM_BUFFER_TYPES: GLenum = 15;

// Framebuffer targets: 16-18
pub const GL_FRAMEBUFFER: GLenum = 16;
pub const GL_DRAW_FRAMEBUFFER: GLenum = 17;
pub const GL_READ_FRAMEBUFFER: GLenum = 18;

// Color attachments: 19-26
pub const GL_COLOR_ATTACHMENT0: GLenum = 19;
pub const GL_COLOR_ATTACHMENT1: GLenum = 20;
pub const GL_COLOR_ATTACHMENT2: GLenum = 21;
pub const GL_COLOR_ATTACHMENT3: GLenum = 22;
pub const GL_COLOR_ATTACHMENT4: GLenum = 23;
pub const GL_COLOR_ATTACHMENT5: GLenum = 24;
pub const GL_COLOR_ATTACHMENT6: GLenum = 25;
pub const GL_COLOR_ATTACHMENT7: GLenum = 26;

// Other attachments: 27-29
pub const GL_DEPTH_ATTACHMENT: GLenum = 27;
pub const GL_STENCIL_ATTACHMENT: GLenum = 28;
pub const GL_DEPTH_STENCIL_ATTACHMENT: GLenum = 29;

// Renderbuffer: 30
pub const GL_RENDERBUFFER: GLenum = 30;

// Buffer usage: 31-39
pub const GL_STREAM_DRAW: GLenum = 31;
pub const GL_STREAM_READ: GLenum = 32;
pub const GL_STREAM_COPY: GLenum = 33;
pub const GL_STATIC_DRAW: GLenum = 34;
pub const GL_STATIC_READ: GLenum = 35;
pub const GL_STATIC_COPY: GLenum = 36;
pub const GL_DYNAMIC_DRAW: GLenum = 37;
pub const GL_DYNAMIC_READ: GLenum = 38;
pub const GL_DYNAMIC_COPY: GLenum = 39;

// Access modes: 40-42
pub const GL_READ_ONLY: GLenum = 40;
pub const GL_WRITE_ONLY: GLenum = 41;
pub const GL_READ_WRITE: GLenum = 42;

// Polygon modes: 43-45
pub const GL_POINT: GLenum = 43;
pub const GL_LINE: GLenum = 44;
pub const GL_FILL: GLenum = 45;

// Primitive types: 46-52
pub const GL_POINTS: GLenum = 46;
pub const GL_LINES: GLenum = 47;
pub const GL_LINE_STRIP: GLenum = 48;
pub const GL_LINE_LOOP: GLenum = 49;
pub const GL_TRIANGLES: GLenum = 50;
pub const GL_TRIANGLE_STRIP: GLenum = 51;
pub const GL_TRIANGLE_FAN: GLenum = 52;

// Adjacency primitives: 53-56
pub const GL_LINE_STRIP_ADJACENCY: GLenum = 53;
pub const GL_LINES_ADJACENCY: GLenum = 54;
pub const GL_TRIANGLES_ADJACENCY: GLenum = 55;
pub const GL_TRIANGLE_STRIP_ADJACENCY: GLenum = 56;

// Comparison functions: 57-64
pub const GL_LESS: GLenum = 57;
pub const GL_LEQUAL: GLenum = 58;
pub const GL_GREATER: GLenum = 59;
pub const GL_GEQUAL: GLenum = 60;
pub const GL_EQUAL: GLenum = 61;
pub const GL_NOTEQUAL: GLenum = 62;
pub const GL_ALWAYS: GLenum = 63;
pub const GL_NEVER: GLenum = 64;

// Blend functions: 65-83
pub const GL_ZERO: GLenum = 65;
pub const GL_ONE: GLenum = 66;
pub const GL_SRC_COLOR: GLenum = 67;
pub const GL_ONE_MINUS_SRC_COLOR: GLenum = 68;
pub const GL_DST_COLOR: GLenum = 69;
pub const GL_ONE_MINUS_DST_COLOR: GLenum = 70;
pub const GL_SRC_ALPHA: GLenum = 71;
pub const GL_ONE_MINUS_SRC_ALPHA: GLenum = 72;
pub const GL_DST_ALPHA: GLenum = 73;
pub const GL_ONE_MINUS_DST_ALPHA: GLenum = 74;
pub const GL_CONSTANT_COLOR: GLenum = 75;
pub const GL_ONE_MINUS_CONSTANT_COLOR: GLenum = 76;
pub const GL_CONSTANT_ALPHA: GLenum = 77;
pub const GL_ONE_MINUS_CONSTANT_ALPHA: GLenum = 78;
pub const GL_SRC_ALPHA_SATURATE: GLenum = 79;
pub const NUM_BLEND_FUNCS: GLenum = 80;
pub const GL_SRC1_COLOR: GLenum = 81;
pub const GL_ONE_MINUS_SRC1_COLOR: GLenum = 82;
pub const GL_SRC1_ALPHA: GLenum = 83;
pub const GL_ONE_MINUS_SRC1_ALPHA: GLenum = 84;

// Blend equations: 85-90
pub const GL_FUNC_ADD: GLenum = 85;
pub const GL_FUNC_SUBTRACT: GLenum = 86;
pub const GL_FUNC_REVERSE_SUBTRACT: GLenum = 87;
pub const GL_MIN: GLenum = 88;
pub const GL_MAX: GLenum = 89;
pub const NUM_BLEND_EQUATIONS: GLenum = 90;

// Texture types: 91-99
pub const GL_TEXTURE_UNBOUND: GLenum = 91;
pub const GL_TEXTURE_1D: GLenum = 92;
pub const GL_TEXTURE_2D: GLenum = 93;
pub const GL_TEXTURE_3D: GLenum = 94;
pub const GL_TEXTURE_1D_ARRAY: GLenum = 95;
pub const GL_TEXTURE_2D_ARRAY: GLenum = 96;
pub const GL_TEXTURE_RECTANGLE: GLenum = 97;
pub const GL_TEXTURE_CUBE_MAP: GLenum = 98;
pub const GL_NUM_TEXTURE_TYPES: GLenum = 99;

// Cube map faces: 100-105
pub const GL_TEXTURE_CUBE_MAP_POSITIVE_X: GLenum = 100;
pub const GL_TEXTURE_CUBE_MAP_NEGATIVE_X: GLenum = 101;
pub const GL_TEXTURE_CUBE_MAP_POSITIVE_Y: GLenum = 102;
pub const GL_TEXTURE_CUBE_MAP_NEGATIVE_Y: GLenum = 103;
pub const GL_TEXTURE_CUBE_MAP_POSITIVE_Z: GLenum = 104;
pub const GL_TEXTURE_CUBE_MAP_NEGATIVE_Z: GLenum = 105;

// Texture parameters: 106-122
pub const GL_TEXTURE_BASE_LEVEL: GLenum = 106;
pub const GL_TEXTURE_BORDER_COLOR: GLenum = 107;
pub const GL_TEXTURE_COMPARE_FUNC: GLenum = 108;
pub const GL_TEXTURE_COMPARE_MODE: GLenum = 109;
pub const GL_TEXTURE_LOD_BIAS: GLenum = 110;
pub const GL_TEXTURE_MIN_FILTER: GLenum = 111;
pub const GL_TEXTURE_MAG_FILTER: GLenum = 112;
pub const GL_TEXTURE_MIN_LOD: GLenum = 113;
pub const GL_TEXTURE_MAX_LOD: GLenum = 114;
pub const GL_TEXTURE_MAX_LEVEL: GLenum = 115;
pub const GL_TEXTURE_SWIZZLE_R: GLenum = 116;
pub const GL_TEXTURE_SWIZZLE_G: GLenum = 117;
pub const GL_TEXTURE_SWIZZLE_B: GLenum = 118;
pub const GL_TEXTURE_SWIZZLE_A: GLenum = 119;
pub const GL_TEXTURE_SWIZZLE_RGBA: GLenum = 120;
pub const GL_TEXTURE_WRAP_S: GLenum = 121;
pub const GL_TEXTURE_WRAP_T: GLenum = 122;
pub const GL_TEXTURE_WRAP_R: GLenum = 123;

// Texture wrap/filter modes: 124-133
pub const GL_REPEAT: GLenum = 124;
pub const GL_CLAMP_TO_EDGE: GLenum = 125;
pub const GL_CLAMP_TO_BORDER: GLenum = 126;
pub const GL_MIRRORED_REPEAT: GLenum = 127;
pub const GL_NEAREST: GLenum = 128;
pub const GL_LINEAR: GLenum = 129;
pub const GL_NEAREST_MIPMAP_NEAREST: GLenum = 130;
pub const GL_NEAREST_MIPMAP_LINEAR: GLenum = 131;
pub const GL_LINEAR_MIPMAP_NEAREST: GLenum = 132;
pub const GL_LINEAR_MIPMAP_LINEAR: GLenum = 133;

// Pixel formats: 134-147
pub const PGL_ONE_ALPHA: GLenum = 134;
pub const GL_ALPHA: GLenum = 135;
pub const GL_LUMINANCE: GLenum = 136;
pub const GL_LUMINANCE_ALPHA: GLenum = 137;

// Color formats: 138-147
pub const GL_RED: GLenum = 138;
pub const GL_RG: GLenum = 139;
pub const GL_RGB: GLenum = 140;
pub const GL_BGR: GLenum = 141;
pub const GL_RGBA: GLenum = 142;
pub const GL_BGRA: GLenum = 143;
pub const GL_COMPRESSED_RED: GLenum = 144;
pub const GL_COMPRESSED_RG: GLenum = 145;
pub const GL_COMPRESSED_RGB: GLenum = 146;
pub const GL_COMPRESSED_RGBA: GLenum = 147;

// Depth/stencil formats: 148-157
pub const GL_DEPTH_COMPONENT16: GLenum = 148;
pub const GL_DEPTH_COMPONENT24: GLenum = 149;
pub const GL_DEPTH_COMPONENT32: GLenum = 150;
pub const GL_DEPTH_COMPONENT32F: GLenum = 151;
pub const GL_DEPTH24_STENCIL8: GLenum = 152;
pub const GL_DEPTH32F_STENCIL8: GLenum = 153;
pub const GL_STENCIL_INDEX1: GLenum = 154;
pub const GL_STENCIL_INDEX4: GLenum = 155;
pub const GL_STENCIL_INDEX8: GLenum = 156;
pub const GL_STENCIL_INDEX16: GLenum = 157;

// Pixel store: 158-159
pub const GL_UNPACK_ALIGNMENT: GLenum = 158;
pub const GL_PACK_ALIGNMENT: GLenum = 159;

// Texture units: 160-167
pub const GL_TEXTURE0: GLenum = 160;
pub const GL_TEXTURE1: GLenum = 161;
pub const GL_TEXTURE2: GLenum = 162;
pub const GL_TEXTURE3: GLenum = 163;
pub const GL_TEXTURE4: GLenum = 164;
pub const GL_TEXTURE5: GLenum = 165;
pub const GL_TEXTURE6: GLenum = 166;
pub const GL_TEXTURE7: GLenum = 167;

// Enable/disable caps: 168-178
pub const GL_CULL_FACE: GLenum = 168;
pub const GL_DEPTH_TEST: GLenum = 169;
pub const GL_DEPTH_CLAMP: GLenum = 170;
pub const GL_LINE_SMOOTH: GLenum = 171;
pub const GL_BLEND: GLenum = 172;
pub const GL_COLOR_LOGIC_OP: GLenum = 173;
pub const GL_POLYGON_OFFSET_POINT: GLenum = 174;
pub const GL_POLYGON_OFFSET_LINE: GLenum = 175;
pub const GL_POLYGON_OFFSET_FILL: GLenum = 176;
pub const GL_SCISSOR_TEST: GLenum = 177;
pub const GL_STENCIL_TEST: GLenum = 178;

// Provoking vertex: 179-180
pub const GL_FIRST_VERTEX_CONVENTION: GLenum = 179;
pub const GL_LAST_VERTEX_CONVENTION: GLenum = 180;

// Point sprite: 181-183
pub const GL_POINT_SPRITE_COORD_ORIGIN: GLenum = 181;
pub const GL_UPPER_LEFT: GLenum = 182;
pub const GL_LOWER_LEFT: GLenum = 183;

// Face/winding: 184-188
pub const GL_FRONT: GLenum = 184;
pub const GL_BACK: GLenum = 185;
pub const GL_FRONT_AND_BACK: GLenum = 186;
pub const GL_CCW: GLenum = 187;
pub const GL_CW: GLenum = 188;

// Logic ops: 189-204
pub const GL_CLEAR: GLenum = 189;
pub const GL_SET: GLenum = 190;
pub const GL_COPY: GLenum = 191;
pub const GL_COPY_INVERTED: GLenum = 192;
pub const GL_NOOP: GLenum = 193;
pub const GL_AND: GLenum = 194;
pub const GL_NAND: GLenum = 195;
pub const GL_OR: GLenum = 196;
pub const GL_NOR: GLenum = 197;
pub const GL_XOR: GLenum = 198;
pub const GL_EQUIV: GLenum = 199;
pub const GL_AND_REVERSE: GLenum = 200;
pub const GL_AND_INVERTED: GLenum = 201;
pub const GL_OR_REVERSE: GLenum = 202;
pub const GL_OR_INVERTED: GLenum = 203;
pub const GL_INVERT: GLenum = 204;

// Stencil ops: 205-210
pub const GL_KEEP: GLenum = 205;
pub const GL_REPLACE: GLenum = 206;
pub const GL_INCR: GLenum = 207;
pub const GL_INCR_WRAP: GLenum = 208;
pub const GL_DECR: GLenum = 209;
pub const GL_DECR_WRAP: GLenum = 210;

// Data types: 211-219
pub const GL_UNSIGNED_BYTE: GLenum = 211;
pub const GL_BYTE: GLenum = 212;
pub const GL_UNSIGNED_SHORT: GLenum = 213;
pub const GL_SHORT: GLenum = 214;
pub const GL_UNSIGNED_INT: GLenum = 215;
pub const GL_INT: GLenum = 216;
pub const GL_FLOAT: GLenum = 217;
pub const GL_DOUBLE: GLenum = 218;
pub const GL_BITMAP: GLenum = 219;

// String queries: 220-223
pub const GL_VENDOR: GLenum = 220;
pub const GL_RENDERER: GLenum = 221;
pub const GL_VERSION: GLenum = 222;
pub const GL_SHADING_LANGUAGE_VERSION: GLenum = 223;

// Get parameters: 224-232
pub const GL_POLYGON_OFFSET_FACTOR: GLenum = 224;
pub const GL_POLYGON_OFFSET_UNITS: GLenum = 225;
pub const GL_POINT_SIZE: GLenum = 226;
pub const GL_LINE_WIDTH: GLenum = 227;
pub const GL_ALIASED_LINE_WIDTH_RANGE: GLenum = 228;
pub const GL_SMOOTH_LINE_WIDTH_RANGE: GLenum = 229;
pub const GL_SMOOTH_LINE_WIDTH_GRANULARITY: GLenum = 230;
pub const GL_DEPTH_CLEAR_VALUE: GLenum = 231;
pub const GL_DEPTH_RANGE: GLenum = 232;

// Stencil front state: 233-239
pub const GL_STENCIL_WRITE_MASK: GLenum = 233;
pub const GL_STENCIL_REF: GLenum = 234;
pub const GL_STENCIL_VALUE_MASK: GLenum = 235;
pub const GL_STENCIL_FUNC: GLenum = 236;
pub const GL_STENCIL_FAIL: GLenum = 237;
pub const GL_STENCIL_PASS_DEPTH_FAIL: GLenum = 238;
pub const GL_STENCIL_PASS_DEPTH_PASS: GLenum = 239;

// Stencil back state: 240-246
pub const GL_STENCIL_BACK_WRITE_MASK: GLenum = 240;
pub const GL_STENCIL_BACK_REF: GLenum = 241;
pub const GL_STENCIL_BACK_VALUE_MASK: GLenum = 242;
pub const GL_STENCIL_BACK_FUNC: GLenum = 243;
pub const GL_STENCIL_BACK_FAIL: GLenum = 244;
pub const GL_STENCIL_BACK_PASS_DEPTH_FAIL: GLenum = 245;
pub const GL_STENCIL_BACK_PASS_DEPTH_PASS: GLenum = 246;

// Blend state queries: 247-252
pub const GL_LOGIC_OP_MODE: GLenum = 247;
pub const GL_BLEND_SRC_RGB: GLenum = 248;
pub const GL_BLEND_SRC_ALPHA: GLenum = 249;
pub const GL_BLEND_DST_RGB: GLenum = 250;
pub const GL_BLEND_DST_ALPHA: GLenum = 251;
pub const GL_BLEND_EQUATION_RGB: GLenum = 252;
pub const GL_BLEND_EQUATION_ALPHA: GLenum = 253;

// Face/depth queries: 254-258
pub const GL_CULL_FACE_MODE: GLenum = 254;
pub const GL_FRONT_FACE: GLenum = 255;
pub const GL_DEPTH_FUNC: GLenum = 256;
pub const GL_PROVOKING_VERTEX: GLenum = 257;
pub const GL_POLYGON_MODE: GLenum = 258;

// Version queries: 259-260
pub const GL_MAJOR_VERSION: GLenum = 259;
pub const GL_MINOR_VERSION: GLenum = 260;

// Texture bindings: 261-270
pub const GL_TEXTURE_BINDING_1D: GLenum = 261;
pub const GL_TEXTURE_BINDING_1D_ARRAY: GLenum = 262;
pub const GL_TEXTURE_BINDING_2D: GLenum = 263;
pub const GL_TEXTURE_BINDING_2D_ARRAY: GLenum = 264;
pub const GL_TEXTURE_BINDING_2D_MULTISAMPLE: GLenum = 265;
pub const GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY: GLenum = 266;
pub const GL_TEXTURE_BINDING_3D: GLenum = 267;
pub const GL_TEXTURE_BINDING_BUFFER: GLenum = 268;
pub const GL_TEXTURE_BINDING_CUBE_MAP: GLenum = 269;
pub const GL_TEXTURE_BINDING_RECTANGLE: GLenum = 270;

// Buffer/VAO/program bindings: 271-274
pub const GL_ARRAY_BUFFER_BINDING: GLenum = 271;
pub const GL_ELEMENT_ARRAY_BUFFER_BINDING: GLenum = 272;
pub const GL_VERTEX_ARRAY_BINDING: GLenum = 273;
pub const GL_CURRENT_PROGRAM: GLenum = 274;

// Viewport/scissor: 275-276
pub const GL_VIEWPORT: GLenum = 275;
pub const GL_SCISSOR_BOX: GLenum = 276;

// Max values: 277-282
pub const GL_MAX_TEXTURE_BUFFER_SIZE: GLenum = 277;
pub const GL_MAX_TEXTURE_IMAGE_UNITS: GLenum = 278;
pub const GL_MAX_TEXTURE_LOD_BIAS: GLenum = 279;
pub const GL_MAX_TEXTURE_SIZE: GLenum = 280;
pub const GL_MAX_3D_TEXTURE_SIZE: GLenum = 281;
pub const GL_MAX_ARRAY_TEXTURE_LAYERS: GLenum = 282;

// Debug output: 283
pub const GL_DEBUG_OUTPUT: GLenum = 283;

// Debug sources: 284-289
pub const GL_DEBUG_SOURCE_API: GLenum = 284;
pub const GL_DEBUG_SOURCE_SHADER_COMPILER: GLenum = 285;
pub const GL_DEBUG_SOURCE_WINDOW_SYSTEM: GLenum = 286;
pub const GL_DEBUG_SOURCE_THIRD_PARTY: GLenum = 287;
pub const GL_DEBUG_SOURCE_APPLICATION: GLenum = 288;
pub const GL_DEBUG_SOURCE_OTHER: GLenum = 289;

// Debug types: 290-298
pub const GL_DEBUG_TYPE_ERROR: GLenum = 290;
pub const GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: GLenum = 291;
pub const GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: GLenum = 292;
pub const GL_DEBUG_TYPE_PERFORMANCE: GLenum = 293;
pub const GL_DEBUG_TYPE_PORTABILITY: GLenum = 294;
pub const GL_DEBUG_TYPE_MARKER: GLenum = 295;
pub const GL_DEBUG_TYPE_PUSH_GROUP: GLenum = 296;
pub const GL_DEBUG_TYPE_POP_GROUP: GLenum = 297;
pub const GL_DEBUG_TYPE_OTHER: GLenum = 298;

// Debug severity: 299-302
pub const GL_DEBUG_SEVERITY_HIGH: GLenum = 299;
pub const GL_DEBUG_SEVERITY_MEDIUM: GLenum = 300;
pub const GL_DEBUG_SEVERITY_LOW: GLenum = 301;
pub const GL_DEBUG_SEVERITY_NOTIFICATION: GLenum = 302;

// Max debug message length: 303
pub const GL_MAX_DEBUG_MESSAGE_LENGTH: GLenum = 303;

// Shader types: 304-309
pub const GL_COMPUTE_SHADER: GLenum = 304;
pub const GL_VERTEX_SHADER: GLenum = 305;
pub const GL_TESS_CONTROL_SHADER: GLenum = 306;
pub const GL_TESS_EVALUATION_SHADER: GLenum = 307;
pub const GL_GEOMETRY_SHADER: GLenum = 308;
pub const GL_FRAGMENT_SHADER: GLenum = 309;

// Shader/program queries: 310-312
pub const GL_INFO_LOG_LENGTH: GLenum = 310;
pub const GL_COMPILE_STATUS: GLenum = 311;
pub const GL_LINK_STATUS: GLenum = 312;

// ---------------------------------------------------------------------------
// Bit-flag constants (not part of the sequential enum)
// ---------------------------------------------------------------------------

pub const GL_COLOR_BUFFER_BIT: u32 = 1 << 10;
pub const GL_DEPTH_BUFFER_BIT: u32 = 1 << 11;
pub const GL_STENCIL_BUFFER_BIT: u32 = 1 << 12;

// ---------------------------------------------------------------------------
// Stencil bit constants
// ---------------------------------------------------------------------------

pub const GL_STENCIL_BITS: u32 = 8;
pub const PGL_STENCIL_MASK: u32 = 0xFF;

// ---------------------------------------------------------------------------
// Implementation limits
// ---------------------------------------------------------------------------

pub const GL_MAX_VERTEX_ATTRIBS: usize = 8;
pub const PGL_MAX_VERTICES: usize = 500000;
pub const GL_MAX_VERTEX_OUTPUT_COMPONENTS: usize = 4 * GL_MAX_VERTEX_ATTRIBS;
pub const GL_MAX_DRAW_BUFFERS: usize = 4;
pub const GL_MAX_COLOR_ATTACHMENTS: usize = 4;
pub const PGL_MAX_ALIASED_WIDTH: f32 = 2048.0;
pub const PGL_MAX_TEXTURE_SIZE: i32 = 16384;
pub const PGL_MAX_3D_TEXTURE_SIZE: i32 = 8192;
pub const PGL_MAX_ARRAY_TEXTURE_LAYERS: i32 = 8192;
pub const PGL_MAX_DEBUG_MESSAGE_LENGTH: usize = 256;
pub const PGL_MAX_SMOOTH_WIDTH: f32 = 1.0;
pub const PGL_SMOOTH_GRANULARITY: f32 = 1.0;

// ---------------------------------------------------------------------------
// Interpolation modes
// ---------------------------------------------------------------------------

pub const PGL_SMOOTH: GLenum = 0;
pub const PGL_FLAT: GLenum = 1;
pub const PGL_NOPERSPECTIVE: GLenum = 2;

// ---------------------------------------------------------------------------
// Shader function types
// ---------------------------------------------------------------------------

/// Built-in shader variables that mirror GLSL built-ins.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct ShaderBuiltins {
    // vertex inputs
    pub gl_InstanceID: GLint,
    pub gl_BaseInstance: GLint,
    // vertex outputs
    pub gl_Position: Vec4,
    // fragment inputs
    pub gl_FragCoord: Vec4,
    pub gl_PointCoord: Vec2,
    pub gl_FrontFacing: bool,
    // fragment outputs
    pub gl_FragColor: Vec4,
    pub gl_FragDepth: f32,
    pub discard: bool,
}

impl Default for ShaderBuiltins {
    fn default() -> Self {
        Self {
            gl_InstanceID: 0,
            gl_BaseInstance: 0,
            gl_Position: Vec4::default(),
            gl_FragCoord: Vec4::default(),
            gl_PointCoord: Vec2::default(),
            gl_FrontFacing: false,
            gl_FragColor: Vec4::default(),
            gl_FragDepth: 0.0,
            discard: false,
        }
    }
}

/// Vertex shader function signature (C-compatible).
pub type VertFunc = unsafe extern "C" fn(
    vs_output: *mut f32,
    vertex_attribs: *mut Vec4,
    builtins: *mut ShaderBuiltins,
    uniforms: *mut c_void,
);

/// Fragment shader function signature (C-compatible).
pub type FragFunc = unsafe extern "C" fn(
    fs_input: *mut f32,
    builtins: *mut ShaderBuiltins,
    uniforms: *mut c_void,
);

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// A compiled and linked shader program.
#[derive(Clone)]
pub struct GlProgram {
    pub vertex_shader: VertFunc,
    pub fragment_shader: FragFunc,
    pub uniform: *mut c_void,
    pub vs_output_size: GLsizei,
    pub interpolation: [GLenum; GL_MAX_VERTEX_OUTPUT_COMPONENTS],
    pub fragdepth_or_discard: bool,
    pub deleted: bool,
}

// Raw pointer fields require manual Send/Sync — same safety guarantees as C.
unsafe impl Send for GlProgram {}
unsafe impl Sync for GlProgram {}

/// Default no-op vertex shader used as placeholder.
unsafe extern "C" fn default_vert_shader(
    _vs_output: *mut f32,
    _vertex_attribs: *mut Vec4,
    _builtins: *mut ShaderBuiltins,
    _uniforms: *mut c_void,
) {
}

/// Default no-op fragment shader used as placeholder.
unsafe extern "C" fn default_frag_shader(
    _fs_input: *mut f32,
    _builtins: *mut ShaderBuiltins,
    _uniforms: *mut c_void,
) {
}

impl Default for GlProgram {
    fn default() -> Self {
        Self {
            vertex_shader: default_vert_shader,
            fragment_shader: default_frag_shader,
            uniform: core::ptr::null_mut(),
            vs_output_size: 0,
            interpolation: [PGL_SMOOTH; GL_MAX_VERTEX_OUTPUT_COMPONENTS],
            fragdepth_or_discard: false,
            deleted: false,
        }
    }
}

/// A GPU buffer object.
#[derive(Clone, Debug)]
pub struct GlBuffer {
    pub size: GLsizei,
    pub type_: GLenum,
    pub data: Vec<u8>,
    /// Raw pointer to user-owned data (set by pgl_buffer_data with own=false).
    /// When non-null, this takes priority over `data` for reading.
    pub user_data: *mut u8,
    pub deleted: bool,
    pub user_owned: bool,
}

impl Default for GlBuffer {
    fn default() -> Self {
        Self {
            size: 0,
            type_: 0,
            data: Vec::new(),
            user_data: core::ptr::null_mut(),
            deleted: false,
            user_owned: false,
        }
    }
}

/// A single vertex attribute description.
#[derive(Clone, Copy, Debug)]
pub struct GlVertexAttrib {
    pub size: GLint,
    pub type_: GLenum,
    pub stride: GLsizei,
    pub offset: GLsizeiptr,
    pub normalized: bool,
    pub buf: GLuint,
    pub enabled: bool,
    pub divisor: GLuint,
}

impl Default for GlVertexAttrib {
    fn default() -> Self {
        Self {
            size: 4,
            type_: GL_FLOAT,
            stride: 0,
            offset: 0,
            normalized: false,
            buf: 0,
            enabled: false,
            divisor: 0,
        }
    }
}

/// A vertex array object (VAO).
#[derive(Clone, Debug)]
pub struct GlVertexArray {
    pub vertex_attribs: [GlVertexAttrib; GL_MAX_VERTEX_ATTRIBS],
    pub element_buffer: GLuint,
    pub deleted: bool,
}

impl Default for GlVertexArray {
    fn default() -> Self {
        Self {
            vertex_attribs: core::array::from_fn(|_| GlVertexAttrib::default()),
            element_buffer: 0,
            deleted: false,
        }
    }
}

/// A texture object.
#[derive(Clone, Debug)]
pub struct GlTexture {
    pub w: GLsizei,
    pub h: GLsizei,
    pub d: GLsizei,
    pub mag_filter: GLenum,
    pub min_filter: GLenum,
    pub wrap_s: GLenum,
    pub wrap_t: GLenum,
    pub wrap_r: GLenum,
    pub format: GLenum,
    pub type_: GLenum,
    pub deleted: bool,
    pub user_owned: bool,
    pub border_color: Vec4,
    pub data: Vec<u8>,
}

impl Default for GlTexture {
    fn default() -> Self {
        Self {
            w: 0,
            h: 0,
            d: 0,
            mag_filter: GL_NEAREST,
            min_filter: GL_NEAREST,
            wrap_s: GL_REPEAT,
            wrap_t: GL_REPEAT,
            wrap_r: GL_REPEAT,
            format: GL_RGBA,
            type_: GL_TEXTURE_UNBOUND,
            deleted: false,
            user_owned: false,
            border_color: Vec4::new(0.0, 0.0, 0.0, 0.0),
            data: Vec::new(),
        }
    }
}

/// A transformed vertex with clip/screen space coordinates and shader outputs.
#[derive(Clone, Copy, Debug)]
pub struct GlVertex {
    pub clip_space: Vec4,
    pub screen_space: Vec4,
    pub clip_code: i32,
    pub edge_flag: i32,
    pub vs_out: [f32; GL_MAX_VERTEX_OUTPUT_COMPONENTS],
}

impl Default for GlVertex {
    fn default() -> Self {
        Self {
            clip_space: Vec4::default(),
            screen_space: Vec4::default(),
            clip_code: 0,
            edge_flag: 0,
            vs_out: [0.0; GL_MAX_VERTEX_OUTPUT_COMPONENTS],
        }
    }
}

impl GlVertex {
    /// Returns a mutable slice into the vertex shader output starting at `offset`
    /// with length `size`.
    pub fn vs_out_slice_mut(&mut self, offset: usize, size: usize) -> &mut [f32] {
        &mut self.vs_out[offset..offset + size]
    }

    /// Returns an immutable slice into the vertex shader output starting at `offset`
    /// with length `size`.
    pub fn vs_out_slice(&self, offset: usize, size: usize) -> &[f32] {
        &self.vs_out[offset..offset + size]
    }
}

/// A framebuffer with raw pixel storage.
#[derive(Clone, Debug, Default)]
pub struct GlFramebuffer {
    pub buf: Vec<u8>,
    pub w: GLsizei,
    pub h: GLsizei,
}

/// Vertex shader output descriptor and buffer.
#[derive(Clone, Debug)]
pub struct VertexShaderOutput {
    pub size: GLsizei,
    pub interpolation: *const GLenum,
    pub output_buf: Vec<f32>,
}

unsafe impl Send for VertexShaderOutput {}
unsafe impl Sync for VertexShaderOutput {}

impl Default for VertexShaderOutput {
    fn default() -> Self {
        Self {
            size: 0,
            interpolation: core::ptr::null(),
            output_buf: Vec::new(),
        }
    }
}
