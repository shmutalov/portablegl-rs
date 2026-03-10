use crate::math::*;
use crate::gl_types::*;

pub type TriangleFunc = u8; // 0 = fill, 1 = line, 2 = point
pub const TRIANGLE_FILL: TriangleFunc = 0;
pub const TRIANGLE_LINE: TriangleFunc = 1;
pub const TRIANGLE_POINT: TriangleFunc = 2;

pub struct GlContext {
    pub vp_mat: Mat4,

    // viewport
    pub xmin: GLint,
    pub ymin: GLint,
    pub width: GLsizei,
    pub height: GLsizei,

    // scissor/guardband clipping bounds
    pub lx: GLint,
    pub ly: GLint,
    pub ux: GLint,
    pub uy: GLint,

    // object collections
    pub vertex_arrays: Vec<GlVertexArray>,
    pub buffers: Vec<GlBuffer>,
    pub textures: Vec<GlTexture>,
    pub programs: Vec<GlProgram>,

    // default textures (one per texture target type)
    pub default_textures: Vec<GlTexture>,

    // current bindings
    pub cur_vertex_array: GLuint,
    pub bound_buffers: Vec<GLuint>,  // indexed by buffer type - GL_ARRAY_BUFFER
    pub bound_textures: Vec<GLuint>, // indexed by texture type - GL_TEXTURE_UNBOUND - 1
    pub cur_texture2d: GLuint,
    pub cur_program: GLuint,

    // error/debug state
    pub error: GLenum,
    pub dbg_output: bool,

    // vertex processing state
    pub vertex_attribs_vs: [Vec4; GL_MAX_VERTEX_ATTRIBS],
    pub builtins: ShaderBuiltins,
    pub vs_output: VertexShaderOutput,
    pub fs_input: [f32; GL_MAX_VERTEX_OUTPUT_COMPONENTS],

    // rendering state flags
    pub depth_test: bool,
    pub line_smooth: bool,
    pub cull_face: bool,
    pub fragdepth_or_discard: bool,
    pub depth_clamp: bool,
    pub depth_mask: bool,
    pub blend: bool,
    pub logic_ops: bool,
    pub poly_offset_pt: bool,
    pub poly_offset_line: bool,
    pub poly_offset_fill: bool,
    pub scissor_test: bool,

    pub color_mask: u32,

    // stencil state
    pub stencil_test: bool,
    pub stencil_writemask: GLuint,
    pub stencil_writemask_back: GLuint,
    pub stencil_ref: GLint,
    pub stencil_ref_back: GLint,
    pub stencil_valuemask: GLuint,
    pub stencil_valuemask_back: GLuint,
    pub stencil_func: GLenum,
    pub stencil_func_back: GLenum,
    pub stencil_sfail: GLenum,
    pub stencil_dpfail: GLenum,
    pub stencil_dppass: GLenum,
    pub stencil_sfail_back: GLenum,
    pub stencil_dpfail_back: GLenum,
    pub stencil_dppass_back: GLenum,
    pub clear_stencil: GLint,
    pub stencil_buf: GlFramebuffer,

    // blend/logic state
    pub logic_func: GLenum,
    pub blend_srgb: GLenum,
    pub blend_sa: GLenum,
    pub blend_drgb: GLenum,
    pub blend_da: GLenum,
    pub blend_eq_rgb: GLenum,
    pub blend_eq_a: GLenum,
    pub cull_mode: GLenum,
    pub front_face: GLenum,
    pub poly_mode_front: GLenum,
    pub poly_mode_back: GLenum,
    pub depth_func: GLenum,
    pub point_spr_origin: GLenum,
    pub provoking_vert: GLenum,

    pub poly_factor: GLfloat,
    pub poly_units: GLfloat,

    pub scissor_lx: GLint,
    pub scissor_ly: GLint,
    pub scissor_w: GLsizei,
    pub scissor_h: GLsizei,

    pub unpack_alignment: GLint,
    pub pack_alignment: GLint,

    pub clear_color: u32,
    pub blend_color: Vec4,
    pub point_size: GLfloat,
    pub line_width: GLfloat,
    pub clear_depth: GLfloat,
    pub depth_range_near: GLfloat,
    pub depth_range_far: GLfloat,

    // draw mode function indices (0=fill, 1=line, 2=point)
    pub draw_triangle_front: TriangleFunc,
    pub draw_triangle_back: TriangleFunc,

    // depth buffer
    pub zbuf: GlFramebuffer,
    pub back_buffer: GlFramebuffer,

    pub user_alloced_backbuf: bool,

    // processed vertices
    pub glverts: Vec<GlVertex>,
}

impl Default for GlContext {
    fn default() -> Self {
        Self {
            vp_mat: Mat4::default(),

            xmin: 0,
            ymin: 0,
            width: 0,
            height: 0,

            lx: 0,
            ly: 0,
            ux: 0,
            uy: 0,

            vertex_arrays: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            programs: Vec::new(),

            default_textures: Vec::new(),

            cur_vertex_array: 0,
            bound_buffers: Vec::new(),
            bound_textures: Vec::new(),
            cur_texture2d: 0,
            cur_program: 0,

            error: 0,
            dbg_output: false,

            vertex_attribs_vs: [Vec4::default(); GL_MAX_VERTEX_ATTRIBS],
            builtins: ShaderBuiltins::default(),
            vs_output: VertexShaderOutput::default(),
            fs_input: [0.0; GL_MAX_VERTEX_OUTPUT_COMPONENTS],

            depth_test: false,
            line_smooth: false,
            cull_face: false,
            fragdepth_or_discard: false,
            depth_clamp: false,
            depth_mask: false,
            blend: false,
            logic_ops: false,
            poly_offset_pt: false,
            poly_offset_line: false,
            poly_offset_fill: false,
            scissor_test: false,

            color_mask: 0,

            stencil_test: false,
            stencil_writemask: 0,
            stencil_writemask_back: 0,
            stencil_ref: 0,
            stencil_ref_back: 0,
            stencil_valuemask: 0,
            stencil_valuemask_back: 0,
            stencil_func: 0,
            stencil_func_back: 0,
            stencil_sfail: 0,
            stencil_dpfail: 0,
            stencil_dppass: 0,
            stencil_sfail_back: 0,
            stencil_dpfail_back: 0,
            stencil_dppass_back: 0,
            clear_stencil: 0,
            stencil_buf: GlFramebuffer::default(),

            logic_func: 0,
            blend_srgb: 0,
            blend_sa: 0,
            blend_drgb: 0,
            blend_da: 0,
            blend_eq_rgb: 0,
            blend_eq_a: 0,
            cull_mode: 0,
            front_face: 0,
            poly_mode_front: 0,
            poly_mode_back: 0,
            depth_func: 0,
            point_spr_origin: 0,
            provoking_vert: 0,

            poly_factor: 0.0,
            poly_units: 0.0,

            scissor_lx: 0,
            scissor_ly: 0,
            scissor_w: 0,
            scissor_h: 0,

            unpack_alignment: 0,
            pack_alignment: 0,

            clear_color: 0,
            blend_color: Vec4::default(),
            point_size: 0.0,
            line_width: 0.0,
            clear_depth: 0.0,
            depth_range_near: 0.0,
            depth_range_far: 0.0,

            draw_triangle_front: TRIANGLE_FILL,
            draw_triangle_back: TRIANGLE_FILL,

            zbuf: GlFramebuffer::default(),
            back_buffer: GlFramebuffer::default(),

            user_alloced_backbuf: false,

            glverts: Vec::new(),
        }
    }
}

impl GlContext {
    pub fn new() -> Self {
        Self::default()
    }
}
