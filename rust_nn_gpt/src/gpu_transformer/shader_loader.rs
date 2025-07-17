use crate::gpu_transformer::embedded_shaders;

/// Loads a compiled SPIR-V shader from embedded data
pub fn load_shader(shader_name: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    match embedded_shaders::get_shader(shader_name) {
        Some(shader_data) => Ok(shader_data.to_vec()),
        None => Err(format!("Shader not found: {shader_name}").into()),
    }
}

/// Available shader names that can be loaded
pub const AVAILABLE_SHADERS: &[&str] = embedded_shaders::AVAILABLE_SHADERS;

/// Loads the linear compute shader
pub fn load_linear_shader() -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    load_shader("linear.comp")
}

/// Loads the batched forward shader
pub fn load_batched_forward_shader() -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    load_shader("linear_batched_forward.comp")
}

/// Loads the batched backward shader
pub fn load_batched_backward_shader() -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    load_shader("linear_batched_backward.comp")
}

/// Loads the Adam update shader
pub fn load_adam_update_shader() -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    load_shader("adam_update.comp")
}
