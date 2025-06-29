//! GELU activation and its derivative, for use in all modules.
//!
//! gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715x^3)))
//! derivative from https://arxiv.org/pdf/1606.08415.pdf

/// Compute the GELU activation for a single value.
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3))).tanh())
}

/// Compute the derivative of GELU for a single value.
pub fn gelu_derivative(x: f32) -> f32 {
    let tanh_arg = std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3));
    let tanh_val = tanh_arg.tanh();
    let left = 0.5 * (1.0 + tanh_val);
    let right = 0.5
        * x
        * (1.0 - tanh_val.powi(2))
        * std::f32::consts::FRAC_2_SQRT_PI
        * (1.0 + 3.0 * 0.044715 * x.powi(2));
    left + right
}
