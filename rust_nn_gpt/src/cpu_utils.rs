//! Shared CPU neural network utilities for matrix operations and parameter updates.

/// Matrix-vector multiply with bias.
/// mat: [rows][cols], vec: [cols], bias: [rows]
pub fn matvec_mul_add_bias(mat: &[Vec<f32>], vec: &[f32], bias: &[f32]) -> Vec<f32> {
    mat.iter()
        .zip(bias)
        .map(|(w_row, b)| w_row.iter().zip(vec).map(|(w, x)| w * x).sum::<f32>() + b)
        .collect()
}

/// Update a matrix parameter with gradients and learning rate.
pub fn update_matrix(param: &mut [Vec<f32>], grad: &[Vec<f32>], lr: f32) {
    for (p_row, g_row) in param.iter_mut().zip(grad) {
        for (p, g) in p_row.iter_mut().zip(g_row) {
            *p -= lr * g;
        }
    }
}

/// Update a vector parameter with gradients and learning rate.
pub fn update_vector(param: &mut [f32], grad: &[f32], lr: f32) {
    for (p, g) in param.iter_mut().zip(grad) {
        *p -= lr * g;
    }
}

/// Flatten a 2D vector into a 1D vector.
pub fn flatten_2d(input: &[Vec<f32>]) -> Vec<f32> {
    input.iter().flat_map(|v| v.iter()).copied().collect()
}

/// Reshape a flat 1D vector into a 2D vector with the given row size.
pub fn reshape_to_2d(flat: &[f32], row_size: usize) -> Vec<Vec<f32>> {
    flat.chunks(row_size).map(|chunk| chunk.to_vec()).collect()
}

/// Compute gradients for a weight matrix and bias vector given upstream gradient and activation.
/// grad_out: [out_dim], activation: [in_dim], returns (grad_w: [out_dim][in_dim], grad_b: [out_dim])
pub fn compute_wb_gradients(
    grad_out: &[f32],
    activation: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut grad_w = vec![vec![0.0; in_dim]; out_dim];
    let mut grad_b = vec![0.0; out_dim];
    for i in 0..out_dim {
        for j in 0..in_dim {
            grad_w[i][j] = grad_out[i] * activation[j];
        }
        grad_b[i] = grad_out[i];
    }
    (grad_w, grad_b)
}

/// Compute gradients for a hidden layer given upstream gradient and weights.
/// grad_out: [out_dim], weights: [out_dim][hidden], returns [hidden]
pub fn compute_hidden_gelu_gradients(
    grad_out: &[f32],
    weights: &[Vec<f32>],
    hidden: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut dloss_dhidden_gelu = vec![0.0; hidden];
    for j in 0..hidden {
        for i in 0..out_dim {
            dloss_dhidden_gelu[j] += grad_out[i] * weights[i][j];
        }
    }
    dloss_dhidden_gelu
}

/// Apply the GELU derivative to a hidden vector and its upstream gradient.
pub fn gelu_backward(hidden: &[f32], grad: &[f32]) -> Vec<f32> {
    hidden
        .iter()
        .zip(grad)
        .map(|(h, g)| crate::gelu::gelu_derivative(*h) * g)
        .collect()
}
