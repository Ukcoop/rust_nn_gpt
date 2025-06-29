use crate::cpu_utils::{
    compute_hidden_gelu_gradients, compute_wb_gradients, flatten_2d, gelu_backward,
    matvec_mul_add_bias, reshape_to_2d, update_matrix, update_vector,
};
use crate::gelu::gelu;

/// Nonlinear input projection: 1D -> 2D (MLP)
pub struct InputProjection {
    w1: Vec<Vec<f32>>, // [hidden][input_dim]
    b1: Vec<f32>,      // [hidden]
    w2: Vec<Vec<f32>>, // [seq_len * embed_dim][hidden]
    b2: Vec<f32>,      // [seq_len * embed_dim]
    input_dim: usize,
    hidden: usize,
    output_dim: usize, // seq_len * embed_dim
    embed_dim: usize,
}

impl InputProjection {
    pub fn new(config: &crate::cpu_transformer::config::TransformerConfig) -> Self {
        let input_dim = config.input_dim;
        let embed_dim = config.height;
        let seq_len = config.height; // use height for both
        let output_dim = seq_len * embed_dim;
        let hidden = output_dim.max(input_dim); // reasonable default
        let w1 = vec![vec![0.01; input_dim]; hidden];
        let b1 = vec![0.0; hidden];
        let w2 = vec![vec![0.01; hidden]; output_dim];
        let b2 = vec![0.0; output_dim];
        Self {
            w1,
            b1,
            w2,
            b2,
            input_dim,
            hidden,
            output_dim,
            embed_dim,
        }
    }
    /// Matrix-vector multiply with bias for first linear layer
    fn linear1(&self, input: &[f32]) -> Vec<f32> {
        matvec_mul_add_bias(&self.w1, input, &self.b1)
    }
    /// Matrix-vector multiply with bias for second linear layer
    fn linear2(&self, input: &[f32]) -> Vec<f32> {
        matvec_mul_add_bias(&self.w2, input, &self.b2)
    }
    pub fn forward(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let hidden = self.linear1(input);
        let hidden_gelu = hidden.iter().map(|&x| gelu(x)).collect::<Vec<_>>();
        let flat_out = self.linear2(&hidden_gelu);
        reshape_to_2d(&flat_out, self.embed_dim)
    }
    pub fn backward(&mut self, input: &[f32], grad_out: &[Vec<f32>], lr: f32) {
        // Flatten grad_out
        let grad_out_flat = flatten_2d(grad_out);

        // Forward pass (save intermediates)
        let hidden = self.linear1(input);
        let hidden_gelu = hidden.iter().map(|&x| gelu(x)).collect::<Vec<_>>();

        // Gradients for w2, b2
        let (grad_w2, grad_b2) = self.compute_w2_b2_gradients(&grad_out_flat, &hidden_gelu);

        // Gradients for hidden_gelu
        let dloss_dhidden_gelu = self.compute_hidden_gelu_gradients(&grad_out_flat);

        // Gradients for hidden (gelu backward)
        let dloss_dhidden = gelu_backward(&hidden, &dloss_dhidden_gelu);

        // Gradients for w1, b1
        let (grad_w1, grad_b1) = self.compute_w1_b1_gradients(&dloss_dhidden, input);

        // Update parameters
        update_matrix(&mut self.w2, &grad_w2, lr);
        update_vector(&mut self.b2, &grad_b2, lr);
        update_matrix(&mut self.w1, &grad_w1, lr);
        update_vector(&mut self.b1, &grad_b1, lr);
    }

    // --- Private helper functions ---
    fn compute_w2_b2_gradients(
        &self,
        grad_out_flat: &[f32],
        hidden_gelu: &[f32],
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        compute_wb_gradients(grad_out_flat, hidden_gelu, self.output_dim, self.hidden)
    }

    fn compute_hidden_gelu_gradients(&self, grad_out_flat: &[f32]) -> Vec<f32> {
        compute_hidden_gelu_gradients(grad_out_flat, &self.w2, self.hidden, self.output_dim)
    }

    fn compute_w1_b1_gradients(
        &self,
        dloss_dhidden: &[f32],
        input: &[f32],
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        compute_wb_gradients(dloss_dhidden, input, self.hidden, self.input_dim)
    }
}
