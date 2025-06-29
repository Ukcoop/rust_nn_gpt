use crate::cpu_utils::{
    compute_hidden_gelu_gradients, compute_wb_gradients, flatten_2d, gelu_backward,
    matvec_mul_add_bias, update_matrix, update_vector,
};
use crate::gelu::gelu;

/// Nonlinear output projection: 2D -> 1D (MLP)
pub struct OutputProjection {
    w1: Vec<Vec<f32>>, // [hidden][seq_len * embed_dim]
    b1: Vec<f32>,      // [hidden]
    w2: Vec<Vec<f32>>, // [output_dim][hidden]
    b2: Vec<f32>,      // [output_dim]
    hidden: usize,
    seq_len: usize,
    embed_dim: usize,
    output_dim: usize,
}

impl OutputProjection {
    pub fn new(config: &crate::cpu_transformer::config::TransformerConfig) -> Self {
        let seq_len = config.height;
        let embed_dim = config.height;
        let output_dim = config.output_dim;
        let input_dim = seq_len * embed_dim;
        let hidden = input_dim.max(output_dim); // reasonable default
        let w1 = vec![vec![0.01; input_dim]; hidden];
        let b1 = vec![0.0; hidden];
        let w2 = vec![vec![0.01; hidden]; output_dim];
        let b2 = vec![0.0; output_dim];
        Self {
            w1,
            b1,
            w2,
            b2,
            hidden,
            seq_len,
            embed_dim,
            output_dim,
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
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<f32> {
        let flat_in = flatten_2d(input);
        let hidden = self.linear1(&flat_in);
        let hidden_gelu = hidden.iter().map(|&x| gelu(x)).collect::<Vec<_>>();
        self.linear2(&hidden_gelu)
    }
    pub fn backward(&mut self, input: &[Vec<f32>], grad_out: &[f32], lr: f32) {
        let flat_in = flatten_2d(input);
        let hidden = self.linear1(&flat_in);
        let hidden_gelu = hidden.iter().map(|&x| gelu(x)).collect::<Vec<_>>();
        let (grad_w2, grad_b2) = self.compute_w2_b2_gradients(grad_out, &hidden_gelu);
        let dloss_dhidden_gelu = self.compute_hidden_gelu_gradients(grad_out);
        let grad_hidden: Vec<f32> = gelu_backward(&hidden, &dloss_dhidden_gelu);
        let (grad_w1, grad_b1) = self.compute_w1_b1_gradients(&grad_hidden, &flat_in);
        update_matrix(&mut self.w2, &grad_w2, lr);
        update_vector(&mut self.b2, &grad_b2, lr);
        update_matrix(&mut self.w1, &grad_w1, lr);
        update_vector(&mut self.b1, &grad_b1, lr);
    }

    // --- Private helper functions ---
    fn compute_w2_b2_gradients(
        &self,
        grad_out: &[f32],
        hidden_gelu: &[f32],
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        compute_wb_gradients(grad_out, hidden_gelu, self.output_dim, self.hidden)
    }

    fn compute_hidden_gelu_gradients(&self, grad_out: &[f32]) -> Vec<f32> {
        compute_hidden_gelu_gradients(grad_out, &self.w2, self.hidden, self.output_dim)
    }

    fn compute_w1_b1_gradients(
        &self,
        dloss_dhidden: &[f32],
        flat_in: &[f32],
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        compute_wb_gradients(
            dloss_dhidden,
            flat_in,
            self.hidden,
            self.seq_len * self.embed_dim,
        )
    }
}
