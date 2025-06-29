use crate::cpu_transformer::config::TransformerConfig;
use crate::cpu_utils::{
    compute_hidden_gelu_gradients, compute_wb_gradients, matvec_mul_add_bias, update_matrix,
    update_vector,
};
use crate::gelu::gelu;

/// Simple FeedForward: two linear layers with ReLU activation
pub struct FeedForward {
    pub w1: Vec<Vec<f32>>, // [ff_dim][embed_dim]
    pub b1: Vec<f32>,      // [ff_dim]
    pub w2: Vec<Vec<f32>>, // [embed_dim][ff_dim]
    pub b2: Vec<f32>,      // [embed_dim]
    pub embed_dim: usize,
    pub ff_dim: usize,
}

impl FeedForward {
    pub fn new(config: &TransformerConfig) -> Self {
        let embed_dim = config.height;
        let ff_dim = config.height;
        let w1 = vec![vec![1.0; embed_dim]; ff_dim];
        let b1 = vec![0.0; ff_dim];
        let w2 = vec![vec![1.0; ff_dim]; embed_dim];
        let b2 = vec![0.0; embed_dim];
        Self {
            w1,
            b1,
            w2,
            b2,
            embed_dim,
            ff_dim,
        }
    }
    /// Forward pass: (input -> Linear -> ReLU -> Linear)
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let hidden = self.linear1(input);
        let hidden_gelu = hidden.iter().map(|&x| gelu(x)).collect::<Vec<_>>();
        self.linear2(&hidden_gelu)
    }
    /// Backward pass for a single input/target pair (MSE loss)
    /// Returns gradients for w1, b1, w2, b2
    pub fn backward(&mut self, input: &[f32], target: &[f32], lr: f32) {
        // Forward pass (save intermediates)
        let hidden = self.linear1(input);
        let hidden_gelu = hidden.iter().map(|&x| gelu(x)).collect::<Vec<_>>();
        let output = self.linear2(&hidden_gelu);

        // Gradients of loss w.r.t. output (MSE)
        let dloss_dout = self.compute_loss_gradients(&output, target);

        // Gradients for w2, b2
        let (grad_w2, grad_b2) = self.compute_w2_b2_gradients(&dloss_dout, &hidden_gelu);

        // Gradients for hidden_gelu
        let dloss_dhidden_gelu = self.compute_hidden_gelu_gradients(&dloss_dout);

        // Gradients for hidden (gelu backward)
        let dloss_dhidden = self.compute_hidden_gradients(&hidden, dloss_dhidden_gelu);

        // Gradients for w1, b1
        let (grad_w1, grad_b1) = self.compute_w1_b1_gradients(&dloss_dhidden, input);

        // Update parameters
        update_matrix(&mut self.w2, &grad_w2, lr);
        update_vector(&mut self.b2, &grad_b2, lr);
        update_matrix(&mut self.w1, &grad_w1, lr);
        update_vector(&mut self.b1, &grad_b1, lr);
    }
    /// Matrix-vector multiply with bias for first linear layer
    fn linear1(&self, input: &[f32]) -> Vec<f32> {
        matvec_mul_add_bias(&self.w1, input, &self.b1)
    }
    /// Matrix-vector multiply with bias for second linear layer
    fn linear2(&self, input: &[f32]) -> Vec<f32> {
        matvec_mul_add_bias(&self.w2, input, &self.b2)
    }
    /// Compute gradients of MSE loss w.r.t. output
    fn compute_loss_gradients(&self, output: &[f32], target: &[f32]) -> Vec<f32> {
        output
            .iter()
            .zip(target)
            .map(|(y, t)| 2.0 * (y - t) / output.len() as f32)
            .collect()
    }
    /// Compute gradients for w2, b2
    fn compute_w2_b2_gradients(
        &self,
        dloss_dout: &[f32],
        hidden_gelu: &[f32],
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        compute_wb_gradients(dloss_dout, hidden_gelu, self.embed_dim, self.ff_dim)
    }
    /// Compute gradients for hidden_gelu
    fn compute_hidden_gelu_gradients(&self, dloss_dout: &[f32]) -> Vec<f32> {
        compute_hidden_gelu_gradients(dloss_dout, &self.w2, self.ff_dim, self.embed_dim)
    }
    /// Compute gradients for hidden (gelu backward)
    fn compute_hidden_gradients(&self, hidden: &[f32], dloss_dhidden_gelu: Vec<f32>) -> Vec<f32> {
        crate::cpu_utils::gelu_backward(hidden, &dloss_dhidden_gelu)
    }
    /// Compute gradients for w1, b1
    fn compute_w1_b1_gradients(
        &self,
        dloss_dhidden: &[f32],
        input: &[f32],
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        compute_wb_gradients(dloss_dhidden, input, self.ff_dim, self.embed_dim)
    }
}
