use super::{FeedForward, LayerNorm, MultiHeadAttention};

pub struct TransformerBlock {
    pub attn: MultiHeadAttention,
    pub norm1: LayerNorm,
    pub ff: FeedForward,
    pub norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(model_dim: usize, num_heads: usize, ff_dim: usize) -> Self {
        TransformerBlock {
            attn: MultiHeadAttention::new(model_dim, num_heads),
            norm1: LayerNorm::new(model_dim),
            ff: FeedForward::new(model_dim, ff_dim),
            norm2: LayerNorm::new(model_dim),
        }
    }

    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Bypass attention block
        let x1: Vec<Vec<f32>> = x.to_vec();
        let x1: Vec<Vec<f32>> = x1.iter().map(|row| self.norm1.forward(row)).collect();
        // Feed-forward without residual + norm
        let ff_out: Vec<Vec<f32>> = x1.iter().map(|row| self.ff.forward(row)).collect();
        ff_out.iter().map(|row| self.norm2.forward(row)).collect()
    }

    pub fn zero_grad(&mut self) {
        self.attn.zero_grad();
        self.norm1.zero_grad();
        self.ff.zero_grad();
        self.norm2.zero_grad();
    }

    pub fn apply_gradients(&mut self, lr: f32, scale: f32) {
        self.ff.apply_gradients(lr, scale);
        self.norm2.apply_gradients(lr, scale);
        self.attn.apply_gradients(lr, scale);
        self.norm1.apply_gradients(lr, scale);
    }

    pub fn apply_gradients_adam(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, scale: f32) {
        self.attn.apply_gradients_adam(lr, beta1, beta2, eps, scale);
        self.norm1
            .apply_gradients_adam(lr, beta1, beta2, eps, scale);
        self.ff.apply_gradients_adam(lr, beta1, beta2, eps, scale);
        self.norm2
            .apply_gradients_adam(lr, beta1, beta2, eps, scale);
    }

    /// Backward for a single sample: grad_output is [model_dim], returns grad_input [model_dim]
    pub fn backward(
        &mut self,
        grad_output: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let grad_norm2 = self.norm2.backward(grad_output)?;
        let grad_ff = self.ff.backward(&grad_norm2)?;
        let grad_norm1 = self.norm1.backward(&grad_ff)?;
        Ok(grad_norm1)
    }

    pub fn clip_gradients(&mut self, max_norm: f32) {
        self.attn.clip_gradients(max_norm);
        self.ff.clip_gradients(max_norm);
        self.norm2.clip_gradients(max_norm);
        self.norm1.clip_gradients(max_norm);
    }

    /// Forward for a single vector (for vector-to-vector transformer)
    pub fn forward_vec(&mut self, x: &[f32]) -> Vec<f32> {
        let x1 = self.norm1.forward(x);
        let ff_out = self.ff.forward(&x1);
        self.norm2.forward(&ff_out)
    }

    /// Backward for a single vector (for vector-to-vector transformer)
    pub fn backward_vec(
        &mut self,
        grad_output: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let grad_norm2 = self.norm2.backward(grad_output)?;
        let grad_ff = self.ff.backward(&grad_norm2)?;
        let grad_norm1 = self.norm1.backward(&grad_ff)?;
        Ok(grad_norm1)
    }
}
