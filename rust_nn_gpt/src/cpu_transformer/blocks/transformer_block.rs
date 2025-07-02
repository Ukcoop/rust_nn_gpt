use super::{FeedForward, LayerNorm, MultiHeadAttention};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
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
        // Standard transformer block: residual + norm + attn + residual + norm + ff
        let mut out = Vec::with_capacity(x.len());
        for row in x.iter() {
            let normed = self.norm1.forward(row);
            let attn_out = self.attn.forward(&[normed]);
            let attn_res = row
                .iter()
                .zip(attn_out[0].iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>();
            let normed2 = self.norm2.forward(&attn_res);
            let ff_out = self.ff.forward(&normed2);
            let ff_res = attn_res
                .iter()
                .zip(ff_out.iter())
                .map(|(a, b)| a + b)
                .collect();
            out.push(ff_res);
        }
        out
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
        // Backward through second residual (ff)
        let grad_ff = grad_output;
        let grad_ff_out = self.ff.backward(grad_ff)?;
        // Residual connection: grad flows to both ff_out and attn_res
        let grad_attn_res: Vec<f32> = grad_output
            .iter()
            .zip(grad_ff_out.iter())
            .map(|(g1, g2)| g1 + g2)
            .collect();
        let grad_norm2 = self.norm2.backward(&grad_attn_res)?;
        // Backward through first residual (attn)
        let grad_attn_out = self.attn.backward(&grad_norm2)?;
        let grad_input: Vec<f32> = grad_norm2
            .iter()
            .zip(grad_attn_out.iter())
            .map(|(g1, g2)| g1 + g2)
            .collect();
        let grad_norm1 = self.norm1.backward(&grad_input)?;
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
        let normed = self.norm1.forward(x);
        let attn_out = self.attn.forward(&[normed]);
        let attn_res: Vec<f32> = x
            .iter()
            .zip(attn_out[0].iter())
            .map(|(a, b)| a + b)
            .collect();
        let normed2 = self.norm2.forward(&attn_res);
        let ff_out = self.ff.forward(&normed2);
        attn_res
            .iter()
            .zip(ff_out.iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    /// Backward for a single vector (for vector-to-vector transformer)
    pub fn backward_vec(
        &mut self,
        grad_output: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Backward through second residual (ff)
        let grad_ff = grad_output;
        let grad_ff_out = self.ff.backward(grad_ff)?;
        // Residual connection: grad flows to both ff_out and attn_res
        let grad_attn_res: Vec<f32> = grad_output
            .iter()
            .zip(grad_ff_out.iter())
            .map(|(g1, g2)| g1 + g2)
            .collect();
        let grad_norm2 = self.norm2.backward(&grad_attn_res)?;
        // Backward through first residual (attn)
        let grad_attn_out = self.attn.backward(&grad_norm2)?;
        let grad_input: Vec<f32> = grad_norm2
            .iter()
            .zip(grad_attn_out.iter())
            .map(|(g1, g2)| g1 + g2)
            .collect();
        let grad_norm1 = self.norm1.backward(&grad_input)?;
        Ok(grad_norm1)
    }
}
