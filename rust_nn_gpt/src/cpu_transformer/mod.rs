pub mod blocks;

use crate::transformer::TransformerConfig;
use blocks::{LayerNorm, Linear, TransformerBlock};

pub struct CpuTransformer {
    pub input_dim: usize,
    pub model_dim: usize,
    pub output_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub input_proj: Linear,
    pub input_norm: LayerNorm,
    pub blocks: Vec<TransformerBlock>,
    pub output_norm: LayerNorm,
    pub output_proj: Linear,
}

impl CpuTransformer {
    pub fn new(config: &TransformerConfig) -> Self {
        let ff_dim = config.model_dim * 4;
        let blocks = (0..config.num_layers)
            .map(|_| TransformerBlock::new(config.model_dim, config.num_heads, ff_dim))
            .collect();
        CpuTransformer {
            input_dim: config.input_dim,
            model_dim: config.model_dim,
            output_dim: config.output_dim,
            num_heads: config.num_heads,
            num_layers: config.num_layers,
            input_proj: Linear::new(config.input_dim, config.model_dim),
            input_norm: LayerNorm::new(config.model_dim),
            blocks,
            output_norm: LayerNorm::new(config.model_dim),
            output_proj: Linear::new(config.model_dim, config.output_dim),
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut x = self.input_proj.forward(input); // [model_dim]
        for block in &mut self.blocks {
            x = block.forward_vec(&x);
        }
        self.output_proj.forward(&x) // [output_dim]
    }

    pub fn train_batch(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let batch_size = inputs.len();
        let lr = 0.01;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let output_dim = self.output_dim;
        let mut total_loss = 0.0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let mut x = self.input_proj.forward(input);
            for block in &mut self.blocks {
                x = block.forward_vec(&x);
            }
            let y = self.output_proj.forward(&x);
            let mut grad_out = vec![0.0; output_dim];
            for i in 0..output_dim {
                let diff = y[i] - target[i];
                total_loss += 0.5 * diff * diff;
                grad_out[i] = diff;
            }
            let mut grad_x = self.output_proj.backward(&x, &grad_out);
            for block in self.blocks.iter_mut().rev() {
                grad_x = block.backward_vec(&grad_x)?;
            }
            self.input_proj.backward(input, &grad_x);
        }
        let scale = 1.0 / (batch_size * output_dim) as f32;
        self.output_proj
            .apply_gradients_adam(lr, beta1, beta2, eps, scale);
        self.input_proj
            .apply_gradients_adam(lr, beta1, beta2, eps, scale);
        for block in &mut self.blocks {
            block.apply_gradients_adam(lr, beta1, beta2, eps, scale);
        }
        Ok(total_loss / batch_size as f32)
    }
}

fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub struct ScaledDotProductAttention;

impl ScaledDotProductAttention {
    pub fn forward(q: &[f32], k: &[f32], v: &[f32]) -> f32 {
        // For 1D case: attention(q, k, v) = softmax(q·k / sqrt(d_k)) * v
        let d_k = q.len() as f32;
        let score = dot(q, k) / d_k.sqrt();
        let attn = softmax(&[score])[0];
        attn * dot(&[1.0], v) // just v for 1D
    }
}
