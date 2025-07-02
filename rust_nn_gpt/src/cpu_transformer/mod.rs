pub mod blocks;

use crate::transformer::TransformerConfig;
use blocks::{LayerNorm, Linear, TransformerBlock};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use crate::transformer_weights::{TransformerWeights, BlockWeights};

#[derive(Serialize, Deserialize)]
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
        x = self.input_norm.forward(&x);
        for block in &mut self.blocks {
            x = block.forward_vec(&x);
        }
        let x = self.output_norm.forward(&x);
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
            x = self.input_norm.forward(&x);
            for block in &mut self.blocks {
                x = block.forward_vec(&x);
            }
            let x = self.output_norm.forward(&x);
            let y = self.output_proj.forward(&x);
            let mut grad_out = vec![0.0; output_dim];
            for i in 0..output_dim {
                let diff = y[i] - target[i];
                total_loss += 0.5 * diff * diff;
                grad_out[i] = diff;
            }
            let grad_x = self.output_proj.backward(&x, &grad_out);
            let mut grad_x = self.output_norm.backward(&grad_x)?;
            for block in self.blocks.iter_mut().rev() {
                grad_x = block.backward_vec(&grad_x)?;
            }
            self.input_norm.backward(&grad_x)?;
            self.input_proj.backward(input, &grad_x);
        }
        let scale = 1.0 / (batch_size * output_dim) as f32;
        self.output_proj
            .apply_gradients_adam(lr, beta1, beta2, eps, scale);
        self.output_norm
            .apply_gradients_adam(lr, beta1, beta2, eps, scale);
        self.input_proj
            .apply_gradients_adam(lr, beta1, beta2, eps, scale);
        self.input_norm
            .apply_gradients_adam(lr, beta1, beta2, eps, scale);
        for block in &mut self.blocks {
            block.apply_gradients_adam(lr, beta1, beta2, eps, scale);
        }
        Ok(total_loss / batch_size as f32)
    }

    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self).map_err(std::io::Error::other)
    }

    pub fn load_json(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(std::io::Error::other)
    }

    pub fn from_weights(weights: &TransformerWeights, config: &TransformerConfig) -> Self {
        let mut cpu = CpuTransformer::new(config);
        cpu.input_proj.weight = weights.input_proj_weight.clone();
        cpu.input_proj.bias = weights.input_proj_bias.clone();
        cpu.input_proj.m_weight = weights.input_proj_m_weight.clone();
        cpu.input_proj.v_weight = weights.input_proj_v_weight.clone();
        cpu.input_proj.m_bias = weights.input_proj_m_bias.clone();
        cpu.input_proj.v_bias = weights.input_proj_v_bias.clone();
        cpu.input_proj.t = weights.input_proj_t;
        cpu.input_norm.gamma = weights.input_norm_gamma.clone();
        cpu.input_norm.beta = weights.input_norm_beta.clone();
        cpu.input_norm.m_gamma = weights.input_norm_m_gamma.clone();
        cpu.input_norm.v_gamma = weights.input_norm_v_gamma.clone();
        cpu.input_norm.m_beta = weights.input_norm_m_beta.clone();
        cpu.input_norm.v_beta = weights.input_norm_v_beta.clone();
        cpu.input_norm.t = weights.input_norm_t;
        for (block, w) in cpu.blocks.iter_mut().zip(weights.blocks.iter()) {
            block.attn.w_q.weight = w.attn_wq_weight.clone();
            block.attn.w_q.bias = w.attn_wq_bias.clone();
            block.attn.w_q.m_weight = w.attn_wq_m_weight.clone();
            block.attn.w_q.v_weight = w.attn_wq_v_weight.clone();
            block.attn.w_q.m_bias = w.attn_wq_m_bias.clone();
            block.attn.w_q.v_bias = w.attn_wq_v_bias.clone();
            block.attn.w_q.t = w.attn_wq_t;
            block.attn.w_k.weight = w.attn_wk_weight.clone();
            block.attn.w_k.bias = w.attn_wk_bias.clone();
            block.attn.w_k.m_weight = w.attn_wk_m_weight.clone();
            block.attn.w_k.v_weight = w.attn_wk_v_weight.clone();
            block.attn.w_k.m_bias = w.attn_wk_m_bias.clone();
            block.attn.w_k.v_bias = w.attn_wk_v_bias.clone();
            block.attn.w_k.t = w.attn_wk_t;
            block.attn.w_v.weight = w.attn_wv_weight.clone();
            block.attn.w_v.bias = w.attn_wv_bias.clone();
            block.attn.w_v.m_weight = w.attn_wv_m_weight.clone();
            block.attn.w_v.v_weight = w.attn_wv_v_weight.clone();
            block.attn.w_v.m_bias = w.attn_wv_m_bias.clone();
            block.attn.w_v.v_bias = w.attn_wv_v_bias.clone();
            block.attn.w_v.t = w.attn_wv_t;
            block.attn.w_o.weight = w.attn_wo_weight.clone();
            block.attn.w_o.bias = w.attn_wo_bias.clone();
            block.attn.w_o.m_weight = w.attn_wo_m_weight.clone();
            block.attn.w_o.v_weight = w.attn_wo_v_weight.clone();
            block.attn.w_o.m_bias = w.attn_wo_m_bias.clone();
            block.attn.w_o.v_bias = w.attn_wo_v_bias.clone();
            block.attn.w_o.t = w.attn_wo_t;
            block.norm1.gamma = w.norm1_gamma.clone();
            block.norm1.beta = w.norm1_beta.clone();
            block.norm1.m_gamma = w.norm1_m_gamma.clone();
            block.norm1.v_gamma = w.norm1_v_gamma.clone();
            block.norm1.m_beta = w.norm1_m_beta.clone();
            block.norm1.v_beta = w.norm1_v_beta.clone();
            block.norm1.t = w.norm1_t;
            block.norm2.gamma = w.norm2_gamma.clone();
            block.norm2.beta = w.norm2_beta.clone();
            block.norm2.m_gamma = w.norm2_m_gamma.clone();
            block.norm2.v_gamma = w.norm2_v_gamma.clone();
            block.norm2.m_beta = w.norm2_m_beta.clone();
            block.norm2.v_beta = w.norm2_v_beta.clone();
            block.norm2.t = w.norm2_t;
            block.ff.linear1.weight = w.ff_linear1_weight.clone();
            block.ff.linear1.bias = w.ff_linear1_bias.clone();
            block.ff.linear1.m_weight = w.ff_linear1_m_weight.clone();
            block.ff.linear1.v_weight = w.ff_linear1_v_weight.clone();
            block.ff.linear1.m_bias = w.ff_linear1_m_bias.clone();
            block.ff.linear1.v_bias = w.ff_linear1_v_bias.clone();
            block.ff.linear1.t = w.ff_linear1_t;
            block.ff.linear2.weight = w.ff_linear2_weight.clone();
            block.ff.linear2.bias = w.ff_linear2_bias.clone();
            block.ff.linear2.m_weight = w.ff_linear2_m_weight.clone();
            block.ff.linear2.v_weight = w.ff_linear2_v_weight.clone();
            block.ff.linear2.m_bias = w.ff_linear2_m_bias.clone();
            block.ff.linear2.v_bias = w.ff_linear2_v_bias.clone();
            block.ff.linear2.t = w.ff_linear2_t;
        }
        cpu.output_norm.gamma = weights.output_norm_gamma.clone();
        cpu.output_norm.beta = weights.output_norm_beta.clone();
        cpu.output_norm.m_gamma = weights.output_norm_m_gamma.clone();
        cpu.output_norm.v_gamma = weights.output_norm_v_gamma.clone();
        cpu.output_norm.m_beta = weights.output_norm_m_beta.clone();
        cpu.output_norm.v_beta = weights.output_norm_v_beta.clone();
        cpu.output_norm.t = weights.output_norm_t;
        cpu.output_proj.weight = weights.output_proj_weight.clone();
        cpu.output_proj.bias = weights.output_proj_bias.clone();
        cpu.output_proj.m_weight = weights.output_proj_m_weight.clone();
        cpu.output_proj.v_weight = weights.output_proj_v_weight.clone();
        cpu.output_proj.m_bias = weights.output_proj_m_bias.clone();
        cpu.output_proj.v_bias = weights.output_proj_v_bias.clone();
        cpu.output_proj.t = weights.output_proj_t;
        cpu
    }
}

impl From<&CpuTransformer> for TransformerWeights {
    fn from(cpu: &CpuTransformer) -> Self {
        TransformerWeights {
            input_proj_weight: cpu.input_proj.weight.clone(),
            input_proj_bias: cpu.input_proj.bias.clone(),
            input_proj_m_weight: cpu.input_proj.m_weight.clone(),
            input_proj_v_weight: cpu.input_proj.v_weight.clone(),
            input_proj_m_bias: cpu.input_proj.m_bias.clone(),
            input_proj_v_bias: cpu.input_proj.v_bias.clone(),
            input_proj_t: cpu.input_proj.t,
            input_norm_gamma: cpu.input_norm.gamma.clone(),
            input_norm_beta: cpu.input_norm.beta.clone(),
            input_norm_m_gamma: cpu.input_norm.m_gamma.clone(),
            input_norm_v_gamma: cpu.input_norm.v_gamma.clone(),
            input_norm_m_beta: cpu.input_norm.m_beta.clone(),
            input_norm_v_beta: cpu.input_norm.v_beta.clone(),
            input_norm_t: cpu.input_norm.t,
            blocks: cpu.blocks.iter().map(|block| {
                BlockWeights {
                    attn_wq_weight: block.attn.w_q.weight.clone(),
                    attn_wq_bias: block.attn.w_q.bias.clone(),
                    attn_wq_m_weight: block.attn.w_q.m_weight.clone(),
                    attn_wq_v_weight: block.attn.w_q.v_weight.clone(),
                    attn_wq_m_bias: block.attn.w_q.m_bias.clone(),
                    attn_wq_v_bias: block.attn.w_q.v_bias.clone(),
                    attn_wq_t: block.attn.w_q.t,
                    attn_wk_weight: block.attn.w_k.weight.clone(),
                    attn_wk_bias: block.attn.w_k.bias.clone(),
                    attn_wk_m_weight: block.attn.w_k.m_weight.clone(),
                    attn_wk_v_weight: block.attn.w_k.v_weight.clone(),
                    attn_wk_m_bias: block.attn.w_k.m_bias.clone(),
                    attn_wk_v_bias: block.attn.w_k.v_bias.clone(),
                    attn_wk_t: block.attn.w_k.t,
                    attn_wv_weight: block.attn.w_v.weight.clone(),
                    attn_wv_bias: block.attn.w_v.bias.clone(),
                    attn_wv_m_weight: block.attn.w_v.m_weight.clone(),
                    attn_wv_v_weight: block.attn.w_v.v_weight.clone(),
                    attn_wv_m_bias: block.attn.w_v.m_bias.clone(),
                    attn_wv_v_bias: block.attn.w_v.v_bias.clone(),
                    attn_wv_t: block.attn.w_v.t,
                    attn_wo_weight: block.attn.w_o.weight.clone(),
                    attn_wo_bias: block.attn.w_o.bias.clone(),
                    attn_wo_m_weight: block.attn.w_o.m_weight.clone(),
                    attn_wo_v_weight: block.attn.w_o.v_weight.clone(),
                    attn_wo_m_bias: block.attn.w_o.m_bias.clone(),
                    attn_wo_v_bias: block.attn.w_o.v_bias.clone(),
                    attn_wo_t: block.attn.w_o.t,
                    norm1_gamma: block.norm1.gamma.clone(),
                    norm1_beta: block.norm1.beta.clone(),
                    norm1_m_gamma: block.norm1.m_gamma.clone(),
                    norm1_v_gamma: block.norm1.v_gamma.clone(),
                    norm1_m_beta: block.norm1.m_beta.clone(),
                    norm1_v_beta: block.norm1.v_beta.clone(),
                    norm1_t: block.norm1.t,
                    norm2_gamma: block.norm2.gamma.clone(),
                    norm2_beta: block.norm2.beta.clone(),
                    norm2_m_gamma: block.norm2.m_gamma.clone(),
                    norm2_v_gamma: block.norm2.v_gamma.clone(),
                    norm2_m_beta: block.norm2.m_beta.clone(),
                    norm2_v_beta: block.norm2.v_beta.clone(),
                    norm2_t: block.norm2.t,
                    ff_linear1_weight: block.ff.linear1.weight.clone(),
                    ff_linear1_bias: block.ff.linear1.bias.clone(),
                    ff_linear1_m_weight: block.ff.linear1.m_weight.clone(),
                    ff_linear1_v_weight: block.ff.linear1.v_weight.clone(),
                    ff_linear1_m_bias: block.ff.linear1.m_bias.clone(),
                    ff_linear1_v_bias: block.ff.linear1.v_bias.clone(),
                    ff_linear1_t: block.ff.linear1.t,
                    ff_linear2_weight: block.ff.linear2.weight.clone(),
                    ff_linear2_bias: block.ff.linear2.bias.clone(),
                    ff_linear2_m_weight: block.ff.linear2.m_weight.clone(),
                    ff_linear2_v_weight: block.ff.linear2.v_weight.clone(),
                    ff_linear2_m_bias: block.ff.linear2.m_bias.clone(),
                    ff_linear2_v_bias: block.ff.linear2.v_bias.clone(),
                    ff_linear2_t: block.ff.linear2.t,
                }
            }).collect(),
            output_norm_gamma: cpu.output_norm.gamma.clone(),
            output_norm_beta: cpu.output_norm.beta.clone(),
            output_norm_m_gamma: cpu.output_norm.m_gamma.clone(),
            output_norm_v_gamma: cpu.output_norm.v_gamma.clone(),
            output_norm_m_beta: cpu.output_norm.m_beta.clone(),
            output_norm_v_beta: cpu.output_norm.v_beta.clone(),
            output_norm_t: cpu.output_norm.t,
            output_proj_weight: cpu.output_proj.weight.clone(),
            output_proj_bias: cpu.output_proj.bias.clone(),
            output_proj_m_weight: cpu.output_proj.m_weight.clone(),
            output_proj_v_weight: cpu.output_proj.v_weight.clone(),
            output_proj_m_bias: cpu.output_proj.m_bias.clone(),
            output_proj_v_bias: cpu.output_proj.v_bias.clone(),
            output_proj_t: cpu.output_proj.t,
        }
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
