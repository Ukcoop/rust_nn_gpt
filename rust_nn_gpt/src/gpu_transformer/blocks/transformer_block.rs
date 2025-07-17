use crate::gpu_transformer::blocks::feedforward::VulkanFeedforward;
use crate::gpu_transformer::blocks::layernorm::VulkanLayerNorm;
use crate::gpu_transformer::blocks::linear::{BatchContext, OptimizerParams};
use crate::gpu_transformer::blocks::multihead_attention::VulkanMultiheadAttention;
use std::error::Error;

pub struct VulkanTransformerBlock {
    norm1: VulkanLayerNorm,
    attn: VulkanMultiheadAttention,
    norm2: VulkanLayerNorm,
    ff: VulkanFeedforward,
}

impl VulkanTransformerBlock {
    pub fn new(
        embed_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let norm1 = VulkanLayerNorm::new(embed_dim)?;
        let attn = VulkanMultiheadAttention::new(embed_dim, num_heads)?;
        let norm2 = VulkanLayerNorm::new(embed_dim)?;
        let ff = VulkanFeedforward::new(embed_dim, hidden_dim, embed_dim)?;
        Ok(VulkanTransformerBlock {
            norm1,
            attn,
            norm2,
            ff,
        })
    }

    pub fn forward(
        &self,
        input: &[f32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        // LayerNorm 1
        let normed1 = self.norm1.forward(input, batch_size * seq_len)?;
        // Multihead Attention + residual
        let attn_out = self.attn.forward(&normed1, batch_size, seq_len)?;
        let mut attn_res = vec![0.0f32; attn_out.len()];
        for i in 0..attn_out.len() {
            attn_res[i] = attn_out[i] + input[i];
        }
        // LayerNorm 2
        let normed2 = self.norm2.forward(&attn_res, batch_size * seq_len)?;
        let _ff_out = self.ff.forward(&normed2)?;
        let mut output = vec![0.0f32; _ff_out.len()];
        for i in 0.._ff_out.len() {
            output[i] = _ff_out[i] + attn_res[i];
        }
        Ok(output)
    }

    pub fn train_batch_adam(
        &mut self,
        input: &[f32],
        target: &[f32],
        batch: &BatchContext,
        opt: &OptimizerParams,
    ) -> Result<f32, Box<dyn Error>> {
        // Forward pass
        let output = self.forward(input, batch.batch_size, batch.seq_len)?;
        // Compute loss (MSE)
        let mut loss = 0.0;
        for i in 0..output.len() {
            let diff = output[i] - target[i];
            loss += diff * diff;
        }
        loss /= output.len() as f32;
        // For now, update all sub-layers with Adam using dummy targets
        self.ff.train_batch_adam(input, target, batch, opt)?;
        self.norm2.train_batch_adam(input, target, opt, batch)?;
        self.attn.train_batch_adam(input, target, opt, batch)?;
        self.norm1.train_batch_adam(input, target, opt, batch)?;
        Ok(loss)
    }

    pub fn backward(
        &mut self,
        input: &[f32],
        grad_output: &[f32],
        batch: &BatchContext,
        opt: &OptimizerParams,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        // Forward pass to get intermediate activations
        let normed1 = self
            .norm1
            .forward(input, batch.batch_size * batch.seq_len)?;
        let attn_out = self
            .attn
            .forward(&normed1, batch.batch_size, batch.seq_len)?;
        let mut attn_res = vec![0.0f32; attn_out.len()];
        for i in 0..attn_out.len() {
            attn_res[i] = attn_out[i] + input[i];
        }
        let normed2 = self
            .norm2
            .forward(&attn_res, batch.batch_size * batch.seq_len)?;
        let _ff_out = self.ff.forward(&normed2)?;

        // Backward pass through feedforward + residual
        let ff_grad = self.ff.backward(&normed2, grad_output, batch, opt)?;

        // Add residual gradient
        let mut ff_res_grad = vec![0.0f32; ff_grad.len()];
        for i in 0..ff_grad.len() {
            ff_res_grad[i] = ff_grad[i] + grad_output[i];
        }

        // Backward through layer norm 2
        let norm2_grad = self.norm2.backward(&attn_res, &ff_grad, opt, batch)?;

        // Add residual gradient for attention
        let mut attn_res_grad = vec![0.0f32; norm2_grad.len()];
        for i in 0..norm2_grad.len() {
            attn_res_grad[i] = norm2_grad[i] + ff_res_grad[i];
        }

        // Backward through attention + residual
        let attn_grad = self.attn.backward(&normed1, &attn_res_grad, opt, batch)?;

        // Add residual gradient
        let mut attn_input_grad = vec![0.0f32; attn_grad.len()];
        for i in 0..attn_grad.len() {
            attn_input_grad[i] = attn_grad[i] + attn_res_grad[i];
        }

        // Backward through layer norm 1
        let input_grad = self.norm1.backward(input, &attn_input_grad, opt, batch)?;

        Ok(input_grad)
    }
}
