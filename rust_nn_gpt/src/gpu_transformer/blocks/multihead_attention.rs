use crate::gpu_transformer::blocks::linear::{BatchContext, OptimizerParams, VulkanLinearLayer};
use std::error::Error;

pub struct VulkanMultiheadAttention {
    q_proj: VulkanLinearLayer,
    k_proj: VulkanLinearLayer,
    v_proj: VulkanLinearLayer,
    out_proj: VulkanLinearLayer,
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
}

impl VulkanMultiheadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Result<Self, Box<dyn Error>> {
        let head_dim = embed_dim / num_heads;
        let q_proj = VulkanLinearLayer::new(embed_dim, embed_dim)?;
        let k_proj = VulkanLinearLayer::new(embed_dim, embed_dim)?;
        let v_proj = VulkanLinearLayer::new(embed_dim, embed_dim)?;
        let out_proj = VulkanLinearLayer::new(embed_dim, embed_dim)?;
        Ok(VulkanMultiheadAttention {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            embed_dim,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(
        &self,
        input: &[f32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        // Project to Q, K, V
        let weights_q = self.q_proj.get_weights()?;
        let bias_q = self.q_proj.get_bias()?;
        let q = self
            .q_proj
            .forward(input, &weights_q, &bias_q, self.embed_dim, self.embed_dim)?;
        let weights_k = self.k_proj.get_weights()?;
        let bias_k = self.k_proj.get_bias()?;
        let k = self
            .k_proj
            .forward(input, &weights_k, &bias_k, self.embed_dim, self.embed_dim)?;
        let weights_v = self.v_proj.get_weights()?;
        let bias_v = self.v_proj.get_bias()?;
        let v = self
            .v_proj
            .forward(input, &weights_v, &bias_v, self.embed_dim, self.embed_dim)?;
        // Reshape and compute attention (CPU for now)
        let mut output = vec![0.0f32; batch_size * seq_len * self.embed_dim];
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    // Q for this token
                    let q_offset =
                        b * seq_len * self.embed_dim + i * self.embed_dim + h * self.head_dim;
                    let q_vec = &q[q_offset..q_offset + self.head_dim];
                    // Compute attention scores for all tokens
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let k_offset =
                            b * seq_len * self.embed_dim + j * self.embed_dim + h * self.head_dim;
                        let k_vec = &k[k_offset..k_offset + self.head_dim];
                        let mut dot = 0.0;
                        for d in 0..self.head_dim {
                            dot += q_vec[d] * k_vec[d];
                        }
                        scores[j] = dot / (self.head_dim as f32).sqrt();
                    }
                    // Softmax
                    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exp_scores: Vec<f32> =
                        scores.iter().map(|&s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();
                    let attn: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();
                    // Weighted sum of V
                    let mut attn_out = vec![0.0f32; self.head_dim];
                    for j in 0..seq_len {
                        let v_offset =
                            b * seq_len * self.embed_dim + j * self.embed_dim + h * self.head_dim;
                        let v_vec = &v[v_offset..v_offset + self.head_dim];
                        for d in 0..self.head_dim {
                            attn_out[d] += attn[j] * v_vec[d];
                        }
                    }
                    // Write to output
                    let out_offset =
                        b * seq_len * self.embed_dim + i * self.embed_dim + h * self.head_dim;
                    output[out_offset..(self.head_dim + out_offset)]
                        .copy_from_slice(&attn_out[..self.head_dim]);
                }
            }
        }
        // Final output projection
        let weights_o = self.out_proj.get_weights()?;
        let bias_o = self.out_proj.get_bias()?;
        let final_out =
            self.out_proj
                .forward(&output, &weights_o, &bias_o, self.embed_dim, self.embed_dim)?;
        Ok(final_out)
    }

    pub fn train_batch_adam(
        &mut self,
        input: &[f32],
        target: &[f32],
        opt: &OptimizerParams,
        batch: &BatchContext,
    ) -> Result<f32, Box<dyn Error>> {
        // Forward pass
        let output = self.forward(input, batch.batch_size, batch.seq_len)?;
        // Compute loss and gradients (MSE)
        let mut loss = 0.0;
        for i in 0..output.len() {
            let diff = output[i] - target[i];
            loss += diff * diff;
        }
        loss /= output.len() as f32;

        // For now, update all linear layers with Adam using dummy targets
        self.out_proj.train_batch_adam(input, target, opt, batch)?;
        self.q_proj.train_batch_adam(input, input, opt, batch)?;
        self.k_proj.train_batch_adam(input, input, opt, batch)?;
        self.v_proj.train_batch_adam(input, input, opt, batch)?;
        Ok(loss)
    }

    pub fn backward(
        &mut self,
        input: &[f32],
        grad_output: &[f32],
        opt: &OptimizerParams,
        batch: &BatchContext,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        // Forward pass to get intermediate activations
        let weights_q = self.q_proj.get_weights()?;
        let bias_q = self.q_proj.get_bias()?;
        let q = self
            .q_proj
            .forward(input, &weights_q, &bias_q, self.embed_dim, self.embed_dim)?;
        let weights_k = self.k_proj.get_weights()?;
        let bias_k = self.k_proj.get_bias()?;
        let k = self
            .k_proj
            .forward(input, &weights_k, &bias_k, self.embed_dim, self.embed_dim)?;
        let weights_v = self.v_proj.get_weights()?;
        let bias_v = self.v_proj.get_bias()?;
        let v = self
            .v_proj
            .forward(input, &weights_v, &bias_v, self.embed_dim, self.embed_dim)?;

        // Compute attention output
        let mut attn_output = vec![0.0f32; batch.batch_size * batch.seq_len * self.embed_dim];
        for b in 0..batch.batch_size {
            for h in 0..self.num_heads {
                for i in 0..batch.seq_len {
                    let q_offset =
                        b * batch.seq_len * self.embed_dim + i * self.embed_dim + h * self.head_dim;
                    let q_vec = &q[q_offset..q_offset + self.head_dim];
                    let mut scores = vec![0.0f32; batch.seq_len];
                    for j in 0..batch.seq_len {
                        let k_offset = b * batch.seq_len * self.embed_dim
                            + j * self.embed_dim
                            + h * self.head_dim;
                        let k_vec = &k[k_offset..k_offset + self.head_dim];
                        let mut dot = 0.0;
                        for d in 0..self.head_dim {
                            dot += q_vec[d] * k_vec[d];
                        }
                        scores[j] = dot / (self.head_dim as f32).sqrt();
                    }
                    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exp_scores: Vec<f32> =
                        scores.iter().map(|&s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();
                    let attn: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();
                    let mut attn_out = vec![0.0f32; self.head_dim];
                    for j in 0..batch.seq_len {
                        let v_offset = b * batch.seq_len * self.embed_dim
                            + j * self.embed_dim
                            + h * self.head_dim;
                        let v_vec = &v[v_offset..v_offset + self.head_dim];
                        for d in 0..self.head_dim {
                            attn_out[d] += attn[j] * v_vec[d];
                        }
                    }
                    let out_offset =
                        b * batch.seq_len * self.embed_dim + i * self.embed_dim + h * self.head_dim;
                    attn_output[out_offset..(self.head_dim + out_offset)]
                        .copy_from_slice(&attn_out[..self.head_dim]);
                }
            }
        }

        let weights_o = self.out_proj.get_weights()?;
        let bias_o = self.out_proj.get_bias()?;
        let _final_out = self.out_proj.forward(
            &attn_output,
            &weights_o,
            &bias_o,
            self.embed_dim,
            self.embed_dim,
        )?;

        // Backward pass through output projection
        let _attn_grad = self
            .out_proj
            .backward(&attn_output, grad_output, opt, batch)?;

        // For now, we'll use a simplified backward pass for the attention mechanism
        // In a full implementation, we'd need to compute gradients through the attention scores
        // and backpropagate through Q, K, V projections

        // Update the projection layers with simplified gradients
        // This is a simplified approach - in practice, we'd need proper attention gradients
        self.q_proj.train_batch_adam(input, &q, opt, batch)?;
        self.k_proj.train_batch_adam(input, &k, opt, batch)?;
        self.v_proj.train_batch_adam(input, &v, opt, batch)?;

        // For now, return a simplified input gradient
        // In a proper implementation, this would be computed from the attention gradients
        let mut grad_input = vec![0.0f32; input.len()];

        // Simplified: distribute gradients equally across input dimensions
        for i in 0..input.len() {
            grad_input[i] = grad_output[i] / self.num_heads as f32;
        }

        Ok(grad_input)
    }
}
