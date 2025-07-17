use std::error::Error;

use crate::transformer::TransformerConfig;

pub mod blocks;
pub mod compute_pipeline;
pub mod shader_loader;
pub mod vulkan_context;

pub mod embedded_shaders {
    include!(concat!(env!("OUT_DIR"), "/embedded_shaders.rs"));
}

use blocks::layernorm::VulkanLayerNorm;
use blocks::linear::{BatchContext, OptimizerParams, VulkanLinearLayer};
use blocks::transformer_block::VulkanTransformerBlock;

pub struct VulkanTransformer {
    // Input processing
    input_proj: VulkanLinearLayer,
    input_norm: VulkanLayerNorm,

    // Transformer blocks
    blocks: Vec<VulkanTransformerBlock>,

    // Output processing
    output_norm: VulkanLayerNorm,
    output_proj: VulkanLinearLayer,

    // Configuration
    input_dim: usize,
    model_dim: usize,
    output_dim: usize,
    seq_len: usize,
}

impl VulkanTransformer {
    pub fn new(config: &TransformerConfig) -> Result<Self, Box<dyn Error>> {
        let input_dim = config.input_dim;
        let model_dim = config.model_dim;
        let output_dim = config.output_dim;
        let num_layers = config.num_layers;
        let num_heads = config.num_heads;
        let seq_len = 1; // For now, use sequence length of 1 since we're not doing sequence processing

        // Create input processing layers
        let input_proj = VulkanLinearLayer::new(input_dim, model_dim)?;
        let input_norm = VulkanLayerNorm::new(model_dim)?;

        // Create transformer blocks
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(VulkanTransformerBlock::new(
                model_dim,
                model_dim * 4,
                num_heads,
            )?);
        }

        // Create output processing layers
        let output_norm = VulkanLayerNorm::new(model_dim)?;
        let output_proj = VulkanLinearLayer::new(model_dim, output_dim)?;

        Ok(VulkanTransformer {
            input_proj,
            input_norm,
            blocks,
            output_norm,
            output_proj,
            input_dim,
            model_dim,
            output_dim,
            seq_len,
        })
    }

    pub fn forward(&self, input: &[f32], batch_size: usize) -> Result<Vec<f32>, Box<dyn Error>> {
        // Input projection
        let weights_in = self.input_proj.get_weights()?;
        let bias_in = self.input_proj.get_bias()?;
        let mut x = self.input_proj.forward(
            input,
            &weights_in,
            &bias_in,
            self.input_dim,
            self.model_dim,
        )?;

        // Input normalization
        x = self.input_norm.forward(&x, batch_size)?;

        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(&x, batch_size, self.seq_len)?;
        }

        // Output normalization
        x = self.output_norm.forward(&x, batch_size)?;

        // Output projection
        let weights_out = self.output_proj.get_weights()?;
        let bias_out = self.output_proj.get_bias()?;
        let output = self.output_proj.forward(
            &x,
            &weights_out,
            &bias_out,
            self.model_dim,
            self.output_dim,
        )?;

        Ok(output)
    }

    pub fn train_batch_adam(
        &mut self,
        input: &[f32],
        target: &[f32],
        opt: &OptimizerParams,
        batch: &BatchContext,
    ) -> Result<f32, Box<dyn Error>> {
        // Forward pass through all layers
        let weights_in = self.input_proj.get_weights()?;
        let bias_in = self.input_proj.get_bias()?;
        let mut x = self.input_proj.forward(
            input,
            &weights_in,
            &bias_in,
            self.input_dim,
            self.model_dim,
        )?;

        let mut activations = Vec::with_capacity(self.blocks.len() + 3); // +3 for input_proj, input_norm, and output_norm
        activations.push(x.clone()); // Store input projection output

        x = self.input_norm.forward(&x, batch.batch_size)?;
        activations.push(x.clone()); // Store input norm output

        for block in &self.blocks {
            x = block.forward(&x, batch.batch_size, self.seq_len)?;
            activations.push(x.clone()); // Store each block's output
        }

        x = self.output_norm.forward(&x, batch.batch_size)?;
        activations.push(x.clone()); // Store output norm output

        let weights_out = self.output_proj.get_weights()?;
        let bias_out = self.output_proj.get_bias()?;
        let output = self.output_proj.forward(
            &x,
            &weights_out,
            &bias_out,
            self.model_dim,
            self.output_dim,
        )?;

        // Compute loss (MSE) at the final output
        let mut loss = 0.0;
        for i in 0..output.len() {
            let diff = output[i] - target[i];
            loss += diff * diff;
        }
        loss /= output.len() as f32;

        // Compute output gradients (derivative of MSE loss)
        let mut grad_out = vec![0.0f32; output.len()];
        for i in 0..output.len() {
            grad_out[i] = 2.0 * (output[i] - target[i]) / output.len() as f32;
        }

        // Backward pass through output projection
        let batch = BatchContext {
            batch_size: batch.batch_size,
            seq_len: 1,
        };
        let opt = OptimizerParams {
            learning_rate: opt.learning_rate,
            beta1: opt.beta1,
            beta2: opt.beta2,
            epsilon: opt.epsilon,
        };

        let grad_x = self.output_proj.backward(&x, &grad_out, &opt, &batch)?;

        // Backward pass through output normalization
        let batch_ctx = BatchContext {
            batch_size: batch.batch_size,
            seq_len: 1,
        };
        let opt = OptimizerParams {
            learning_rate: opt.learning_rate,
            beta1: opt.beta1,
            beta2: opt.beta2,
            epsilon: opt.epsilon,
        };
        let grad_x = self.output_norm.backward(
            &activations[activations.len() - 2],
            &grad_x,
            &opt,
            &batch_ctx,
        )?;

        // Backward pass through transformer blocks in reverse order
        let batch_ctx = BatchContext {
            batch_size: batch.batch_size,
            seq_len: self.seq_len,
        };
        let opt = OptimizerParams {
            learning_rate: opt.learning_rate,
            beta1: opt.beta1,
            beta2: opt.beta2,
            epsilon: opt.epsilon,
        };
        let mut current_grad = grad_x;
        for (block_idx, block) in self.blocks.iter_mut().enumerate().rev() {
            let input_activation = &activations[block_idx + 2]; // +2 for input_proj and input_norm
            current_grad = block.backward(input_activation, &current_grad, &batch_ctx, &opt)?;
        }

        // Backward pass through input normalization
        let batch_ctx = BatchContext {
            batch_size: batch.batch_size,
            seq_len: 1,
        };
        let opt = OptimizerParams {
            learning_rate: opt.learning_rate,
            beta1: opt.beta1,
            beta2: opt.beta2,
            epsilon: opt.epsilon,
        };
        let grad_x = self
            .input_norm
            .backward(&activations[0], &current_grad, &opt, &batch_ctx)?;

        // Backward pass through input projection
        let batch = BatchContext {
            batch_size: batch.batch_size,
            seq_len: 1,
        };
        let opt = OptimizerParams {
            learning_rate: opt.learning_rate,
            beta1: opt.beta1,
            beta2: opt.beta2,
            epsilon: opt.epsilon,
        };
        let _input_grad = self.input_proj.backward(input, &grad_x, &opt, &batch)?;

        Ok(loss)
    }
}
