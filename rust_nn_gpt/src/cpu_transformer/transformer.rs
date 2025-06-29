use crate::cpu_transformer::block::TransformerBlock;
use crate::cpu_transformer::config::TransformerConfig;
use crate::cpu_transformer::input_projection::InputProjection;
use crate::cpu_transformer::output_projection::OutputProjection;

/// CPU transformer implementation: a stack of blocks
pub struct CpuTransformer {
    pub input_proj: InputProjection,
    pub output_proj: OutputProjection,
    pub blocks: Vec<TransformerBlock>,
    pub config: TransformerConfig,
}

impl CpuTransformer {
    pub fn new(config: &TransformerConfig) -> Self {
        let input_proj = InputProjection::new(config);
        let output_proj = OutputProjection::new(config);
        let blocks = (0..config.layers)
            .map(|_| TransformerBlock::new(config))
            .collect();
        Self {
            input_proj,
            output_proj,
            blocks,
            config: config.clone(),
        }
    }
    /// Forward pass: 1D input -> 2D -> transformer -> 1D output
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let x2d = self.compute_x2d(input);
        self.output_proj.forward(&x2d)
    }
    /// Backward pass for a single input/target pair (MSE loss)
    pub fn backward(
        &mut self,
        input: &[f32],
        target: &[f32],
        lr: f32,
    ) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        let (intermediates, output) = self.forward_with_intermediates(input);
        let (mse, grad_out) = self.compute_loss_and_grad(&output, target);
        let last_intermediate = intermediates.last().ok_or_else(|| {
            Box::<dyn std::error::Error + Send + Sync>::from(
                "No intermediate found in backward pass",
            )
        })?;
        self.output_proj.backward(last_intermediate, &grad_out, lr);
        // TODO: Backprop through transformer blocks and input_proj for full training
        Ok(mse)
    }
    /// Train for a single input/target pair (MSE loss)
    /// TODO: Backprop through transformer blocks and input_proj for full training
    pub fn train(&mut self, input: &[f32], target: &[f32], lr: f32) -> f32 {
        let x2d = self.compute_x2d(input);
        let output = self.output_proj.forward(&x2d);
        let mse = self.compute_loss(&output, target);
        let grad_out = self.compute_grad(&output, target);
        self.output_proj.backward(&x2d, &grad_out, lr);
        // TODO: Backprop through transformer blocks and input_proj for full training
        mse
    }
    // --- Private helper functions ---
    fn compute_x2d(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let mut x2d = self.input_proj.forward(input);
        for block in &self.blocks {
            x2d = x2d.into_iter().map(|token| block.forward(&token)).collect();
        }
        x2d
    }
    fn forward_with_intermediates(&self, input: &[f32]) -> (Vec<Vec<Vec<f32>>>, Vec<f32>) {
        let mut x2d = self.input_proj.forward(input);
        let mut intermediates = vec![x2d.clone()];
        for block in &self.blocks {
            x2d = x2d.iter().map(|token| block.forward(token)).collect();
            intermediates.push(x2d.clone());
        }
        let output = self.output_proj.forward(&x2d);
        (intermediates, output)
    }
    fn compute_loss_and_grad(&self, output: &[f32], target: &[f32]) -> (f32, Vec<f32>) {
        let mse = self.compute_loss(output, target);
        let grad_out = self.compute_grad(output, target);
        (mse, grad_out)
    }
    fn compute_loss(&self, output: &[f32], target: &[f32]) -> f32 {
        output
            .iter()
            .zip(target)
            .map(|(y, t)| (y - t).powi(2))
            .sum::<f32>()
            / output.len() as f32
    }
    fn compute_grad(&self, output: &[f32], target: &[f32]) -> Vec<f32> {
        output
            .iter()
            .zip(target)
            .map(|(y, t)| 2.0 * (y - t) / output.len() as f32)
            .collect()
    }
}
