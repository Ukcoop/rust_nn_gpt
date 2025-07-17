use crate::gpu_transformer::blocks::linear::{BatchContext, OptimizerParams, VulkanLinearLayer};
use std::error::Error;

pub struct VulkanFeedforward {
    linear1: VulkanLinearLayer,
    linear2: VulkanLinearLayer,
    hidden_dim: usize,
    input_dim: usize,
    output_dim: usize,
}

impl VulkanFeedforward {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let linear1 = VulkanLinearLayer::new(input_dim, hidden_dim)?;
        let linear2 = VulkanLinearLayer::new(hidden_dim, output_dim)?;
        Ok(VulkanFeedforward {
            linear1,
            linear2,
            hidden_dim,
            input_dim,
            output_dim,
        })
    }

    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
        // First linear
        let weights1 = self.linear1.get_weights()?;
        let bias1 = self.linear1.get_bias()?;
        let hidden =
            self.linear1
                .forward(input, &weights1, &bias1, self.input_dim, self.hidden_dim)?;
        // GELU activation (CPU for now)
        let gelu: Vec<f32> = hidden
            .iter()
            .map(|&x| {
                0.5 * x
                    * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x * x * x)).tanh())
            })
            .collect();
        // Second linear
        let weights2 = self.linear2.get_weights()?;
        let bias2 = self.linear2.get_bias()?;
        let output =
            self.linear2
                .forward(&gelu, &weights2, &bias2, self.hidden_dim, self.output_dim)?;
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
        let weights1 = self.linear1.get_weights()?;
        let bias1 = self.linear1.get_bias()?;
        let hidden =
            self.linear1
                .forward(input, &weights1, &bias1, self.input_dim, self.hidden_dim)?;
        let gelu: Vec<f32> = hidden
            .iter()
            .map(|&x| {
                0.5 * x
                    * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x * x * x)).tanh())
            })
            .collect();
        let weights2 = self.linear2.get_weights()?;
        let bias2 = self.linear2.get_bias()?;
        let output =
            self.linear2
                .forward(&gelu, &weights2, &bias2, self.hidden_dim, self.output_dim)?;
        // Compute loss and gradients (MSE)
        let mut loss = 0.0;
        let mut grad_output = vec![0.0f32; output.len()];
        for i in 0..output.len() {
            let diff = output[i] - target[i];
            loss += diff * diff;
            grad_output[i] = 2.0 * diff;
        }
        loss /= output.len() as f32;

        // Backward pass (not implemented on GPU yet)
        // For now, just update both linear layers with Adam using the same gradients
        self.linear2.train_batch_adam(&gelu, target, opt, batch)?;
        self.linear1.train_batch_adam(input, &hidden, opt, batch)?;
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
        let weights1 = self.linear1.get_weights()?;
        let bias1 = self.linear1.get_bias()?;
        let hidden =
            self.linear1
                .forward(input, &weights1, &bias1, self.input_dim, self.hidden_dim)?;

        // GELU activation and its derivative
        let mut gelu = Vec::with_capacity(hidden.len());
        let mut gelu_derivative = Vec::with_capacity(hidden.len());
        for &x in &hidden {
            let x_cubed = x * x * x;
            let inner = std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x_cubed);
            let tanh_inner = inner.tanh();
            gelu.push(0.5 * x * (1.0 + tanh_inner));

            // GELU derivative
            let sech_squared = 1.0 - tanh_inner * tanh_inner;
            let derivative = 0.5 * (1.0 + tanh_inner)
                + 0.5
                    * x
                    * sech_squared
                    * std::f32::consts::FRAC_2_SQRT_PI
                    * (1.0 + 3.0 * 0.044715 * x * x);
            gelu_derivative.push(derivative);
        }

        let weights2 = self.linear2.get_weights()?;
        let bias2 = self.linear2.get_bias()?;
        let _output =
            self.linear2
                .forward(&gelu, &weights2, &bias2, self.hidden_dim, self.output_dim)?;

        // Backward pass through linear2
        let linear2_grad = self.linear2.backward(&gelu, grad_output, opt, batch)?;

        // Backward through GELU
        let mut gelu_grad = vec![0.0f32; linear2_grad.len()];
        for i in 0..linear2_grad.len() {
            gelu_grad[i] = linear2_grad[i] * gelu_derivative[i];
        }

        // Backward through linear1
        let input_grad = self.linear1.backward(input, &gelu_grad, opt, batch)?;

        Ok(input_grad)
    }
}
