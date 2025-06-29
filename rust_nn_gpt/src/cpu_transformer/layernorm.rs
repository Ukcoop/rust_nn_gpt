/// Simple LayerNorm: normalizes input to zero mean and unit variance, with learnable scale (gamma) and shift (beta)
pub struct LayerNorm {
    pub embed_dim: usize,
    pub gamma: Vec<f32>, // scale
    pub beta: Vec<f32>,  // shift
}

impl LayerNorm {
    pub fn new(config: &crate::cpu_transformer::config::TransformerConfig) -> Self {
        let embed_dim = config.height;
        Self {
            embed_dim,
            gamma: vec![1.0; embed_dim],
            beta: vec![0.0; embed_dim],
        }
    }
    /// Forward pass: normalize input to zero mean and unit variance, then scale and shift
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let (mean, std) = self.compute_mean_std(input);
        self.normalize_and_scale(input, mean, std)
    }
    /// Backward pass for a single input/target pair (stub for demonstration)
    /// Updates gamma and beta using a simple gradient descent step on the output error.
    pub fn backward(&mut self, _input: &[f32], output_grad: &[f32], lr: f32) {
        let (mean, std) = self.compute_mean_std(output_grad);
        for i in 0..self.embed_dim {
            let normed = Self::normalize_value(output_grad[i], mean, std);
            // dLoss/dGamma = dLoss/dOut * normed
            let grad_gamma = normed;
            // dLoss/dBeta = dLoss/dOut
            let grad_beta = normed;
            Self::update_param(&mut self.gamma[i], grad_gamma, lr);
            Self::update_param(&mut self.beta[i], grad_beta, lr);
        }
    }
    /// Compute mean and std of input
    fn compute_mean_std(&self, input: &[f32]) -> (f32, f32) {
        let n = input.len() as f32;
        if n == 0.0 {
            return (0.0, 1.0);
        }
        let mean = input.iter().sum::<f32>() / n;
        let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = var.sqrt().max(1e-6);
        (mean, std)
    }
    /// Normalize and scale an entire input vector
    fn normalize_and_scale(&self, input: &[f32], mean: f32, std: f32) -> Vec<f32> {
        input
            .iter()
            .enumerate()
            .map(|(i, x)| self.gamma[i] * Self::normalize_value(*x, mean, std) + self.beta[i])
            .collect()
    }
    /// Normalize a single value
    fn normalize_value(x: f32, mean: f32, std: f32) -> f32 {
        (x - mean) / std
    }
    /// Update a single parameter with gradient and learning rate
    fn update_param(param: &mut f32, grad: f32, lr: f32) {
        *param -= lr * grad;
    }
}
