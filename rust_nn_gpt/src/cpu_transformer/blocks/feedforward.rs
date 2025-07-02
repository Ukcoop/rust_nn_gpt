use super::linear::Linear;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct FeedForward {
    pub linear1: Linear,
    pub linear2: Linear,
    #[serde(skip)]
    pub last_input: Option<Vec<f32>>,
    #[serde(skip)]
    pub last_relu: Option<Vec<f32>>,
}

impl FeedForward {
    pub fn new(model_dim: usize, hidden_dim: usize) -> Self {
        FeedForward {
            linear1: Linear::new(model_dim, hidden_dim),
            linear2: Linear::new(hidden_dim, model_dim),
            last_input: None,
            last_relu: None,
        }
    }

    pub fn forward(&mut self, x: &[f32]) -> Vec<f32> {
        self.last_input = Some(x.to_vec());
        let h = self.linear1.forward(x);
        let h_relu = h.par_iter().map(|v| v.max(0.0)).collect::<Vec<_>>(); // ReLU
        self.last_relu = Some(h_relu.clone());
        self.linear2.forward(&h_relu)
    }

    pub fn zero_grad(&mut self) {
        self.linear1.zero_grad();
        self.linear2.zero_grad();
    }

    pub fn apply_gradients(&mut self, lr: f32, scale: f32) {
        self.linear1.apply_gradients(lr, scale);
        self.linear2.apply_gradients(lr, scale);
    }

    pub fn apply_gradients_adam(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, scale: f32) {
        self.linear1
            .apply_gradients_adam(lr, beta1, beta2, eps, scale);
        self.linear2
            .apply_gradients_adam(lr, beta1, beta2, eps, scale);
    }

    /// Backward for a single sample: grad_output is [model_dim], returns grad_input [model_dim]
    pub fn backward(
        &mut self,
        grad_output: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let relu = self
            .last_relu
            .as_ref()
            .ok_or_else(|| Box::<dyn std::error::Error>::from("Call forward before backward"))?;
        let input = self
            .last_input
            .as_ref()
            .ok_or_else(|| Box::<dyn std::error::Error>::from("Call forward before backward"))?;
        let grad_h_relu = self.linear2.backward(relu, grad_output);
        let grad_h: Vec<f32> = relu
            .iter()
            .zip(grad_h_relu.iter())
            .map(|(&r, &g)| if r > 0.0 { g } else { 0.0 })
            .collect();
        Ok(self.linear1.backward(input, &grad_h))
    }

    pub fn clip_gradients(&mut self, max_norm: f32) {
        self.linear1.clip_gradients(max_norm);
        self.linear2.clip_gradients(max_norm);
    }
}
