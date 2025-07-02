use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct LayerNorm {
    pub gamma: Vec<f32>, // scale
    pub beta: Vec<f32>,  // shift
    pub eps: f32,
    pub grad_gamma: Vec<f32>,
    pub grad_beta: Vec<f32>,
    // Adam state
    pub m_gamma: Vec<f32>,
    pub v_gamma: Vec<f32>,
    pub m_beta: Vec<f32>,
    pub v_beta: Vec<f32>,
    pub t: usize,
    #[serde(skip)]
    pub last_input: Option<Vec<f32>>,
    #[serde(skip)]
    pub last_norm: Option<Vec<f32>>,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        LayerNorm {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            eps: 1e-5,
            grad_gamma: vec![0.0; dim],
            grad_beta: vec![0.0; dim],
            m_gamma: vec![0.0; dim],
            v_gamma: vec![0.0; dim],
            m_beta: vec![0.0; dim],
            v_beta: vec![0.0; dim],
            t: 0,
            last_input: None,
            last_norm: None,
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        self.last_input = Some(input.to_vec());
        let mean = input.iter().sum::<f32>() / input.len() as f32;
        let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;
        let norm: Vec<f32> = input
            .par_iter()
            .map(|x| (*x - mean) / (var + self.eps).sqrt())
            .collect();
        self.last_norm = Some(norm.clone());
        norm.par_iter()
            .enumerate()
            .map(|(i, n)| self.gamma[i] * n + self.beta[i])
            .collect()
    }

    pub fn apply_gradients(&mut self, lr: f32, scale: f32) {
        for (g, gg) in self.gamma.iter_mut().zip(self.grad_gamma.iter()) {
            *g -= lr * gg * scale;
        }
        for (b, gb) in self.beta.iter_mut().zip(self.grad_beta.iter()) {
            *b -= lr * gb * scale;
        }
        // Zero gradients after applying
        for g in &mut self.grad_gamma {
            *g = 0.0;
        }
        for g in &mut self.grad_beta {
            *g = 0.0;
        }
    }

    pub fn apply_gradients_adam(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, scale: f32) {
        self.t += 1;
        let t = self.t as f32;
        for i in 0..self.gamma.len() {
            let g = self.grad_gamma[i] * scale;
            self.m_gamma[i] = beta1 * self.m_gamma[i] + (1.0 - beta1) * g;
            self.v_gamma[i] = beta2 * self.v_gamma[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m_gamma[i] / (1.0 - beta1.powf(t));
            let v_hat = self.v_gamma[i] / (1.0 - beta2.powf(t));
            self.gamma[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        for i in 0..self.beta.len() {
            let g = self.grad_beta[i] * scale;
            self.m_beta[i] = beta1 * self.m_beta[i] + (1.0 - beta1) * g;
            self.v_beta[i] = beta2 * self.v_beta[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m_beta[i] / (1.0 - beta1.powf(t));
            let v_hat = self.v_beta[i] / (1.0 - beta2.powf(t));
            self.beta[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        // Zero gradients after applying
        for g in &mut self.grad_gamma {
            *g = 0.0;
        }
        for g in &mut self.grad_beta {
            *g = 0.0;
        }
    }

    /// Backward for a single sample: grad_output is [dim], returns grad_input [dim]
    pub fn backward(
        &mut self,
        grad_output: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let input = self
            .last_input
            .as_ref()
            .ok_or_else(|| Box::<dyn std::error::Error>::from("Call forward before backward"))?;
        let norm = self
            .last_norm
            .as_ref()
            .ok_or_else(|| Box::<dyn std::error::Error>::from("Call forward before backward"))?;
        let n = input.len() as f32;
        let mean = input.iter().sum::<f32>() / n;
        let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = (var + self.eps).sqrt();
        for i in 0..n as usize {
            self.grad_gamma[i] += grad_output[i] * norm[i];
            self.grad_beta[i] += grad_output[i];
        }
        let mut grad_input = vec![0.0; n as usize];
        let gamma = &self.gamma;
        let sum1: f32 = (0..n as usize).map(|i| grad_output[i] * gamma[i]).sum();
        let sum2: f32 = (0..n as usize)
            .map(|i| grad_output[i] * gamma[i] * (input[i] - mean))
            .sum();
        for i in 0..n as usize {
            grad_input[i] = (1.0 / std) * gamma[i] * grad_output[i]
                - (1.0 / (n * std)) * gamma[i] * sum1
                - (input[i] - mean) * (1.0 / (n * std.powi(3))) * gamma[i] * sum2;
        }
        Ok(grad_input)
    }

    pub fn clip_gradients(&mut self, max_norm: f32) {
        let mut norm_sq = 0.0;
        for g in &self.grad_gamma {
            norm_sq += g * g;
        }
        for g in &self.grad_beta {
            norm_sq += g * g;
        }
        let norm = norm_sq.sqrt();
        if norm > max_norm {
            let scale = max_norm / norm;
            for g in &mut self.grad_gamma {
                *g *= scale;
            }
            for g in &mut self.grad_beta {
                *g *= scale;
            }
        }
    }

    pub fn zero_grad(&mut self) {
        for g in &mut self.grad_gamma {
            *g = 0.0;
        }
        for g in &mut self.grad_beta {
            *g = 0.0;
        }
    }
}
