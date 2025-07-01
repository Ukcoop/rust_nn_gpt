use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use rayon::prelude::*;

pub struct Linear {
    pub weight: Vec<Vec<f32>>, // shape: [out_dim][in_dim]
    pub bias: Vec<f32>,        // shape: [out_dim]
    pub grad_weight: Vec<Vec<f32>>,
    pub grad_bias: Vec<f32>,
    // Adam state
    pub m_weight: Vec<Vec<f32>>,
    pub v_weight: Vec<Vec<f32>>,
    pub m_bias: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub t: usize,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        // Xavier/Glorot uniform initialization
        let mut rng = rand::thread_rng();
        let limit = (6.0f32 / (in_dim as f32 + out_dim as f32)).sqrt();
        let dist = Uniform::new(-limit, limit);
        let weight: Vec<Vec<f32>> = (0..out_dim)
            .map(|_| (0..in_dim).map(|_| dist.sample(&mut rng)).collect())
            .collect();
        let bias: Vec<f32> = (0..out_dim).map(|_| rng.gen_range(-1e-2..1e-2)).collect();
        let grad_weight = vec![vec![0.0; in_dim]; out_dim];
        let grad_bias = vec![0.0; out_dim];
        let m_weight = vec![vec![0.0; in_dim]; out_dim];
        let v_weight = vec![vec![0.0; in_dim]; out_dim];
        let m_bias = vec![0.0; out_dim];
        let v_bias = vec![0.0; out_dim];
        let t = 0;
        Linear {
            weight,
            bias,
            grad_weight,
            grad_bias,
            m_weight,
            v_weight,
            m_bias,
            v_bias,
            t,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        // input: [in_dim]
        // output: [out_dim]
        self.weight
            .par_iter()
            .zip(self.bias.par_iter())
            .map(|(w_row, b)| {
                w_row
                    .iter()
                    .zip(input.iter())
                    .map(|(w, i)| w * i)
                    .sum::<f32>()
                    + b
            })
            .collect()
    }

    pub fn zero_grad(&mut self) {
        for row in &mut self.grad_weight {
            for g in row.iter_mut() {
                *g = 0.0;
            }
        }
        for g in &mut self.grad_bias {
            *g = 0.0;
        }
    }

    pub fn accumulate_grad(&mut self, grad_w: &[Vec<f32>], grad_b: &[f32]) {
        for (gw_row, grad_row) in self.grad_weight.iter_mut().zip(grad_w.iter()) {
            for (g, d) in gw_row.iter_mut().zip(grad_row.iter()) {
                *g += d;
            }
        }
        for (g, d) in self.grad_bias.iter_mut().zip(grad_b.iter()) {
            *g += d;
        }
    }

    pub fn apply_gradients(&mut self, lr: f32, scale: f32) {
        for (w_row, gw_row) in self.weight.iter_mut().zip(self.grad_weight.iter()) {
            for (w, g) in w_row.iter_mut().zip(gw_row.iter()) {
                *w -= lr * g * scale;
            }
        }
        for (b, g) in self.bias.iter_mut().zip(self.grad_bias.iter()) {
            *b -= lr * g * scale;
        }
        // Zero gradients after applying
        for row in &mut self.grad_weight {
            for g in row.iter_mut() {
                *g = 0.0;
            }
        }
        for g in &mut self.grad_bias {
            *g = 0.0;
        }
    }

    pub fn apply_gradients_adam(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, scale: f32) {
        self.t += 1;
        let t = self.t as f32;
        // Update weights
        for i in 0..self.weight.len() {
            for j in 0..self.weight[0].len() {
                let g = self.grad_weight[i][j] * scale;
                // Adam moments
                self.m_weight[i][j] = beta1 * self.m_weight[i][j] + (1.0 - beta1) * g;
                self.v_weight[i][j] = beta2 * self.v_weight[i][j] + (1.0 - beta2) * g * g;
                // Bias correction
                let m_hat = self.m_weight[i][j] / (1.0 - beta1.powf(t));
                let v_hat = self.v_weight[i][j] / (1.0 - beta2.powf(t));
                // Update
                self.weight[i][j] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
        // Update biases
        for i in 0..self.bias.len() {
            let g = self.grad_bias[i] * scale;
            self.m_bias[i] = beta1 * self.m_bias[i] + (1.0 - beta1) * g;
            self.v_bias[i] = beta2 * self.v_bias[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m_bias[i] / (1.0 - beta1.powf(t));
            let v_hat = self.v_bias[i] / (1.0 - beta2.powf(t));
            self.bias[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        // Zero gradients after applying
        for row in &mut self.grad_weight {
            for g in row.iter_mut() {
                *g = 0.0;
            }
        }
        for g in &mut self.grad_bias {
            *g = 0.0;
        }
    }

    pub fn clip_gradients(&mut self, max_norm: f32) {
        // Compute global norm
        let mut norm_sq = 0.0;
        for row in &self.grad_weight {
            for g in row {
                norm_sq += g * g;
            }
        }
        for g in &self.grad_bias {
            norm_sq += g * g;
        }
        let norm = norm_sq.sqrt();
        if norm > max_norm {
            let scale = max_norm / norm;
            for row in &mut self.grad_weight {
                for g in row.iter_mut() {
                    *g *= scale;
                }
            }
            for g in &mut self.grad_bias {
                *g *= scale;
            }
        }
    }

    pub fn backward(&mut self, input: &[f32], grad_output: &[f32]) -> Vec<f32> {
        let in_dim = input.len();
        let out_dim = grad_output.len();
        let grad_w: Vec<Vec<f32>> = (0..out_dim)
            .into_par_iter()
            .map(|i| (0..in_dim).map(|j| grad_output[i] * input[j]).collect())
            .collect();
        let grad_b: Vec<f32> = grad_output.par_iter().copied().collect();
        let grad_input: Vec<f32> = (0..in_dim)
            .into_par_iter()
            .map(|j| {
                (0..out_dim)
                    .map(|i| grad_output[i] * self.weight[i][j])
                    .sum::<f32>()
            })
            .collect();
        self.accumulate_grad(&grad_w, &grad_b);
        grad_input
    }
}
