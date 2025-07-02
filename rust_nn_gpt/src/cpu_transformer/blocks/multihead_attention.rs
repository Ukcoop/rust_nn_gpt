use super::linear::Linear;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

type HeadResult2 = (Vec<Vec<f32>>, Vec<Vec<f32>>);
type HeadResult3 = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);

#[derive(Serialize, Deserialize)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    #[serde(skip)]
    pub last_input: Option<Vec<Vec<f32>>>,
    #[serde(skip)]
    pub last_q: Option<Vec<Vec<f32>>>,
    #[serde(skip)]
    pub last_k: Option<Vec<Vec<f32>>>,
    #[serde(skip)]
    pub last_v: Option<Vec<Vec<f32>>>,
    #[serde(skip)]
    pub last_attn_weights: Option<Vec<Vec<Vec<f32>>>>,
    #[serde(skip)]
    pub last_heads_out: Option<Vec<Vec<f32>>>,
}

impl MultiHeadAttention {
    pub fn new(model_dim: usize, num_heads: usize) -> Self {
        let head_dim = model_dim / num_heads;
        MultiHeadAttention {
            num_heads,
            head_dim,
            w_q: Linear::new(model_dim, model_dim),
            w_k: Linear::new(model_dim, model_dim),
            w_v: Linear::new(model_dim, model_dim),
            w_o: Linear::new(model_dim, model_dim),
            last_input: None,
            last_q: None,
            last_k: None,
            last_v: None,
            last_attn_weights: None,
            last_heads_out: None,
        }
    }

    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = x.len();
        let model_dim = self.num_heads * self.head_dim;
        let q: Vec<Vec<f32>> = x.iter().map(|xi| self.w_q.forward(xi)).collect();
        let k: Vec<Vec<f32>> = x.iter().map(|xi| self.w_k.forward(xi)).collect();
        let v: Vec<Vec<f32>> = x.iter().map(|xi| self.w_v.forward(xi)).collect();
        // Parallelize over heads, each thread returns (attn_weights_head, heads_out_head)
        let head_results: Vec<HeadResult2> = (0..self.num_heads)
            .into_par_iter()
            .map(|head| {
                let start = head * self.head_dim;
                let end = start + self.head_dim;
                let mut attn_weights_head = vec![vec![0.0; seq_len]; seq_len];
                let mut heads_out_head = vec![vec![0.0; self.head_dim]; seq_len];
                for i in 0..seq_len {
                    let qi = &q[i][start..end];
                    let scores: Vec<f32> = (0..seq_len)
                        .map(|j| {
                            let kj = &k[j][start..end];
                            dot(qi, kj) / (self.head_dim as f32).sqrt()
                        })
                        .collect();
                    let attn_weights = softmax(&scores);
                    attn_weights_head[i] = attn_weights.clone();
                    let mut attn_out = vec![0.0; self.head_dim];
                    for j in 0..seq_len {
                        for d in 0..self.head_dim {
                            attn_out[d] += attn_weights[j] * v[j][start + d];
                        }
                    }
                    heads_out_head[i].copy_from_slice(&attn_out[..self.head_dim]);
                }
                (attn_weights_head, heads_out_head)
            })
            .collect();
        // Merge results
        let mut attn_weights_all = vec![vec![vec![0.0; seq_len]; seq_len]; self.num_heads];
        let mut heads_out = vec![vec![0.0; model_dim]; seq_len];
        for (head, (attn_weights_head, heads_out_head)) in head_results.into_iter().enumerate() {
            let start = head * self.head_dim;
            let end = start + self.head_dim;
            for i in 0..seq_len {
                attn_weights_all[head][i] = attn_weights_head[i].clone();
                heads_out[i][start..end].copy_from_slice(&heads_out_head[i]);
            }
        }
        let out: Vec<Vec<f32>> = heads_out.iter().map(|h| self.w_o.forward(h)).collect();
        self.last_input = Some(x.to_vec());
        self.last_q = Some(q);
        self.last_k = Some(k);
        self.last_v = Some(v);
        self.last_attn_weights = Some(attn_weights_all);
        self.last_heads_out = Some(heads_out);
        out
    }

    pub fn apply_gradients(&mut self, lr: f32, scale: f32) {
        self.w_q.apply_gradients(lr, scale);
        self.w_k.apply_gradients(lr, scale);
        self.w_v.apply_gradients(lr, scale);
        self.w_o.apply_gradients(lr, scale);
    }

    pub fn apply_gradients_adam(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, scale: f32) {
        self.w_q.apply_gradients_adam(lr, beta1, beta2, eps, scale);
        self.w_k.apply_gradients_adam(lr, beta1, beta2, eps, scale);
        self.w_v.apply_gradients_adam(lr, beta1, beta2, eps, scale);
        self.w_o.apply_gradients_adam(lr, beta1, beta2, eps, scale);
    }

    pub fn backward(
        &mut self,
        grad_output: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let x = self
            .last_input
            .as_ref()
            .ok_or_else(|| Box::<dyn std::error::Error>::from("Call forward before backward"))?;
        let q = self
            .last_q
            .as_ref()
            .ok_or_else(|| Box::<dyn std::error::Error>::from("Call forward before backward"))?;
        let k = self
            .last_k
            .as_ref()
            .ok_or_else(|| Box::<dyn std::error::Error>::from("Call forward before backward"))?;
        let v = self
            .last_v
            .as_ref()
            .ok_or_else(|| Box::<dyn std::error::Error>::from("Call forward before backward"))?;
        let attn_weights_all = self
            .last_attn_weights
            .as_ref()
            .ok_or_else(|| Box::<dyn std::error::Error>::from("Call forward before backward"))?;
        let heads_out = self
            .last_heads_out
            .as_ref()
            .ok_or_else(|| Box::<dyn std::error::Error>::from("Call forward before backward"))?;
        let seq_len = x.len();
        let model_dim = self.num_heads * self.head_dim;
        let grad_heads_out = self.w_o.backward(&heads_out[0], grad_output);
        let mut grad_input = vec![0.0; model_dim];
        let head_results: Vec<HeadResult3> = (0..self.num_heads)
            .into_par_iter()
            .map(|head| {
                let start = head * self.head_dim;
                let end = start + self.head_dim;
                let mut grad_q_head = vec![vec![0.0; model_dim]; seq_len];
                let mut grad_k_head = vec![vec![0.0; model_dim]; seq_len];
                let mut grad_v_head = vec![vec![0.0; model_dim]; seq_len];
                for i in 0..seq_len {
                    let grad_attn_out = &grad_heads_out[start..end];
                    let attn_weights = &attn_weights_all[head][i];
                    let mut grad_attn_weights = vec![0.0; seq_len];
                    for j in 0..seq_len {
                        for d in 0..self.head_dim {
                            grad_attn_weights[j] += grad_attn_out[d] * v[j][start + d];
                        }
                    }
                    for j in 0..seq_len {
                        for d in 0..self.head_dim {
                            grad_v_head[j][start + d] += attn_weights[j] * grad_attn_out[d];
                        }
                    }
                    let qi = &q[i][start..end];
                    let mut scores = vec![0.0; seq_len];
                    for j in 0..seq_len {
                        let kj = &k[j][start..end];
                        scores[j] = dot(qi, kj) / (self.head_dim as f32).sqrt();
                    }
                    let softmax_scores = softmax(&scores);
                    let mut grad_scores = vec![0.0; seq_len];
                    for j in 0..seq_len {
                        for l in 0..seq_len {
                            let delta = if j == l { 1.0 } else { 0.0 };
                            grad_scores[j] += softmax_scores[l]
                                * (delta - softmax_scores[j])
                                * grad_attn_weights[l];
                        }
                    }
                    for j in 0..seq_len {
                        let kj = &k[j][start..end];
                        let scale = 1.0 / (self.head_dim as f32).sqrt();
                        for d in 0..self.head_dim {
                            grad_q_head[i][start + d] += grad_scores[j] * kj[d] * scale;
                            grad_k_head[j][start + d] += grad_scores[j] * qi[d] * scale;
                        }
                    }
                }
                (grad_q_head, grad_k_head, grad_v_head)
            })
            .collect();
        let mut grad_q_all = vec![vec![0.0; model_dim]; seq_len];
        let mut grad_k_all = vec![vec![0.0; model_dim]; seq_len];
        let mut grad_v_all = vec![vec![0.0; model_dim]; seq_len];
        for (grad_q_head, grad_k_head, grad_v_head) in head_results {
            for i in 0..seq_len {
                for d in 0..model_dim {
                    grad_q_all[i][d] += grad_q_head[i][d];
                    grad_k_all[i][d] += grad_k_head[i][d];
                    grad_v_all[i][d] += grad_v_head[i][d];
                }
            }
        }
        for i in 0..seq_len {
            let grad_q = &grad_q_all[i];
            let grad_k = &grad_k_all[i];
            let grad_v = &grad_v_all[i];
            let grad_input_q = self.w_q.backward(&x[i], grad_q);
            let grad_input_k = self.w_k.backward(&x[i], grad_k);
            let grad_input_v = self.w_v.backward(&x[i], grad_v);
            for d in 0..model_dim {
                grad_input[d] += grad_input_q[d] + grad_input_k[d] + grad_input_v[d];
            }
        }
        Ok(grad_input)
    }

    pub fn clip_gradients(&mut self, max_norm: f32) {
        self.w_q.clip_gradients(max_norm);
        self.w_k.clip_gradients(max_norm);
        self.w_v.clip_gradients(max_norm);
        self.w_o.clip_gradients(max_norm);
    }

    pub fn zero_grad(&mut self) {
        self.w_q.zero_grad();
        self.w_k.zero_grad();
        self.w_v.zero_grad();
        self.w_o.zero_grad();
    }
}
