use crate::cpu_transformer::config::TransformerConfig;

/// Multi-head self-attention (CPU, reference)
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub embed_dim: usize,
    pub head_dim: usize,
    // [num_heads][head_dim][embed_dim]
    pub w_q: Vec<Vec<Vec<f32>>>,
    pub w_k: Vec<Vec<Vec<f32>>>,
    pub w_v: Vec<Vec<Vec<f32>>>,
    // Final projection: [embed_dim][num_heads * head_dim]
    pub w_o: Vec<Vec<f32>>,
}

impl MultiHeadAttention {
    pub fn new(config: &TransformerConfig) -> Self {
        let embed_dim = config.height;
        let num_heads = 1; // single head for simplicity
        let head_dim = embed_dim / num_heads;
        let w_q = vec![vec![vec![1.0; embed_dim]; head_dim]; num_heads];
        let w_k = vec![vec![vec![1.0; embed_dim]; head_dim]; num_heads];
        let w_v = vec![vec![vec![1.0; embed_dim]; head_dim]; num_heads];
        let w_o = vec![vec![1.0; num_heads * head_dim]; embed_dim];
        Self {
            num_heads,
            embed_dim,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }
    /// Forward pass: input is [embed_dim], output is [embed_dim]
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let all_heads = self.compute_all_heads(input);
        self.project_output(&all_heads)
    }
    /// Backward pass for a single input/target pair (stub for demonstration)
    /// Updates weights using a simple gradient descent step on the output error.
    pub fn backward(&mut self, _input: &[f32], output_grad: &[f32], lr: f32) {
        Self::update_output_proj(&mut self.w_o, output_grad, lr);
        // Full backprop through attention is complex; this is a stub for extensibility.
    }
    /// Compute all attention heads and concatenate their outputs
    fn compute_all_heads(&self, input: &[f32]) -> Vec<f32> {
        let mut all_heads = Vec::with_capacity(self.num_heads * self.head_dim);
        for h in 0..self.num_heads {
            let (q, k, v) = self.compute_qkv(h, input);
            let attn = self.compute_attention(&q, &k);
            let head_out = self.apply_attention(&v, attn);
            all_heads.extend(head_out);
        }
        all_heads
    }
    /// Compute Q, K, V vectors for a given head
    fn compute_qkv(&self, h: usize, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q = Self::matvec(&self.w_q[h], input);
        let k = Self::matvec(&self.w_k[h], input);
        let v = Self::matvec(&self.w_v[h], input);
        (q, k, v)
    }
    /// Compute attention score and apply softmax
    fn compute_attention(&self, q: &[f32], k: &[f32]) -> f32 {
        let score = Self::dot(q, k) / (self.head_dim as f32).sqrt();
        Self::softmax(&[score])[0]
    }
    /// Apply attention weight to value vector
    fn apply_attention(&self, v: &[f32], attn: f32) -> Vec<f32> {
        v.iter().map(|vi| attn * vi).collect()
    }
    /// Project concatenated heads to output
    fn project_output(&self, all_heads: &[f32]) -> Vec<f32> {
        Self::matvec(&self.w_o, all_heads)
    }
    /// Matrix-vector multiplication
    fn matvec(mat: &[Vec<f32>], vec: &[f32]) -> Vec<f32> {
        mat.iter()
            .map(|row| row.iter().zip(vec).map(|(w, x)| w * x).sum())
            .collect()
    }
    /// Dot product of two vectors
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
    /// Softmax over a slice
    fn softmax(xs: &[f32]) -> Vec<f32> {
        let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }
    /// Update output projection weights with gradients and learning rate
    fn update_output_proj(w_o: &mut [Vec<f32>], output_grad: &[f32], lr: f32) {
        for i in 0..w_o.len() {
            for j in 0..w_o[0].len() {
                w_o[i][j] -= lr * output_grad[i];
            }
        }
    }
}
