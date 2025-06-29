use super::{FeedForward, LayerNorm, MultiHeadAttention};
use crate::gpu_transformer::wgpu_utils;

/// A single transformer block for the GPU backend
pub struct TransformerBlock {
    pub mha: MultiHeadAttention,
    pub norm1: LayerNorm,
    pub ff: FeedForward,
    pub norm2: LayerNorm,
    // TODO: Add more fields as needed (e.g., residual buffers)
}

impl TransformerBlock {
    /// Create a new TransformerBlock (scaffold, not yet functional)
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dim: usize,
        num_heads: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mha = MultiHeadAttention::new(device, queue, dim, dim, num_heads).await?;
        let norm1 = LayerNorm::new(device, queue, dim).await?;
        let ff = FeedForward::new(device, dim, dim, dim).await?;
        let norm2 = LayerNorm::new(device, queue, dim).await?;
        Ok(Self {
            mha,
            norm1,
            ff,
            norm2,
        })
    }
    /// Forward pass for 2D input: input is [input_dim][height], returns [input_dim][height]
    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        input
            .iter()
            .map(|row| self.forward_row(device, queue, row))
            .collect()
    }
    /// Train for 2D input/target: both are [input_dim][height], returns average loss
    pub fn train(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[Vec<f32>],
        target: &[Vec<f32>],
        lr: f32,
    ) -> f32 {
        assert_eq!(input.len(), target.len());
        let mut total_loss = 0.0;
        let n = input.len();
        for (row, tgt) in input.iter().zip(target.iter()) {
            total_loss += self.train_row(device, queue, row, tgt, lr);
        }
        total_loss / (n as f32 * 2.0)
    }
    // --- Private helper functions ---
    fn forward_row(&self, device: &wgpu::Device, queue: &wgpu::Queue, row: &[f32]) -> Vec<f32> {
        let normed1 = self.apply_norm1(queue, device, row);
        let attn_out = self.apply_attention(queue, device, &normed1);
        let res1 = self.add_residual(row, &attn_out);
        let normed2 = self.apply_norm2(queue, device, &res1);
        let ff_out = self.apply_feedforward(queue, device, &normed2);
        let res2 = self.add_residual(&normed2, &ff_out);
        self.final_norm(queue, device, &res2)
    }
    fn apply_norm1(&self, queue: &wgpu::Queue, device: &wgpu::Device, row: &[f32]) -> Vec<f32> {
        self.norm1.set_input(queue, row);
        self.norm1.forward(device, queue)
    }
    fn apply_attention(
        &self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        normed1: &[f32],
    ) -> Vec<f32> {
        self.mha.set_input(queue, normed1);
        let (_, _, _, _, attn_out) = self.mha.forward_with_intermediates(device, queue, normed1);
        attn_out
    }
    fn add_residual(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }
    fn apply_norm2(&self, queue: &wgpu::Queue, device: &wgpu::Device, res1: &[f32]) -> Vec<f32> {
        self.norm2.set_input(queue, res1);
        self.norm2.forward(device, queue)
    }
    fn apply_feedforward(
        &self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        normed2: &[f32],
    ) -> Vec<f32> {
        self.ff.set_input(queue, normed2);
        self.ff.forward(device, queue)
    }
    fn final_norm(&self, queue: &wgpu::Queue, device: &wgpu::Device, res2: &[f32]) -> Vec<f32> {
        self.norm2.set_input(queue, res2);
        self.norm2.forward(device, queue)
    }
    fn train_row(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        row: &[f32],
        tgt: &[f32],
        lr: f32,
    ) -> f32 {
        // norm1
        let mean1 = row.iter().sum::<f32>() / row.len() as f32;
        let var1 = row.iter().map(|x| (x - mean1).powi(2)).sum::<f32>() / row.len() as f32;
        let gamma1 =
            wgpu_utils::read_buffer_f32(device, queue, &self.norm1.gamma_buf, self.norm1.dim);
        let d_output1 = row.iter().map(|_| 1.0).collect::<Vec<f32>>(); // Placeholder, should be real grad from next layer
        let (norm1_grads, _) =
            self.norm1
                .backward_gpu(device, queue, row, &[mean1], &[var1], &gamma1, &d_output1);
        self.norm1
            .update_parameters(device, queue, &norm1_grads, lr);
        self.norm1.set_input(queue, row);
        let normed1 = self.norm1.forward(device, queue);
        // mha
        self.mha.set_input(queue, &normed1);
        let (_, _, _, _, attn_out) = self.mha.forward_with_intermediates(device, queue, &normed1);
        // norm2
        let mean2 = attn_out.iter().sum::<f32>() / attn_out.len() as f32;
        let var2 =
            attn_out.iter().map(|x| (x - mean2).powi(2)).sum::<f32>() / attn_out.len() as f32;
        let gamma2 =
            wgpu_utils::read_buffer_f32(device, queue, &self.norm2.gamma_buf, self.norm2.dim);
        // ff
        self.norm2.set_input(queue, &attn_out);
        let normed2 = self.norm2.forward(device, queue);
        self.ff.set_input(queue, &normed2);
        let w2 = wgpu_utils::read_buffer_f32(
            device,
            queue,
            &self.ff.w2_buf,
            self.ff.out_dim * self.ff.hidden_dim,
        );
        let b2 = wgpu_utils::read_buffer_f32(device, queue, &self.ff.b2_buf, self.ff.out_dim);
        let w1 = wgpu_utils::read_buffer_f32(
            device,
            queue,
            &self.ff.w1_buf,
            self.ff.hidden_dim * self.ff.in_dim,
        );
        let b1 = wgpu_utils::read_buffer_f32(device, queue, &self.ff.b1_buf, self.ff.hidden_dim);
        let mut pre_hidden = vec![0.0; self.ff.hidden_dim];
        let mut hidden = vec![0.0; self.ff.hidden_dim];
        for i in 0..self.ff.hidden_dim {
            let mut sum = 0.0;
            for j in 0..self.ff.in_dim {
                sum += w1[i * self.ff.in_dim + j] * normed2[j];
            }
            pre_hidden[i] = sum + b1[i];
            hidden[i] = FeedForward::gelu(pre_hidden[i]);
        }
        let mut output = vec![0.0; self.ff.out_dim];
        for i in 0..self.ff.out_dim {
            let mut sum = 0.0;
            for j in 0..self.ff.hidden_dim {
                sum += w2[i * self.ff.hidden_dim + j] * hidden[j];
            }
            output[i] = sum + b2[i];
        }
        // Compute grad_output (MSE loss)
        let mut grad_output = vec![0.0; self.ff.out_dim];
        let mut ff_loss = 0.0;
        for i in 0..self.ff.out_dim {
            let diff = output[i] - tgt[i];
            ff_loss += diff * diff;
            grad_output[i] = 2.0 * diff / self.ff.out_dim as f32;
        }
        ff_loss /= self.ff.out_dim as f32;
        // GPU backward pass for ff
        let ff_grads = self.ff.backward_gpu(
            device,
            queue,
            &normed2,
            &pre_hidden,
            &hidden,
            &grad_output,
            &w2,
            &w1,
        );
        self.ff.update_parameters(device, queue, &ff_grads, lr);
        //let _ff_out = self.ff.forward_cpu(device, queue, &normed2);
        // Backward pass for norm2
        let (norm2_grads, _d_input2) = self.norm2.backward_gpu(
            device,
            queue,
            &attn_out,
            &[mean2],
            &[var2],
            &gamma2,
            &grad_output,
        );
        self.norm2
            .update_parameters(device, queue, &norm2_grads, lr);
        // Accumulate losses (no final_norm)
        ff_loss
    }
}
// TODO: Add more methods for training, parameter updates, etc.
