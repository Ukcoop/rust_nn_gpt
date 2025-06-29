use super::TransformerBlock;

pub struct TransformerStack {
    pub blocks: Vec<TransformerBlock>,
    pub dim: usize,
    pub num_heads: usize,
}

impl TransformerStack {
    /// Create a new TransformerStack with the given number of blocks
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dim: usize,
        num_heads: usize,
        num_layers: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(device, queue, dim, num_heads).await?);
        }
        Ok(Self {
            blocks,
            dim,
            num_heads,
        })
    }
    /// Forward pass through the stack: apply each block in sequence to 2D tensor
    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        self.forward_blocks(device, queue, input)
    }
    /// Train all blocks in the stack sequentially, passing 2D input through each block
    pub fn train(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[Vec<f32>],
        target: &[Vec<f32>],
        lr: f32,
    ) -> f32 {
        self.train_blocks(device, queue, input, target, lr)
    }
    // --- Private helper functions ---
    fn forward_blocks(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let mut x = input.to_vec();
        for block in &self.blocks {
            x = block.forward(device, queue, &x);
        }
        x
    }
    fn train_blocks(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[Vec<f32>],
        target: &[Vec<f32>],
        lr: f32,
    ) -> f32 {
        let n = self.blocks.len();
        let mut xs = Vec::with_capacity(n + 1);
        xs.push(input.to_vec());
        for i in 0..n {
            let x_next = self.blocks[i].forward(device, queue, &xs[i]);
            xs.push(x_next);
        }
        let mut total_loss = 0.0;
        for i in 0..n {
            let block_target = if i == n - 1 {
                target.to_vec()
            } else {
                xs[i + 1].clone()
            };
            total_loss += self.blocks[i].train(device, queue, &xs[i], &block_target, lr);
        }
        total_loss / n as f32
    }
}
// TODO: Add more methods for training, parameter updates, etc.
