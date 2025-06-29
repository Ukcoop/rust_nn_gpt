//! GPU-based transformer layers (to be implemented with wgpu)

use rand::Rng;

pub mod blocks;
pub mod gpu_utils;
pub mod wgpu_utils;

pub use blocks::*;

/// GPU-based transformer implementation using wgpu
pub struct GpuTransformer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub input_proj: LinearLayer,
    pub output_proj: LinearLayer,
    pub in_dim: usize,
    pub out_dim: usize,
    pub height: usize,
    pub input_weights: wgpu::Buffer,
    pub input_bias: wgpu::Buffer,
    pub output_weights: wgpu::Buffer,
    pub output_bias: wgpu::Buffer,
    pub transformer_stack: Option<TransformerStack>,
}

impl GpuTransformer {
    /// Create a new GpuTransformer from config
    pub async fn new(
        config: &crate::cpu_transformer::TransformerConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let (device, queue) = wgpu_utils::initialize_wgpu().await.map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                format!("Failed to init wgpu: {e}").into()
            },
        )?;
        let in_dim = config.input_dim;
        let out_dim = config.output_dim;
        let height = config.height;
        let input_proj = LinearLayer::new(&device, &queue, in_dim, height).await?;
        let output_proj = LinearLayer::new(&device, &queue, height, out_dim).await?;
        let transformer_stack = if config.layers > 0 {
            Some(TransformerStack::new(&device, &queue, height, height, config.layers).await?)
        } else {
            None
        };
        // Initialize weights and biases with random normals
        let mut rng = rand::thread_rng();
        let normal = rand_distr::Normal::new(0.0, 0.01)
            .map_err(|e| format!("Failed to create normal distribution: {e}"))?;
        let input_weights_vec: Vec<f32> = (0..height * in_dim)
            .map(|_| rng.sample::<f32, _>(normal))
            .collect();
        let input_bias_vec = vec![0.0; height];
        let output_weights_vec: Vec<f32> = (0..out_dim * height)
            .map(|_| rng.sample::<f32, _>(normal))
            .collect();
        let output_bias_vec = vec![0.0; out_dim];
        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let input_weights = wgpu_utils::create_storage_buffer_with_data(
            &device,
            "Input Weights",
            &input_weights_vec,
            Some(storage_usage),
        );
        let input_bias = wgpu_utils::create_storage_buffer_with_data(
            &device,
            "Input Bias",
            &input_bias_vec,
            Some(storage_usage),
        );
        let output_weights = wgpu_utils::create_storage_buffer_with_data(
            &device,
            "Output Weights",
            &output_weights_vec,
            Some(storage_usage),
        );
        let output_bias = wgpu_utils::create_storage_buffer_with_data(
            &device,
            "Output Bias",
            &output_bias_vec,
            Some(storage_usage),
        );
        // Set initial weights/biases in LinearLayers
        input_proj.set_weights(&queue, &input_weights_vec);
        input_proj.set_bias(&queue, &input_bias_vec);
        output_proj.set_weights(&queue, &output_weights_vec);
        output_proj.set_bias(&queue, &output_bias_vec);
        Ok(Self {
            device,
            queue,
            input_proj,
            output_proj,
            in_dim,
            out_dim,
            height,
            input_weights,
            input_bias,
            output_weights,
            output_bias,
            transformer_stack,
        })
    }
    /// Forward pass: input -> output
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let x_proj = self.project_input(input);
        let x_stack = self.forward_stack(&x_proj);
        self.project_output(&x_stack)
    }
    /// Train for a single input/target pair (MSE loss)
    pub fn train(&mut self, input: &[f32], target: &[f32], lr: f32) -> f32 {
        // 1. Forward pass through the model
        let output = self.forward(input);
        // 2. Compute mean squared error loss
        let loss = self.compute_loss(&output, target);
        // 3. Backward pass and parameter updates (hybrid pattern)
        let x_proj = self.project_input(input);
        let x_stack = self.forward_stack(&x_proj);
        let grad_output = self.compute_grad_output(&output, target);
        let output_weights = wgpu_utils::read_buffer_f32(
            &self.device,
            &self.queue,
            &self.output_proj.weights_buf,
            self.output_proj.out_dim * self.output_proj.in_dim,
        );
        let (output_grads, d_stack_out) = self.output_proj.backward_gpu(
            &self.device,
            &self.queue,
            &x_stack,
            &grad_output,
            &output_weights,
        );
        self.output_proj
            .update_parameters(&self.device, &self.queue, &output_grads, lr);
        let grad_input_proj = self.compute_grad_input_proj(&d_stack_out);
        let input_weights = wgpu_utils::read_buffer_f32(
            &self.device,
            &self.queue,
            &self.input_proj.weights_buf,
            self.input_proj.out_dim * self.input_proj.in_dim,
        );
        let (input_grads, _) = self.input_proj.backward_gpu(
            &self.device,
            &self.queue,
            input,
            &grad_input_proj,
            &input_weights,
        );
        self.input_proj
            .update_parameters(&self.device, &self.queue, &input_grads, lr);
        // 4. Return the average of all losses
        loss
    }
    // --- Private helper functions ---
    fn project_input(&self, input: &[f32]) -> Vec<f32> {
        self.input_proj
            .forward_vec(&self.device, &self.queue, input)
    }
    fn forward_stack(&self, x_proj: &[f32]) -> Vec<f32> {
        if let Some(stack) = &self.transformer_stack {
            stack.forward(&self.device, &self.queue, &[x_proj.to_vec()])[0].clone()
        } else {
            x_proj.to_vec()
        }
    }
    fn project_output(&self, x_stack: &[f32]) -> Vec<f32> {
        assert_eq!(
            x_stack.len(),
            self.output_proj.in_dim,
            "x_stack.len() must match output_proj.in_dim"
        );
        self.output_proj
            .forward_vec(&self.device, &self.queue, x_stack)
    }
    fn compute_loss(&self, output: &[f32], target: &[f32]) -> f32 {
        output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| {
                let diff = o - t;
                diff * diff
            })
            .sum::<f32>()
            / output.len() as f32
    }
    fn compute_grad_output(&self, output: &[f32], target: &[f32]) -> Vec<f32> {
        (0..self.out_dim)
            .map(|i| 2.0 * (output[i] - target[i]) / self.out_dim as f32)
            .collect()
    }
    fn compute_grad_input_proj(&self, d_stack_out: &[f32]) -> Vec<f32> {
        let mut grad_input_proj = vec![0.0; self.height];
        grad_input_proj[..self.height.min(d_stack_out.len())]
            .copy_from_slice(&d_stack_out[..self.height.min(d_stack_out.len())]);
        grad_input_proj
    }
}
