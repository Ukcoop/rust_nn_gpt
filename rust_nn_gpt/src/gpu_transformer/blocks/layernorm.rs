use crate::gpu_transformer::gpu_utils::write_f32_slice;
use crate::gpu_transformer::wgpu_utils;

pub struct LayerNorm {
    pub dim: usize,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub input_buf: wgpu::Buffer,
    pub output_buf: wgpu::Buffer,
    pub gamma_buf: wgpu::Buffer,
    pub beta_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    // Persistent backward buffers
    pub bw_input_buf: wgpu::Buffer,
    pub bw_mean_buf: wgpu::Buffer,
    pub bw_variance_buf: wgpu::Buffer,
    pub bw_gamma_buf: wgpu::Buffer,
    pub bw_d_output_buf: wgpu::Buffer,
    pub bw_d_gamma_buf: wgpu::Buffer,
    pub bw_d_beta_buf: wgpu::Buffer,
    pub bw_d_input_buf: wgpu::Buffer,
    pub bw_bind_group_layout: wgpu::BindGroupLayout,
    pub bw_pipeline: wgpu::ComputePipeline,
    pub bw_bind_group: wgpu::BindGroup,
}

pub struct LayerNormGradients {
    pub d_gamma: Vec<f32>,
    pub d_beta: Vec<f32>,
}

/// Layer normalization for the GPU backend
impl LayerNorm {
    /// Create a new LayerNorm layer (scaffold, not yet functional)
    pub async fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Use include_str! for the layernorm WGSL shader
        let shader_src = include_str!("../shaders/layernorm.wgsl");
        let _shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LayerNorm Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        // --- Forward pass resources (output buffer is read_write!) ---
        let forward_specs = [
            wgpu_utils::BufferSpec {
                label: "LayerNorm Input",
                size: dim * std::mem::size_of::<f32>(),
                usage: None,
            },
            wgpu_utils::BufferSpec {
                label: "LayerNorm Gamma",
                size: dim * std::mem::size_of::<f32>(),
                usage: Some(
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                ),
            },
            wgpu_utils::BufferSpec {
                label: "LayerNorm Beta",
                size: dim * std::mem::size_of::<f32>(),
                usage: Some(
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                ),
            },
            wgpu_utils::BufferSpec {
                label: "LayerNorm Output",
                size: dim * std::mem::size_of::<f32>(),
                usage: Some(
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                ),
            },
        ];
        // The output buffer (index 3) must be read_write (read_only: false) to match the WGSL shader
        let forward_descs = [
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // input
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // gamma
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // beta
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            }, // output (read_write!)
        ];
        let forward = wgpu_utils::create_layer_gpu_resources(
            device,
            &forward_specs,
            &forward_descs,
            "LayerNorm BindGroup",
            "LayerNorm BGL",
        );
        let bind_group_layout = forward.bind_group_layout;
        let bind_group = forward.bind_group;
        let mut buffers = forward.buffers.into_iter();
        let input_buf = buffers.next().ok_or("input_buf missing")?;
        let gamma_buf = buffers.next().ok_or("gamma_buf missing")?;
        let beta_buf = buffers.next().ok_or("beta_buf missing")?;
        let output_buf = buffers.next().ok_or("output_buf missing")?;
        // Deduplicated backward buffer and bind group creation
        let bw_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let bw_specs = [
            wgpu_utils::BufferSpec {
                label: "LayerNorm Backward Input",
                size: dim * std::mem::size_of::<f32>(),
                usage: Some(bw_usage),
            },
            wgpu_utils::BufferSpec {
                label: "LayerNorm Backward Mean",
                size: dim * std::mem::size_of::<f32>(),
                usage: Some(bw_usage),
            },
            wgpu_utils::BufferSpec {
                label: "LayerNorm Backward Variance",
                size: dim * std::mem::size_of::<f32>(),
                usage: Some(bw_usage),
            },
            wgpu_utils::BufferSpec {
                label: "LayerNorm Backward Gamma",
                size: dim * std::mem::size_of::<f32>(),
                usage: Some(bw_usage),
            },
            wgpu_utils::BufferSpec {
                label: "LayerNorm Backward d_output",
                size: dim * std::mem::size_of::<f32>(),
                usage: Some(bw_usage),
            },
            wgpu_utils::BufferSpec {
                label: "LayerNorm Backward d_gamma",
                size: dim * std::mem::size_of::<f32>(),
                usage: Some(bw_usage),
            },
            wgpu_utils::BufferSpec {
                label: "LayerNorm Backward d_beta",
                size: dim * std::mem::size_of::<f32>(),
                usage: Some(bw_usage),
            },
            wgpu_utils::BufferSpec {
                label: "LayerNorm Backward d_input",
                size: dim * std::mem::size_of::<f32>(),
                usage: Some(bw_usage),
            },
        ];
        let bw_descs = [
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            },
        ];
        let bw = wgpu_utils::create_layer_gpu_resources(
            device,
            &bw_specs,
            &bw_descs,
            "LayerNorm Backward BindGroup",
            "LayerNorm Backward BGL",
        );
        let bw_bind_group_layout = bw.bind_group_layout;
        let bw_bind_group = bw.bind_group;
        let mut bw_buffers = bw.buffers.into_iter();
        let bw_input_buf = bw_buffers.next().ok_or("bw_input_buf missing")?;
        let bw_mean_buf = bw_buffers.next().ok_or("bw_mean_buf missing")?;
        let bw_variance_buf = bw_buffers.next().ok_or("bw_variance_buf missing")?;
        let bw_gamma_buf = bw_buffers.next().ok_or("bw_gamma_buf missing")?;
        let bw_d_output_buf = bw_buffers.next().ok_or("bw_d_output_buf missing")?;
        let bw_d_gamma_buf = bw_buffers.next().ok_or("bw_d_gamma_buf missing")?;
        let bw_d_beta_buf = bw_buffers.next().ok_or("bw_d_beta_buf missing")?;
        let bw_d_input_buf = bw_buffers.next().ok_or("bw_d_input_buf missing")?;
        // Backward pipeline and bind group
        let shader_src = std::fs::read_to_string(
            "rust_nn_gpt/src/gpu_transformer/shaders/layernorm_backward.wgsl",
        )?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LayerNorm Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let _bw_bgl_entries = wgpu_utils::make_bgl_entries(&[
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            },
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            },
        ]);
        let bw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LayerNorm Backward Pipeline Layout"),
            bind_group_layouts: &[&bw_bind_group_layout],
            push_constant_ranges: &[],
        });
        let bw_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LayerNorm Backward Pipeline"),
            layout: Some(&bw_pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });
        // Forward pipeline: must use the forward bind group layout (with output as read_write)
        let shader_src = include_str!("../shaders/layernorm.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LayerNorm Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LayerNorm Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout], // <-- This is the forward layout
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LayerNorm Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });
        // Debug assertion: output buffer must be read_write in the layout
        debug_assert!(
            !forward_descs[3].read_only,
            "Output buffer must be read_write for LayerNorm forward!"
        );
        Ok(Self {
            dim,
            pipeline,
            bind_group_layout,
            input_buf,
            output_buf,
            gamma_buf,
            beta_buf,
            bind_group,
            bw_input_buf,
            bw_mean_buf,
            bw_variance_buf,
            bw_gamma_buf,
            bw_d_output_buf,
            bw_d_gamma_buf,
            bw_d_beta_buf,
            bw_d_input_buf,
            bw_bind_group_layout,
            bw_pipeline,
            bw_bind_group,
        })
    }
    /// Forward pass (to be implemented)
    pub fn forward(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        // Encode and submit
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LayerNorm Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LayerNorm Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(((self.dim as f32) / 64.0).ceil() as u32, 1, 1);
        }
        // Create staging buffer for readback
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LayerNorm Staging Buffer"),
            size: (self.dim * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Copy output buffer to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.output_buf,
            0,
            &staging_buf,
            0,
            (self.dim * std::mem::size_of::<f32>()) as u64,
        );
        queue.submit(Some(encoder.finish()));
        // Read back from staging buffer
        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            if sender.send(v).is_err() {
                eprintln!("Failed to send map_async result");
            }
        });
        let _ = device.poll(wgpu::MaintainBase::Wait);
        futures::executor::block_on(receiver.receive());
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();
        result
    }
    /// Set input buffer
    pub fn set_input(&self, queue: &wgpu::Queue, input: &[f32]) {
        assert_eq!(input.len(), self.dim);
        write_f32_slice(queue, &self.input_buf, input);
    }
    /// Set gamma buffer
    pub fn set_gamma(&self, queue: &wgpu::Queue, gamma: &[f32]) {
        assert_eq!(gamma.len(), self.dim);
        write_f32_slice(queue, &self.gamma_buf, gamma);
    }
    /// Set beta buffer
    pub fn set_beta(&self, queue: &wgpu::Queue, beta: &[f32]) {
        assert_eq!(beta.len(), self.dim);
        write_f32_slice(queue, &self.beta_buf, beta);
    }
    /// Train gamma and beta with SGD: input, target, learning rate. Returns loss.
    pub fn train(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[f32],
        target: &[f32],
        lr: f32,
    ) -> f32 {
        assert_eq!(input.len(), self.dim);
        assert_eq!(target.len(), self.dim);
        // Read gamma and beta
        let mut gamma = wgpu_utils::read_buffer_f32(device, queue, &self.gamma_buf, self.dim);
        let mut beta = wgpu_utils::read_buffer_f32(device, queue, &self.beta_buf, self.dim);
        // Forward pass: y = gamma * norm(x) + beta
        let mean = input.iter().sum::<f32>() / self.dim as f32;
        let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.dim as f32;
        let eps = 1e-5;
        let std = (var + eps).sqrt();
        let normed: Vec<f32> = input.iter().map(|x| (x - mean) / std).collect();
        let output: Vec<f32> = normed
            .iter()
            .zip(gamma.iter())
            .map(|(n, g)| n * g)
            .zip(beta.iter())
            .map(|(ng, b)| ng + b)
            .collect();
        // Compute loss and gradient (MSE)
        let mut loss = 0.0;
        let mut grad_output = vec![0.0; self.dim];
        for i in 0..self.dim {
            let diff = output[i] - target[i];
            loss += diff * diff;
            grad_output[i] = 2.0 * diff / self.dim as f32;
        }
        // Gradients for gamma and beta
        let grad_gamma: Vec<f32> = normed
            .iter()
            .zip(grad_output.iter())
            .map(|(n, g)| n * g)
            .collect();
        let grad_beta = grad_output.clone();
        // SGD update
        for i in 0..self.dim {
            gamma[i] -= lr * grad_gamma[i];
            beta[i] -= lr * grad_beta[i];
        }
        self.set_gamma(queue, &gamma);
        self.set_beta(queue, &beta);
        loss / self.dim as f32
    }
    pub fn compute_gradients(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[f32],
        target: &[f32],
    ) -> (LayerNormGradients, f32) {
        let dim = self.dim;
        let gamma = wgpu_utils::read_buffer_f32(device, queue, &self.gamma_buf, dim);
        let beta = wgpu_utils::read_buffer_f32(device, queue, &self.beta_buf, dim);
        // Forward pass: y = gamma * norm(x) + beta
        let mean = input.iter().sum::<f32>() / dim as f32;
        let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / dim as f32;
        let eps = 1e-5;
        let mut normed = vec![0.0; dim];
        for i in 0..dim {
            normed[i] = (input[i] - mean) / (var + eps).sqrt();
        }
        let mut output = vec![0.0; dim];
        for i in 0..dim {
            output[i] = gamma[i] * normed[i] + beta[i];
        }
        // Compute loss and grad_output
        let mut grad_output = vec![0.0; dim];
        let mut loss = 0.0;
        for i in 0..dim {
            let diff = output[i] - target[i];
            loss += diff * diff;
            grad_output[i] = 2.0 * diff / dim as f32;
        }
        loss /= dim as f32;
        // Gradients for gamma and beta
        let d_gamma: Vec<f32> = normed
            .iter()
            .zip(grad_output.iter())
            .map(|(n, g)| n * g)
            .collect();
        let d_beta = grad_output.clone();
        (LayerNormGradients { d_gamma, d_beta }, loss)
    }
    pub fn update_parameters(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gradients: &LayerNormGradients,
        lr: f32,
    ) {
        let mut gamma = wgpu_utils::read_buffer_f32(device, queue, &self.gamma_buf, self.dim);
        let mut beta = wgpu_utils::read_buffer_f32(device, queue, &self.beta_buf, self.dim);
        for i in 0..self.dim {
            gamma[i] -= lr * gradients.d_gamma[i];
            beta[i] -= lr * gradients.d_beta[i];
        }
        self.set_gamma(queue, &gamma);
        self.set_beta(queue, &beta);
    }
    #[allow(clippy::too_many_arguments)]
    pub fn backward_gpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[f32],
        mean: &[f32],
        var: &[f32],
        gamma: &[f32],
        d_output: &[f32],
    ) -> (LayerNormGradients, Vec<f32>) {
        let dim = self.dim;
        // Write data to persistent buffers
        queue.write_buffer(&self.bw_input_buf, 0, bytemuck::cast_slice(input));
        queue.write_buffer(&self.bw_mean_buf, 0, bytemuck::cast_slice(mean));
        queue.write_buffer(&self.bw_variance_buf, 0, bytemuck::cast_slice(var));
        queue.write_buffer(&self.bw_gamma_buf, 0, bytemuck::cast_slice(gamma));
        queue.write_buffer(&self.bw_d_output_buf, 0, bytemuck::cast_slice(d_output));
        // Dispatch
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LayerNorm Backward Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LayerNorm Backward Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bw_pipeline);
            cpass.set_bind_group(0, &self.bw_bind_group, &[]);
            cpass.dispatch_workgroups(((dim as f32) / 64.0).ceil() as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        // Read back gradients
        let d_gamma = wgpu_utils::read_buffer_f32(device, queue, &self.bw_d_gamma_buf, dim);
        let d_beta = wgpu_utils::read_buffer_f32(device, queue, &self.bw_d_beta_buf, dim);
        let d_input = wgpu_utils::read_buffer_f32(device, queue, &self.bw_d_input_buf, dim);
        (LayerNormGradients { d_gamma, d_beta }, d_input)
    }
}
// TODO: Add more fields and logic for a real LayerNorm, and implement the WGSL shader.

// Remove the impl Default for LayerNorm block entirely.
