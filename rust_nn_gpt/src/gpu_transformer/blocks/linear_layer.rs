use crate::gpu_transformer::gpu_utils::{update_bias, update_weights};
use crate::gpu_transformer::wgpu_utils;
use rand::Rng;

const LINEAR_WGSL_PATH: &str = "rust_nn_gpt/src/gpu_transformer/shaders/linear.wgsl";

pub struct LinearLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub input_buf: wgpu::Buffer,
    pub output_buf: wgpu::Buffer,
    pub weights_buf: wgpu::Buffer,
    pub bias_buf: wgpu::Buffer,
    pub in_dim_buf: wgpu::Buffer,
    pub out_dim_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    // Persistent backward buffers
    pub bw_input_buf: wgpu::Buffer,
    pub bw_d_output_buf: wgpu::Buffer,
    pub bw_weights_buf: wgpu::Buffer,
    pub bw_d_w_buf: wgpu::Buffer,
    pub bw_d_b_buf: wgpu::Buffer,
    pub bw_d_input_buf: wgpu::Buffer,
    pub bw_bind_group_layout: wgpu::BindGroupLayout,
    pub bw_pipeline: wgpu::ComputePipeline,
    pub bw_bind_group: wgpu::BindGroup,
}

pub struct LinearLayerGradients {
    pub d_weights: Vec<f32>,
    pub d_bias: Vec<f32>,
}

impl LinearLayer {
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let shader_src = std::fs::read_to_string(LINEAR_WGSL_PATH)
            .map_err(|e| format!("Failed to read linear.wgsl: {e}"))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Linear WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let bgl_entries = wgpu_utils::make_bgl_entries(&[
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // input_buf
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // weights_buf
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // bias_buf
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            }, // output_buf
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: true,
            }, // in_dim_buf
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: true,
            }, // out_dim_buf
        ]);
        let bind_group_layout =
            wgpu_utils::create_bind_group_layout(device, "LinearLayer BGL", &bgl_entries);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LinearLayer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LinearLayer Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });
        // Precompute buffer sizes
        let input_size = in_dim * std::mem::size_of::<f32>();
        let weights_size = out_dim * in_dim * std::mem::size_of::<f32>();
        let bias_size = out_dim * std::mem::size_of::<f32>();
        let output_size = out_dim * std::mem::size_of::<f32>();
        // Persistent buffers
        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        // Deduplicated forward buffer and bind group creation
        let forward_specs = [
            wgpu_utils::BufferSpec {
                label: "Input Buffer",
                size: input_size,
                usage: Some(storage_usage),
            },
            wgpu_utils::BufferSpec {
                label: "Weights Buffer",
                size: weights_size,
                usage: Some(storage_usage),
            },
            wgpu_utils::BufferSpec {
                label: "Bias Buffer",
                size: bias_size,
                usage: Some(storage_usage),
            },
            wgpu_utils::BufferSpec {
                label: "Output Buffer",
                size: output_size,
                usage: Some(storage_usage),
            },
        ];
        let forward_descs = [
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
        ];
        let forward = wgpu_utils::create_layer_gpu_resources(
            device,
            &forward_specs,
            &forward_descs,
            "LinearLayer BindGroup",
            "LinearLayer BGL",
        );
        let bind_group_layout = forward.bind_group_layout;
        let bind_group = forward.bind_group;
        let mut buffers = forward.buffers.into_iter();
        let input_buf = buffers.next().ok_or("input_buf missing")?;
        let weights_buf = buffers.next().ok_or("weights_buf missing")?;
        let bias_buf = buffers.next().ok_or("bias_buf missing")?;
        let output_buf = buffers.next().ok_or("output_buf missing")?;
        let in_dim_buf =
            wgpu_utils::create_uniform_buffer(device, "in_dim Uniform", &[in_dim as u32]);
        let out_dim_buf =
            wgpu_utils::create_uniform_buffer(device, "out_dim Uniform", &[out_dim as u32]);
        let mut rng = rand::thread_rng();
        let normal = rand_distr::Normal::new(0.0, 0.01)
            .map_err(|e| format!("Failed to create normal distribution: {e}"))?;
        let weights_vec: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| rng.sample::<f32, _>(normal))
            .collect();
        queue.write_buffer(&weights_buf, 0, bytemuck::cast_slice(&weights_vec));
        let bias_vec: Vec<f32> = (0..out_dim).map(|_| rng.sample::<f32, _>(normal)).collect();
        queue.write_buffer(&bias_buf, 0, bytemuck::cast_slice(&bias_vec));
        // Deduplicated backward buffer and bind group creation
        let bw_specs = [
            wgpu_utils::BufferSpec {
                label: "Linear Backward Input",
                size: in_dim * std::mem::size_of::<f32>(),
                usage: None,
            },
            wgpu_utils::BufferSpec {
                label: "Linear Backward d_output",
                size: out_dim * std::mem::size_of::<f32>(),
                usage: None,
            },
            wgpu_utils::BufferSpec {
                label: "Linear Backward Weights",
                size: out_dim * in_dim * std::mem::size_of::<f32>(),
                usage: None,
            },
            wgpu_utils::BufferSpec {
                label: "Linear Backward d_w",
                size: out_dim * in_dim * std::mem::size_of::<f32>(),
                usage: Some(storage_usage),
            },
            wgpu_utils::BufferSpec {
                label: "Linear Backward d_b",
                size: out_dim * std::mem::size_of::<f32>(),
                usage: Some(storage_usage),
            },
            wgpu_utils::BufferSpec {
                label: "Linear Backward d_input",
                size: in_dim * std::mem::size_of::<f32>(),
                usage: Some(storage_usage),
            },
        ];
        let bw_descs = [
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // input
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // d_output
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // weights
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            }, // d_w
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            }, // d_b
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            }, // d_input
        ];
        let bw = wgpu_utils::create_layer_gpu_resources(
            device,
            &bw_specs,
            &bw_descs,
            "Linear Backward BindGroup",
            "Linear Backward BGL",
        );
        let bw_bind_group_layout = bw.bind_group_layout;
        let bw_bind_group = bw.bind_group;
        let mut bw_buffers = bw.buffers.into_iter();
        let bw_input_buf = bw_buffers.next().ok_or("bw_input_buf missing")?;
        let bw_d_output_buf = bw_buffers.next().ok_or("bw_d_output_buf missing")?;
        let bw_weights_buf = bw_buffers.next().ok_or("bw_weights_buf missing")?;
        let bw_d_w_buf = bw_buffers.next().ok_or("bw_d_w_buf missing")?;
        let bw_d_b_buf = bw_buffers.next().ok_or("bw_d_b_buf missing")?;
        let bw_d_input_buf = bw_buffers.next().ok_or("bw_d_input_buf missing")?;
        // Create backward pipeline as in the original code
        let bw_shader_src = include_str!("../shaders/linear_backward.wgsl");
        let bw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LinearLayer Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(bw_shader_src.into()),
        });
        let bw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LinearLayer Backward Pipeline Layout"),
            bind_group_layouts: &[&bw_bind_group_layout],
            push_constant_ranges: &[],
        });
        let bw_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LinearLayer Backward Pipeline"),
            layout: Some(&bw_pipeline_layout),
            module: &bw_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });
        Ok(Self {
            in_dim,
            out_dim,
            pipeline,
            bind_group_layout,
            input_buf,
            output_buf,
            weights_buf,
            bias_buf,
            in_dim_buf,
            out_dim_buf,
            bind_group,
            bw_input_buf,
            bw_d_output_buf,
            bw_weights_buf,
            bw_d_w_buf,
            bw_d_b_buf,
            bw_d_input_buf,
            bw_bind_group_layout,
            bw_pipeline,
            bw_bind_group,
        })
    }
    pub fn set_input(&self, queue: &wgpu::Queue, input: &[f32]) {
        assert_eq!(input.len(), self.in_dim);
        let cast_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(input.as_ptr() as *const u8, std::mem::size_of_val(input))
        };
        queue.write_buffer(&self.input_buf, 0, cast_slice);
    }
    pub fn set_weights(&self, queue: &wgpu::Queue, weights: &[f32]) {
        assert_eq!(weights.len(), self.out_dim * self.in_dim);
        let cast_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                weights.as_ptr() as *const u8,
                std::mem::size_of_val(weights),
            )
        };
        queue.write_buffer(&self.weights_buf, 0, cast_slice);
    }
    pub fn set_bias(&self, queue: &wgpu::Queue, bias: &[f32]) {
        assert_eq!(bias.len(), self.out_dim);
        let cast_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(bias.as_ptr() as *const u8, std::mem::size_of_val(bias))
        };
        queue.write_buffer(&self.bias_buf, 0, cast_slice);
    }
    pub fn forward(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        // Encode and submit
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LinearLayer Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LinearLayer Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(((self.out_dim as f32) / 64.0).ceil() as u32, 1, 1);
        }
        // Create staging buffer for readback
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (self.out_dim * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Copy output buffer to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.output_buf,
            0,
            &staging_buf,
            0,
            (self.out_dim * std::mem::size_of::<f32>()) as u64,
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
    /// Compute gradients (on GPU, but here as CPU fallback)
    pub fn compute_gradients(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[f32],
        target: &[f32],
    ) -> (LinearLayerGradients, f32) {
        let weights = wgpu_utils::read_buffer_f32(
            device,
            queue,
            &self.weights_buf,
            self.out_dim * self.in_dim,
        );
        let bias = wgpu_utils::read_buffer_f32(device, queue, &self.bias_buf, self.out_dim);
        let mut output = vec![0.0; self.out_dim];
        for i in 0..self.out_dim {
            let mut sum = 0.0;
            for j in 0..self.in_dim {
                sum += weights[i * self.in_dim + j] * input[j];
            }
            output[i] = sum + bias[i];
        }
        let mut grad_output = vec![0.0; self.out_dim];
        let mut loss = 0.0;
        for i in 0..self.out_dim {
            let diff = output[i] - target[i];
            loss += diff * diff;
            grad_output[i] = 2.0 * diff / self.out_dim as f32;
        }
        loss /= self.out_dim as f32;
        let mut d_weights = vec![0.0; self.out_dim * self.in_dim];
        let d_bias = grad_output.clone();
        for i in 0..self.out_dim {
            for j in 0..self.in_dim {
                d_weights[i * self.in_dim + j] = grad_output[i] * input[j];
            }
        }
        (LinearLayerGradients { d_weights, d_bias }, loss)
    }
    /// Parameter update (on CPU)
    pub fn update_parameters(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gradients: &LinearLayerGradients,
        lr: f32,
    ) {
        let mut weights = wgpu_utils::read_buffer_f32(
            device,
            queue,
            &self.weights_buf,
            self.out_dim * self.in_dim,
        );
        let mut bias = wgpu_utils::read_buffer_f32(device, queue, &self.bias_buf, self.out_dim);
        update_weights(&mut weights, &gradients.d_weights, lr);
        update_bias(&mut bias, &gradients.d_bias, lr);
        self.set_weights(queue, &weights);
        self.set_bias(queue, &bias);
    }
    /// Forward pass for a batch of inputs: each input is projected to a vector of size out_dim
    pub fn forward_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let batch = inputs.len();
        let mut outputs = Vec::with_capacity(batch);
        let weights = wgpu_utils::read_buffer_f32(
            device,
            queue,
            &self.weights_buf,
            self.out_dim * self.in_dim,
        );
        let bias = wgpu_utils::read_buffer_f32(device, queue, &self.bias_buf, self.out_dim);
        for input in inputs {
            let mut out = vec![0.0; self.out_dim];
            for i in 0..self.out_dim {
                let mut sum = 0.0;
                for j in 0..self.in_dim {
                    sum += weights[i * self.in_dim + j] * input[j];
                }
                out[i] = sum + bias[i];
            }
            outputs.push(out);
        }
        outputs
    }
    /// Train for a batch: inputs is &[f32], targets is &[Vec<f32>], lr is learning rate
    pub fn train_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &[f32],
        targets: &[Vec<f32>],
        lr: f32,
    ) -> f32 {
        let batch = inputs.len();
        assert_eq!(batch, targets.len());
        let mut weights = wgpu_utils::read_buffer_f32(
            device,
            queue,
            &self.weights_buf,
            self.out_dim * self.in_dim,
        );
        let mut bias = wgpu_utils::read_buffer_f32(device, queue, &self.bias_buf, self.out_dim);
        let mut total_loss = 0.0;
        for (idx, &input) in inputs.iter().enumerate() {
            let target = &targets[idx];
            let mut output = vec![0.0; self.out_dim];
            for i in 0..self.out_dim {
                output[i] = weights[i * self.in_dim] * input + bias[i];
            }
            let mut loss = 0.0;
            let mut grad_output = vec![0.0; self.out_dim];
            for i in 0..self.out_dim {
                let diff = output[i] - target[i];
                loss += diff * diff;
                grad_output[i] = 2.0 * diff / self.out_dim as f32;
            }
            for i in 0..self.out_dim {
                weights[i * self.in_dim] -= lr * grad_output[i] * input;
                bias[i] -= lr * grad_output[i];
            }
            total_loss += loss / self.out_dim as f32;
        }
        self.set_weights(queue, &weights);
        self.set_bias(queue, &bias);
        total_loss / batch as f32
    }
    /// Forward pass for a single input vector: projects [in_dim] -> [out_dim]
    pub fn forward_vec(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[f32],
    ) -> Vec<f32> {
        assert_eq!(input.len(), self.in_dim);
        let weights = wgpu_utils::read_buffer_f32(
            device,
            queue,
            &self.weights_buf,
            self.out_dim * self.in_dim,
        );
        let bias = wgpu_utils::read_buffer_f32(device, queue, &self.bias_buf, self.out_dim);
        let mut output = vec![0.0; self.out_dim];
        for i in 0..self.out_dim {
            let mut sum = 0.0;
            for j in 0..self.in_dim {
                sum += weights[i * self.in_dim + j] * input[j];
            }
            output[i] = sum + bias[i];
        }
        output
    }
    pub fn backward_gpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[f32],
        d_output: &[f32],
        weights: &[f32],
    ) -> (LinearLayerGradients, Vec<f32>) {
        // Write data to persistent buffers
        let cast_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(input.as_ptr() as *const u8, std::mem::size_of_val(input))
        };
        queue.write_buffer(&self.bw_input_buf, 0, cast_slice);
        let cast_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                d_output.as_ptr() as *const u8,
                std::mem::size_of_val(d_output),
            )
        };
        queue.write_buffer(&self.bw_d_output_buf, 0, cast_slice);
        let cast_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                weights.as_ptr() as *const u8,
                std::mem::size_of_val(weights),
            )
        };
        queue.write_buffer(&self.bw_weights_buf, 0, cast_slice);
        // Dispatch
        let in_dim = self.in_dim;
        let out_dim = self.out_dim;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Linear Backward Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Linear Backward Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bw_pipeline);
            cpass.set_bind_group(0, &self.bw_bind_group, &[]);
            cpass.dispatch_workgroups(((out_dim.max(in_dim) as f32) / 64.0).ceil() as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        // Read back gradients
        let d_weights =
            wgpu_utils::read_buffer_f32(device, queue, &self.bw_d_w_buf, out_dim * in_dim);
        let d_bias = wgpu_utils::read_buffer_f32(device, queue, &self.bw_d_b_buf, out_dim);
        let d_input = wgpu_utils::read_buffer_f32(device, queue, &self.bw_d_input_buf, in_dim);
        (LinearLayerGradients { d_weights, d_bias }, d_input)
    }
}
