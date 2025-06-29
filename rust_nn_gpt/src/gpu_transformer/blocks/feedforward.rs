use crate::gpu_transformer::gpu_utils::{
    pack_slices, unpack_slices, update_bias, update_weights, write_f32_slice,
};
use crate::gpu_transformer::wgpu_utils;

pub struct FeedForward {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub out_dim: usize,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub input_buf: wgpu::Buffer,
    pub output_buf: wgpu::Buffer,
    pub w1_buf: wgpu::Buffer,
    pub b1_buf: wgpu::Buffer,
    pub w2_buf: wgpu::Buffer,
    pub b2_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    // Packed backward buffers
    pub bw_inputs_buf: wgpu::Buffer, // input | pre_hidden | hidden | d_output
    pub bw_weights_buf: wgpu::Buffer, // w2 | w1
    pub bw_grads_buf: wgpu::Buffer,  // d_w1 | d_b1 | d_w2 | d_b2 | d_input
    pub bw_bind_group_layout: wgpu::BindGroupLayout,
    pub bw_pipeline: wgpu::ComputePipeline,
    pub bw_bind_group: wgpu::BindGroup,
}

pub struct FeedForwardGradients {
    pub d_w1: Vec<f32>,
    pub d_b1: Vec<f32>,
    pub d_w2: Vec<f32>,
    pub d_b2: Vec<f32>,
}

/// FeedForward layer for the GPU backend
impl FeedForward {
    /// Create a new FeedForward layer (2-layer MLP with ReLU)
    pub async fn new(
        device: &wgpu::Device,
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let shader_src = include_str!("../shaders/feedforward.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FeedForward Shader"),
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
            }, // w1_buf
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // b1_buf
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // w2_buf
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // b2_buf
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            }, // output_buf
        ]);
        let bind_group_layout =
            wgpu_utils::create_bind_group_layout(device, "FeedForward BGL", &bgl_entries);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FeedForward Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FeedForward Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });
        // Define usage for buffer specs
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        // Deduplicated forward buffer and bind group creation
        let forward_specs = [
            wgpu_utils::BufferSpec {
                label: "FeedForward Input",
                size: in_dim * std::mem::size_of::<f32>(),
                usage: Some(usage),
            },
            wgpu_utils::BufferSpec {
                label: "FeedForward W1",
                size: hidden_dim * in_dim * std::mem::size_of::<f32>(),
                usage: Some(usage),
            },
            wgpu_utils::BufferSpec {
                label: "FeedForward B1",
                size: hidden_dim * std::mem::size_of::<f32>(),
                usage: Some(usage),
            },
            wgpu_utils::BufferSpec {
                label: "FeedForward W2",
                size: out_dim * hidden_dim * std::mem::size_of::<f32>(),
                usage: Some(usage),
            },
            wgpu_utils::BufferSpec {
                label: "FeedForward B2",
                size: out_dim * std::mem::size_of::<f32>(),
                usage: Some(usage),
            },
            wgpu_utils::BufferSpec {
                label: "FeedForward Output",
                size: out_dim * std::mem::size_of::<f32>(),
                usage: Some(usage),
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
            "FeedForward BindGroup",
            "FeedForward BGL",
        );
        let bind_group_layout = forward.bind_group_layout;
        let bind_group = forward.bind_group;
        let mut buffers = forward.buffers.into_iter();
        let input_buf = buffers.next().ok_or("input_buf missing")?;
        let w1_buf = buffers.next().ok_or("w1_buf missing")?;
        let b1_buf = buffers.next().ok_or("b1_buf missing")?;
        let w2_buf = buffers.next().ok_or("w2_buf missing")?;
        let b2_buf = buffers.next().ok_or("b2_buf missing")?;
        let output_buf = buffers.next().ok_or("output_buf missing")?;
        // Deduplicated backward buffer and bind group creation
        let bw_inputs_size =
            (in_dim + hidden_dim + hidden_dim + out_dim) * std::mem::size_of::<f32>();
        let bw_weights_size =
            (out_dim * hidden_dim + hidden_dim * in_dim) * std::mem::size_of::<f32>();
        let bw_grads_size =
            (hidden_dim * in_dim + hidden_dim + out_dim * hidden_dim + out_dim + in_dim)
                * std::mem::size_of::<f32>();
        // --- Add dims uniform buffer for backward pass ---
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct FFWDims {
            in_dim: u32,
            hidden_dim: u32,
            out_dim: u32,
        }
        // ---
        let bw_specs = [
            wgpu_utils::BufferSpec {
                label: "FFBW Packed Inputs",
                size: bw_inputs_size,
                usage: Some(usage),
            },
            wgpu_utils::BufferSpec {
                label: "FFBW Packed Weights",
                size: bw_weights_size,
                usage: Some(usage),
            },
            wgpu_utils::BufferSpec {
                label: "FFBW Packed Grads",
                size: bw_grads_size,
                usage: Some(usage),
            },
            wgpu_utils::BufferSpec {
                label: "FFBW Dims Uniform",
                size: std::mem::size_of::<FFWDims>(),
                usage: Some(wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            },
        ];
        let bw_descs = [
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // packed inputs
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // packed weights
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            }, // packed grads
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: true,
            }, // dims uniform
        ];
        let bw = wgpu_utils::create_layer_gpu_resources(
            device,
            &bw_specs,
            &bw_descs,
            "FFBW BindGroup",
            "FFBW BGL",
        );
        let mut bw_buffers = bw.buffers.into_iter();
        let bw_inputs_buf = bw_buffers.next().ok_or("bw_inputs_buf missing")?;
        let bw_weights_buf = bw_buffers.next().ok_or("bw_weights_buf missing")?;
        let bw_grads_buf = bw_buffers.next().ok_or("bw_grads_buf missing")?;
        let _bw_dims_buf = bw_buffers.next().ok_or("bw_dims_buf missing")?;
        let bw_bind_group_layout = bw.bind_group_layout;
        let bw_bind_group = bw.bind_group;
        // Create backward pipeline as in the original code
        let bw_shader_src = include_str!("../shaders/feedforward_backward.wgsl");
        let bw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FeedForward Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(bw_shader_src.into()),
        });
        let bw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FeedForward Backward Pipeline Layout"),
            bind_group_layouts: &[&bw_bind_group_layout],
            push_constant_ranges: &[],
        });
        let bw_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FeedForward Backward Pipeline"),
            layout: Some(&bw_pipeline_layout),
            module: &bw_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });
        Ok(Self {
            in_dim,
            hidden_dim,
            out_dim,
            pipeline,
            bind_group_layout,
            input_buf,
            output_buf,
            w1_buf,
            b1_buf,
            w2_buf,
            b2_buf,
            bind_group,
            bw_inputs_buf,
            bw_weights_buf,
            bw_grads_buf,
            bw_bind_group_layout,
            bw_pipeline,
            bw_bind_group,
        })
    }
    /// GELU activation for a single value
    pub fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3))).tanh())
    }
    /// Set input buffer
    pub fn set_input(&self, queue: &wgpu::Queue, input: &[f32]) {
        assert_eq!(input.len(), self.in_dim);
        write_f32_slice(queue, &self.input_buf, input);
    }
    /// Set w1 buffer
    pub fn set_w1(&self, queue: &wgpu::Queue, w1: &[f32]) {
        assert_eq!(w1.len(), self.hidden_dim * self.in_dim);
        write_f32_slice(queue, &self.w1_buf, w1);
    }
    /// Set b1 buffer
    pub fn set_b1(&self, queue: &wgpu::Queue, b1: &[f32]) {
        assert_eq!(b1.len(), self.hidden_dim);
        write_f32_slice(queue, &self.b1_buf, b1);
    }
    /// Set w2 buffer
    pub fn set_w2(&self, queue: &wgpu::Queue, w2: &[f32]) {
        assert_eq!(w2.len(), self.out_dim * self.hidden_dim);
        write_f32_slice(queue, &self.w2_buf, w2);
    }
    /// Set b2 buffer
    pub fn set_b2(&self, queue: &wgpu::Queue, b2: &[f32]) {
        assert_eq!(b2.len(), self.out_dim);
        write_f32_slice(queue, &self.b2_buf, b2);
    }
    /// Forward pass (GPU, for inference)
    pub fn forward(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        // Encode and submit
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FeedForward Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FeedForward Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(((self.out_dim as f32) / 64.0).ceil() as u32, 1, 1);
        }
        // Create staging buffer for readback
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FeedForward Staging Buffer"),
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
    pub fn compute_gradients(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[f32],
        target: &[f32],
    ) -> (FeedForwardGradients, f32) {
        let w1 =
            wgpu_utils::read_buffer_f32(device, queue, &self.w1_buf, self.hidden_dim * self.in_dim);
        let b1 = wgpu_utils::read_buffer_f32(device, queue, &self.b1_buf, self.hidden_dim);
        let w2 = wgpu_utils::read_buffer_f32(
            device,
            queue,
            &self.w2_buf,
            self.out_dim * self.hidden_dim,
        );
        let b2 = wgpu_utils::read_buffer_f32(device, queue, &self.b2_buf, self.out_dim);
        // Forward pass
        let mut hidden = vec![0.0; self.hidden_dim];
        let mut pre_hidden = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut sum = 0.0;
            for j in 0..self.in_dim {
                sum += w1[i * self.in_dim + j] * input[j];
            }
            pre_hidden[i] = sum + b1[i];
            hidden[i] = Self::gelu(pre_hidden[i]);
        }
        let mut output = vec![0.0; self.out_dim];
        for i in 0..self.out_dim {
            let mut sum = 0.0;
            for j in 0..self.hidden_dim {
                sum += w2[i * self.hidden_dim + j] * hidden[j];
            }
            output[i] = sum + b2[i];
        }
        // Compute loss and grad_output
        let mut grad_output = vec![0.0; self.out_dim];
        let mut loss = 0.0;
        for i in 0..self.out_dim {
            let diff = output[i] - target[i];
            loss += diff * diff;
            grad_output[i] = 2.0 * diff / self.out_dim as f32;
        }
        loss /= self.out_dim as f32;
        // Backprop to hidden (GELU derivative)
        let mut grad_hidden = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut sum = 0.0;
            for i in 0..self.out_dim {
                sum += grad_output[i] * w2[i * self.hidden_dim + j];
            }
            // GELU derivative
            let x = pre_hidden[j];
            let tanh_arg = std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3));
            let tanh = tanh_arg.tanh();
            let left = 0.5 * tanh + 0.5;
            let right = 0.5
                * x
                * (1.0 - tanh * tanh)
                * std::f32::consts::FRAC_2_SQRT_PI
                * (1.0 + 3.0 * 0.044715 * x.powi(2));
            let gelu_grad = left + right;
            grad_hidden[j] = sum * gelu_grad;
        }
        // Gradients for w2, b2
        let mut d_w2 = vec![0.0; self.out_dim * self.hidden_dim];
        let d_b2 = grad_output.clone();
        for i in 0..self.out_dim {
            for j in 0..self.hidden_dim {
                d_w2[i * self.hidden_dim + j] = grad_output[i] * hidden[j];
            }
        }
        // Gradients for w1, b1
        let mut d_w1 = vec![0.0; self.hidden_dim * self.in_dim];
        let d_b1 = grad_hidden.clone();
        for i in 0..self.hidden_dim {
            for j in 0..self.in_dim {
                d_w1[i * self.in_dim + j] = grad_hidden[i] * input[j];
            }
        }
        (
            FeedForwardGradients {
                d_w1: d_w1.clone(),
                d_b1,
                d_w2: d_w2.clone(),
                d_b2,
            },
            loss,
        )
    }
    pub fn update_parameters(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gradients: &FeedForwardGradients,
        lr: f32,
    ) {
        let mut w1 =
            wgpu_utils::read_buffer_f32(device, queue, &self.w1_buf, self.hidden_dim * self.in_dim);
        let mut b1 = wgpu_utils::read_buffer_f32(device, queue, &self.b1_buf, self.hidden_dim);
        let mut w2 = wgpu_utils::read_buffer_f32(
            device,
            queue,
            &self.w2_buf,
            self.out_dim * self.hidden_dim,
        );
        let mut b2 = wgpu_utils::read_buffer_f32(device, queue, &self.b2_buf, self.out_dim);
        update_weights(&mut w1, &gradients.d_w1, lr);
        update_bias(&mut b1, &gradients.d_b1, lr);
        update_weights(&mut w2, &gradients.d_w2, lr);
        update_bias(&mut b2, &gradients.d_b2, lr);
        self.set_w1(queue, &w1);
        self.set_b1(queue, &b1);
        self.set_w2(queue, &w2);
        self.set_b2(queue, &b2);
    }
    #[allow(clippy::too_many_arguments)]
    pub fn backward_gpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[f32],
        pre_hidden: &[f32],
        hidden: &[f32],
        grad_output: &[f32],
        w2: &[f32],
        w1: &[f32],
    ) -> FeedForwardGradients {
        let in_dim = self.in_dim;
        let hidden_dim = self.hidden_dim;
        let out_dim = self.out_dim;
        // Pack inputs: input | pre_hidden | hidden | d_output
        let packed_inputs = pack_slices(&[input, pre_hidden, hidden, grad_output]);
        write_f32_slice(queue, &self.bw_inputs_buf, &packed_inputs);
        // Pack weights: w2 | w1
        let mut packed_weights = Vec::with_capacity(out_dim * hidden_dim + hidden_dim * in_dim);
        packed_weights.extend_from_slice(w2);
        packed_weights.extend_from_slice(w1);
        write_f32_slice(queue, &self.bw_weights_buf, &packed_weights);
        // Zero grads buffer before running (optional, for safety)
        let grads_len = hidden_dim * in_dim + hidden_dim + out_dim * hidden_dim + out_dim + in_dim;
        let zero_grads = vec![0.0f32; grads_len];
        write_f32_slice(queue, &self.bw_grads_buf, &zero_grads);
        // Dispatch
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FFBW Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFBW Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bw_pipeline);
            cpass.set_bind_group(0, &self.bw_bind_group, &[]);
            let workgroups = ((in_dim.max(hidden_dim).max(out_dim) as f32) / 64.0).ceil() as u32;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::MaintainBase::Wait);
        // Unpack gradients from packed grads buffer
        let grads = wgpu_utils::read_buffer_f32(device, queue, &self.bw_grads_buf, grads_len);
        let unpacked = unpack_slices(
            &grads,
            &[
                in_dim,
                hidden_dim,
                hidden_dim,
                in_dim * hidden_dim,
                hidden_dim,
                hidden_dim * out_dim,
                out_dim,
            ],
        );
        FeedForwardGradients {
            d_w1: unpacked[3].clone(),
            d_b1: unpacked[4].clone(),
            d_w2: unpacked[5].clone(),
            d_b2: unpacked[6].clone(),
        }
    }
}
// TODO: Update WGSL shader to implement a real 2-layer MLP with ReLU.
// TODO: Update WGSL shader to implement a real 2-layer MLP with ReLU.
