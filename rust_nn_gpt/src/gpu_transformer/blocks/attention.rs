use super::linear_layer::LinearLayer;
use crate::gpu_transformer::gpu_utils::{
    pack_slices, unpack_slices, update_bias, update_weights, write_f32_slice,
};
use crate::gpu_transformer::wgpu_utils;
use rand::Rng;
use rand_distr;
use wgpu::util::DeviceExt;

pub struct MultiHeadAttention {
    pub in_dim: usize,
    pub out_dim: usize,
    pub num_heads: usize,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub input_buf: wgpu::Buffer,
    pub output_buf: wgpu::Buffer,
    pub q_w_buf: wgpu::Buffer,
    pub k_w_buf: wgpu::Buffer,
    pub v_w_buf: wgpu::Buffer,
    pub o_w_buf: wgpu::Buffer,
    pub bias_buf: wgpu::Buffer, // packed: [q_b | k_b | v_b | o_b]
    pub bind_group: wgpu::BindGroup,
    // Persistent backward packed buffers
    pub bw_input_buf: wgpu::Buffer,  // packed input for backward
    pub bw_output_buf: wgpu::Buffer, // packed output for backward
    pub bw_dim_buf: wgpu::Buffer,    // uniform buffer for dim
    pub bw_bind_group_layout: wgpu::BindGroupLayout,
    pub bw_pipeline: wgpu::ComputePipeline,
    pub bw_bind_group: wgpu::BindGroup,
}

pub struct AttentionGradients {
    pub d_q_w: Vec<f32>,
    pub d_k_w: Vec<f32>,
    pub d_v_w: Vec<f32>,
    pub d_o_w: Vec<f32>,
    pub d_q_b: Vec<f32>,
    pub d_k_b: Vec<f32>,
    pub d_v_b: Vec<f32>,
    pub d_o_b: Vec<f32>,
}

pub struct AttentionBackwardGradients {
    pub d_wo: Vec<f32>,
    pub d_q: Vec<f32>,
    pub d_k: Vec<f32>,
    pub d_v: Vec<f32>,
    pub d_input: Vec<f32>,
    pub d_o_b: Vec<f32>, // true output bias gradient
}

/// Multi-head self-attention layer for the GPU backend
impl MultiHeadAttention {
    /// Create a new single-head attention layer
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        in_dim: usize,
        out_dim: usize,
        num_heads: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let shader_src = include_str!("../shaders/attention.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MHA Shader"),
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
            }, // q_w_buf
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // k_w_buf
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // v_w_buf
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // o_w_buf
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // bias_buf
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            }, // output_buf
        ]);
        let bind_group_layout =
            wgpu_utils::create_bind_group_layout(device, "MHA BGL", &bgl_entries);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MHA Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MHA Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });
        // Deduplicated forward buffer and bind group creation
        let usage = Some(
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let forward_specs = [
            wgpu_utils::BufferSpec {
                label: "MHA Input",
                size: in_dim * std::mem::size_of::<f32>(),
                usage,
            },
            wgpu_utils::BufferSpec {
                label: "MHA QW",
                size: in_dim * in_dim * std::mem::size_of::<f32>(),
                usage,
            },
            wgpu_utils::BufferSpec {
                label: "MHA KW",
                size: in_dim * in_dim * std::mem::size_of::<f32>(),
                usage,
            },
            wgpu_utils::BufferSpec {
                label: "MHA VW",
                size: in_dim * in_dim * std::mem::size_of::<f32>(),
                usage,
            },
            wgpu_utils::BufferSpec {
                label: "MHA OW",
                size: in_dim * in_dim * std::mem::size_of::<f32>(),
                usage,
            },
            wgpu_utils::BufferSpec {
                label: "MHA Biases",
                size: (3 * in_dim + out_dim) * std::mem::size_of::<f32>(),
                usage,
            },
            wgpu_utils::BufferSpec {
                label: "MHA Output",
                size: in_dim * std::mem::size_of::<f32>(),
                usage,
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
            "MHA BindGroup",
            "MHA BGL",
        );
        let bind_group_layout = forward.bind_group_layout;
        let bind_group = forward.bind_group;
        let mut buffers = forward.buffers.into_iter();
        let input_buf = buffers.next().ok_or("input_buf missing")?;
        let q_w_buf = buffers.next().ok_or("q_w_buf missing")?;
        let k_w_buf = buffers.next().ok_or("k_w_buf missing")?;
        let v_w_buf = buffers.next().ok_or("v_w_buf missing")?;
        let o_w_buf = buffers.next().ok_or("o_w_buf missing")?;
        let bias_buf = buffers.next().ok_or("bias_buf missing")?;
        let output_buf = buffers.next().ok_or("output_buf missing")?;
        // Initialize weights and biases
        {
            let mut rng = rand::thread_rng();
            let normal = rand_distr::Normal::new(0.0, 0.01)?;
            let q_w: Vec<f32> = (0..in_dim * in_dim)
                .map(|_| rng.sample::<f32, _>(normal))
                .collect();
            let k_w: Vec<f32> = (0..in_dim * in_dim)
                .map(|_| rng.sample::<f32, _>(normal))
                .collect();
            let v_w: Vec<f32> = (0..in_dim * in_dim)
                .map(|_| rng.sample::<f32, _>(normal))
                .collect();
            let o_w: Vec<f32> = (0..in_dim * in_dim)
                .map(|_| rng.sample::<f32, _>(normal))
                .collect();
            let q_b = vec![0.0; in_dim];
            let k_b = vec![0.0; in_dim];
            let v_b = vec![0.0; in_dim];
            let o_b = vec![0.0; out_dim];
            assert_eq!(q_w.len(), in_dim * in_dim, "q_w size mismatch");
            assert_eq!(k_w.len(), in_dim * in_dim, "k_w size mismatch");
            assert_eq!(v_w.len(), in_dim * in_dim, "v_w size mismatch");
            assert_eq!(o_w.len(), in_dim * in_dim, "o_w size mismatch");
            assert_eq!(q_b.len(), in_dim, "q_b size mismatch");
            assert_eq!(k_b.len(), in_dim, "k_b size mismatch");
            assert_eq!(v_b.len(), in_dim, "v_b size mismatch");
            assert_eq!(o_b.len(), out_dim, "o_b size mismatch");
            queue.write_buffer(&q_w_buf, 0, bytemuck::cast_slice(&q_w));
            queue.write_buffer(&k_w_buf, 0, bytemuck::cast_slice(&k_w));
            queue.write_buffer(&v_w_buf, 0, bytemuck::cast_slice(&v_w));
            queue.write_buffer(&o_w_buf, 0, bytemuck::cast_slice(&o_w));
            let mut packed_biases = Vec::with_capacity(3 * in_dim + out_dim);
            packed_biases.extend_from_slice(&q_b);
            packed_biases.extend_from_slice(&k_b);
            packed_biases.extend_from_slice(&v_b);
            packed_biases.extend_from_slice(&o_b);
            assert_eq!(
                packed_biases.len(),
                3 * in_dim + out_dim,
                "packed_biases size mismatch"
            );
            let manual_cast_slice: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    packed_biases.as_ptr() as *const u8,
                    packed_biases.len() * std::mem::size_of::<f32>(),
                )
            };
            queue.write_buffer(&bias_buf, 0, manual_cast_slice);
        }
        // Deduplicated backward buffer and bind group creation
        let bw_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let bw_input_size = (3 * in_dim + in_dim + in_dim * in_dim) * std::mem::size_of::<f32>();
        let bw_output_size =
            (in_dim * in_dim + 3 * in_dim + in_dim + in_dim) * std::mem::size_of::<f32>();
        // Create the uniform buffer first
        let bw_dim_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MHA Backward Dim Uniform"),
            contents: bytemuck::cast_slice(&[in_dim as u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Pass all three buffers in buffer_specs
        let bufs = vec![
            wgpu_utils::create_storage_buffer(
                device,
                "MHA Backward InputBuf",
                bw_input_size,
                Some(bw_usage),
            ),
            wgpu_utils::create_storage_buffer(
                device,
                "MHA Backward OutputBuf",
                bw_output_size,
                Some(bw_usage),
            ),
            bw_dim_buf.clone(),
        ];
        // Define backward buffer descriptions for bind group layout
        let bw_descs = [
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // input_buf
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            }, // output_buf
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: true,
            }, // dim_uniform
        ];
        let bgl_entries = wgpu_utils::make_bgl_entries(&bw_descs);
        let bw_bind_group_layout =
            wgpu_utils::create_bind_group_layout(device, "MHA Backward BGL", &bgl_entries);
        let bw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MHA Backward BindGroup"),
            layout: &bw_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bufs[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs[2].as_entire_binding(),
                },
            ],
        });
        let mut bw_buffers_iter = bufs.into_iter();
        let bw_input_buf = bw_buffers_iter.next().ok_or("bw_input_buf missing")?;
        let bw_output_buf = bw_buffers_iter.next().ok_or("bw_output_buf missing")?;
        // bw_dim_buf is already created
        // Create backward pipeline as in the original code
        let bw_shader_src = include_str!("../shaders/attention_backward.wgsl");
        let bw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MHA Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(bw_shader_src.into()),
        });
        let bw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MHA Backward Pipeline Layout"),
            bind_group_layouts: &[&bw_bind_group_layout],
            push_constant_ranges: &[],
        });
        let bw_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MHA Backward Pipeline"),
            layout: Some(&bw_pipeline_layout),
            module: &bw_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });
        Ok(Self {
            in_dim,
            out_dim,
            num_heads,
            pipeline,
            bind_group_layout,
            input_buf,
            output_buf,
            q_w_buf,
            k_w_buf,
            v_w_buf,
            o_w_buf,
            bias_buf,
            bind_group,
            bw_input_buf,
            bw_output_buf,
            bw_dim_buf,
            bw_bind_group_layout,
            bw_pipeline,
            bw_bind_group,
        })
    }
    /// Forward pass (to be implemented)
    pub fn forward(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        // Encode and submit
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MHA Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MHA Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(((self.out_dim as f32) / 64.0).ceil() as u32, 1, 1);
        }
        // Create staging buffer for readback
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MHA Staging Buffer"),
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
    /// Read a buffer as a Vec<f32>
    pub fn read_buffer_f32(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        len: usize,
    ) -> Vec<f32> {
        wgpu_utils::read_buffer_f32(device, queue, buffer, len)
    }
    /// Set input buffer
    pub fn set_input(&self, queue: &wgpu::Queue, input: &[f32]) {
        assert_eq!(input.len(), self.in_dim);
        write_f32_slice(queue, &self.input_buf, input);
    }
    /// Set q_w buffer
    pub fn set_q_w(&self, queue: &wgpu::Queue, q_w: &[f32]) {
        assert_eq!(q_w.len(), self.in_dim * self.in_dim);
        write_f32_slice(queue, &self.q_w_buf, q_w);
    }
    /// Set k_w buffer
    pub fn set_k_w(&self, queue: &wgpu::Queue, k_w: &[f32]) {
        assert_eq!(k_w.len(), self.in_dim * self.in_dim);
        write_f32_slice(queue, &self.k_w_buf, k_w);
    }
    /// Set v_w buffer
    pub fn set_v_w(&self, queue: &wgpu::Queue, v_w: &[f32]) {
        assert_eq!(v_w.len(), self.in_dim * self.in_dim);
        write_f32_slice(queue, &self.v_w_buf, v_w);
    }
    /// Set o_w buffer
    pub fn set_o_w(&self, queue: &wgpu::Queue, o_w: &[f32]) {
        assert_eq!(o_w.len(), self.out_dim * self.in_dim);
        write_f32_slice(queue, &self.o_w_buf, o_w);
    }
    /// Set biases buffer (packed)
    pub fn set_biases(
        &self,
        queue: &wgpu::Queue,
        q_b: &[f32],
        k_b: &[f32],
        v_b: &[f32],
        o_b: &[f32],
    ) {
        assert_eq!(q_b.len(), self.in_dim);
        assert_eq!(k_b.len(), self.in_dim);
        assert_eq!(v_b.len(), self.in_dim);
        assert_eq!(o_b.len(), self.out_dim);
        let mut packed = Vec::with_capacity(3 * self.in_dim + self.out_dim);
        packed.extend_from_slice(q_b);
        packed.extend_from_slice(k_b);
        packed.extend_from_slice(v_b);
        packed.extend_from_slice(o_b);
        write_f32_slice(queue, &self.bias_buf, &packed);
    }
    pub fn compute_gradients(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[f32],
        target: &[f32],
    ) -> (AttentionGradients, f32) {
        let in_dim = self.in_dim;
        let out_dim = self.out_dim;
        let num_heads = self.num_heads.max(1);
        let head_dim = in_dim / num_heads;
        let o_w = Self::read_buffer_f32(device, queue, &self.o_w_buf, out_dim * in_dim);
        let biases = Self::read_buffer_f32(device, queue, &self.bias_buf, 3 * in_dim + out_dim);
        // Multi-head Q, K, V using LinearLayer
        let mut q_heads = vec![vec![0.0; head_dim]; num_heads];
        let mut k_heads = vec![vec![0.0; head_dim]; num_heads];
        let mut v_heads = vec![vec![0.0; head_dim]; num_heads];
        for h in 0..num_heads {
            let q_layer = LinearLayer {
                in_dim,
                out_dim: head_dim,
                pipeline: self.pipeline.clone(),
                bind_group_layout: self.bind_group_layout.clone(),
                input_buf: self.input_buf.clone(),
                output_buf: self.output_buf.clone(),
                weights_buf: self.q_w_buf.clone(),
                bias_buf: self.bias_buf.clone(),
                in_dim_buf: self.bw_dim_buf.clone(),
                out_dim_buf: self.bw_dim_buf.clone(),
                bind_group: self.bind_group.clone(),
                bw_input_buf: self.bw_input_buf.clone(),
                bw_d_output_buf: self.bw_output_buf.clone(),
                bw_weights_buf: self.q_w_buf.clone(),
                bw_d_w_buf: self.q_w_buf.clone(),
                bw_d_b_buf: self.q_w_buf.clone(),
                bw_d_input_buf: self.q_w_buf.clone(),
                bw_bind_group_layout: self.bw_bind_group_layout.clone(),
                bw_pipeline: self.bw_pipeline.clone(),
                bw_bind_group: self.bw_bind_group.clone(),
            };
            let k_layer = LinearLayer {
                in_dim,
                out_dim: head_dim,
                pipeline: self.pipeline.clone(),
                bind_group_layout: self.bind_group_layout.clone(),
                input_buf: self.input_buf.clone(),
                output_buf: self.output_buf.clone(),
                weights_buf: self.k_w_buf.clone(),
                bias_buf: self.bias_buf.clone(),
                in_dim_buf: self.bw_dim_buf.clone(),
                out_dim_buf: self.bw_dim_buf.clone(),
                bind_group: self.bind_group.clone(),
                bw_input_buf: self.bw_input_buf.clone(),
                bw_d_output_buf: self.bw_output_buf.clone(),
                bw_weights_buf: self.k_w_buf.clone(),
                bw_d_w_buf: self.k_w_buf.clone(),
                bw_d_b_buf: self.k_w_buf.clone(),
                bw_d_input_buf: self.k_w_buf.clone(),
                bw_bind_group_layout: self.bw_bind_group_layout.clone(),
                bw_pipeline: self.bw_pipeline.clone(),
                bw_bind_group: self.bw_bind_group.clone(),
            };
            let v_layer = LinearLayer {
                in_dim,
                out_dim: head_dim,
                pipeline: self.pipeline.clone(),
                bind_group_layout: self.bind_group_layout.clone(),
                input_buf: self.input_buf.clone(),
                output_buf: self.output_buf.clone(),
                weights_buf: self.v_w_buf.clone(),
                bias_buf: self.bias_buf.clone(),
                in_dim_buf: self.bw_dim_buf.clone(),
                out_dim_buf: self.bw_dim_buf.clone(),
                bind_group: self.bind_group.clone(),
                bw_input_buf: self.bw_input_buf.clone(),
                bw_d_output_buf: self.bw_output_buf.clone(),
                bw_weights_buf: self.v_w_buf.clone(),
                bw_d_w_buf: self.v_w_buf.clone(),
                bw_d_b_buf: self.v_w_buf.clone(),
                bw_d_input_buf: self.v_w_buf.clone(),
                bw_bind_group_layout: self.bw_bind_group_layout.clone(),
                bw_pipeline: self.bw_pipeline.clone(),
                bw_bind_group: self.bw_bind_group.clone(),
            };
            q_heads[h] = q_layer.forward_vec(device, queue, input);
            k_heads[h] = k_layer.forward_vec(device, queue, input);
            v_heads[h] = v_layer.forward_vec(device, queue, input);
        }
        // Attention for each head
        let mut attn_out_heads = vec![vec![0.0; head_dim]; num_heads];
        for h in 0..num_heads {
            let scale = (head_dim as f32).sqrt();
            let mut _attn_scores = 0.0;
            for j in 0..head_dim {
                _attn_scores += q_heads[h][j] * k_heads[h][j];
            }
            _attn_scores /= scale;
            let attn = 1.0; // For a single token, softmax([score]) = 1.0
            for j in 0..head_dim {
                attn_out_heads[h][j] = v_heads[h][j] * attn;
            }
        }
        // Concatenate heads
        let mut attn_out = vec![0.0; in_dim];
        for h in 0..num_heads {
            for j in 0..head_dim {
                attn_out[h * head_dim + j] = attn_out_heads[h][j];
            }
        }
        // Output projection
        let mut output = vec![0.0; out_dim];
        for i in 0..out_dim {
            let mut sum = 0.0;
            for j in 0..in_dim {
                sum += attn_out[j] * o_w[i * in_dim + j];
            }
            output[i] = sum + biases[i];
        }
        // Compute loss and grad_output
        let mut grad_output = vec![0.0; out_dim];
        let mut loss = 0.0;
        for i in 0..out_dim {
            let diff = output[i] - target[i];
            loss += diff * diff;
            grad_output[i] = 2.0 * diff / out_dim as f32;
        }
        loss /= out_dim as f32;
        // Gradients for o_w, o_b
        let mut d_o_w = vec![0.0; out_dim * in_dim];
        let d_o_b = grad_output.clone();
        for i in 0..out_dim {
            for j in 0..in_dim {
                d_o_w[i * in_dim + j] = grad_output[i] * attn_out[j];
            }
        }
        // Gradients for attn_out (v)
        let mut grad_attn_out = vec![0.0; in_dim];
        for j in 0..in_dim {
            for i in 0..out_dim {
                grad_attn_out[j] += grad_output[i] * o_w[i * in_dim + j];
            }
        }
        // Gradients for v_w, v_b, q_w, k_w, q_b, k_b, v_b (per head)
        let mut d_v_w = vec![0.0; in_dim * in_dim];
        let mut d_q_w = vec![0.0; in_dim * in_dim];
        let mut d_k_w = vec![0.0; in_dim * in_dim];
        let mut d_v_b = vec![0.0; in_dim];
        let mut d_q_b = vec![0.0; in_dim];
        let mut d_k_b = vec![0.0; in_dim];
        for h in 0..num_heads {
            for j in 0..head_dim {
                let idx = h * head_dim + j;
                let grad_v = grad_attn_out[idx];
                for k2 in 0..in_dim {
                    d_v_w[idx * in_dim + k2] += grad_v * input[k2];
                }
                d_v_b[idx] += grad_v;
                let _scale = (head_dim as f32).sqrt();
                let grad_score = 0.0; // For single token, softmax grad is 0
                for k2 in 0..in_dim {
                    d_q_w[idx * in_dim + k2] += grad_score * input[k2];
                    d_k_w[idx * in_dim + k2] += grad_score * input[k2];
                }
                d_q_b[idx] += grad_score;
                d_k_b[idx] += grad_score;
            }
        }
        (
            AttentionGradients {
                d_q_w: d_q_w.clone(),
                d_k_w: d_k_w.clone(),
                d_v_w: d_v_w.clone(),
                d_o_w: d_o_w.clone(),
                d_q_b: d_q_b.clone(),
                d_k_b: d_k_b.clone(),
                d_v_b: d_v_b.clone(),
                d_o_b,
            },
            loss,
        )
    }
    pub fn update_parameters(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gradients: &AttentionGradients,
        lr: f32,
    ) {
        let mut o_w =
            Self::read_buffer_f32(device, queue, &self.o_w_buf, self.out_dim * self.in_dim);
        let mut biases = Self::read_buffer_f32(
            device,
            queue,
            &self.bias_buf,
            3 * self.in_dim + self.out_dim,
        );
        update_weights(&mut o_w, &gradients.d_o_w, lr);
        update_bias(biases.as_mut_slice(), &gradients.d_o_b, lr);
        update_weights(biases.as_mut_slice(), &gradients.d_v_w, lr);
        update_weights(biases.as_mut_slice(), &gradients.d_q_w, lr);
        update_weights(biases.as_mut_slice(), &gradients.d_k_w, lr);
        update_bias(biases.as_mut_slice(), &gradients.d_v_b, lr);
        update_bias(biases.as_mut_slice(), &gradients.d_q_b, lr);
        update_bias(biases.as_mut_slice(), &gradients.d_k_b, lr);
        self.set_o_w(queue, &o_w);
        self.set_biases(
            queue,
            biases.as_slice(),
            biases.as_slice(),
            biases.as_slice(),
            biases.as_slice(),
        );
        self.set_v_w(queue, biases.as_slice());
        self.set_q_w(queue, biases.as_slice());
        self.set_k_w(queue, biases.as_slice());
    }
    #[allow(clippy::too_many_arguments)]
    pub fn backward_gpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d_output: &[f32],
        w_o: &[f32],
    ) -> (AttentionGradients, Vec<f32>) {
        let dim = self.in_dim;
        // Pack Q, K, V, d_output, w_o into a single input buffer
        let input_packed = pack_slices(&[q, k, v, d_output, w_o]);
        write_f32_slice(queue, &self.bw_input_buf, &input_packed);
        queue.write_buffer(&self.bw_dim_buf, 0, bytemuck::cast_slice(&[dim as u32]));
        // Dispatch
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MHA Backward Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MHA Backward Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bw_pipeline);
            cpass.set_bind_group(0, &self.bw_bind_group, &[]);
            cpass.dispatch_workgroups(((dim as f32) / 64.0).ceil() as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        // Read back and unpack output buffer
        let output_len = dim * dim + 3 * dim + dim + dim + 3 * dim * dim;
        let output_data = Self::read_buffer_f32(device, queue, &self.bw_output_buf, output_len);
        let unpacked = unpack_slices(
            &output_data,
            &[
                dim,
                dim,
                dim,
                dim,
                dim * dim,
                dim * dim,
                dim,
                dim * dim,
                dim,
                dim * dim,
                dim,
                dim * dim,
                dim,
            ],
        );
        let d_weights_q = &unpacked[5];
        let d_bias_q = &unpacked[6];
        let d_weights_k = &unpacked[7];
        let d_bias_k = &unpacked[8];
        let d_weights_v = &unpacked[9];
        let d_bias_v = &unpacked[10];
        let d_weights_o = &unpacked[11];
        let d_bias_o = &unpacked[12];
        (
            AttentionGradients {
                d_q_w: d_weights_q.clone(),
                d_k_w: d_weights_k.clone(),
                d_v_w: d_weights_v.clone(),
                d_o_w: d_weights_o.clone(),
                d_q_b: d_bias_q.clone(),
                d_k_b: d_bias_k.clone(),
                d_v_b: d_bias_v.clone(),
                d_o_b: d_bias_o.clone(),
            },
            unpacked[0].clone(),
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn backward_gpu_generalized(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        attn_weights: &[f32],
        attn_out: &[f32],
        d_output: &[f32],
        w_o: &[f32],
        seq_len: usize,
        dim: usize,
    ) -> AttentionBackwardGradients {
        // 1. Pack Q, K, V into a single buffer
        let packed_qkv = pack_slices(&[q, k, v]);
        // 2. Allocate and bind all buffers
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let qkv_buf =
            wgpu_utils::create_storage_buffer_with_data(device, "QKV", &packed_qkv, Some(usage));
        let attn_weights_buf = wgpu_utils::create_storage_buffer_with_data(
            device,
            "AttnWeights",
            attn_weights,
            Some(usage),
        );
        let attn_out_buf =
            wgpu_utils::create_storage_buffer_with_data(device, "AttnOut", attn_out, Some(usage));
        let d_output_buf =
            wgpu_utils::create_storage_buffer_with_data(device, "DOutput", d_output, Some(usage));
        let w_o_buf = wgpu_utils::create_storage_buffer_with_data(device, "WO", w_o, Some(usage));
        // Output buffers (zero-initialized)
        let d_wo_buf = wgpu_utils::create_storage_buffer(
            device,
            "DWO",
            dim * dim * std::mem::size_of::<f32>(),
            Some(usage),
        );
        let d_qkv_buf = wgpu_utils::create_storage_buffer(
            device,
            "DQKV",
            seq_len * dim * 3 * std::mem::size_of::<f32>(),
            Some(usage),
        );
        let d_input_buf = wgpu_utils::create_storage_buffer(
            device,
            "DInput",
            seq_len * dim * std::mem::size_of::<f32>(),
            Some(usage),
        );
        let d_o_b_buf = wgpu_utils::create_storage_buffer(
            device,
            "DO_B",
            dim * std::mem::size_of::<f32>(),
            Some(usage),
        );
        // Uniform buffer for dim
        let dims_data = [seq_len as u32, dim as u32];
        let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("AttnDims"),
            contents: bytemuck::cast_slice(&dims_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // 3. Create bind group and layout
        let bgl_entries = wgpu_utils::make_bgl_entries(&[
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // qkv
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // attn_weights
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // attn_out
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // d_output
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: false,
            }, // w_o
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            }, // d_wo
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            }, // d_qkv
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            }, // d_input
            wgpu_utils::BufferDesc {
                read_only: true,
                is_uniform: true,
            }, // dims
            wgpu_utils::BufferDesc {
                read_only: false,
                is_uniform: false,
            }, // d_o_b
        ]);
        let bind_group_layout =
            wgpu_utils::create_bind_group_layout(device, "MHA Backward BGL", &bgl_entries);
        let bind_group = wgpu_utils::create_bind_group(
            device,
            "MHA Backward BindGroup",
            &bind_group_layout,
            &[
                (&qkv_buf, 0),
                (&attn_weights_buf, 1),
                (&attn_out_buf, 2),
                (&d_output_buf, 3),
                (&w_o_buf, 4),
                (&d_wo_buf, 5),
                (&d_qkv_buf, 6),
                (&d_input_buf, 7),
                (&dims_buf, 8),
                (&d_o_b_buf, 9),
            ],
        );
        // 4. Dispatch the compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MHA Backward Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MHA Backward Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bw_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                ((seq_len as f32) / 8.0).ceil() as u32,
                ((dim as f32) / 8.0).ceil() as u32,
                1,
            );
        }
        queue.submit(Some(encoder.finish()));
        // 5. Read back gradients
        let d_wo = Self::read_buffer_f32(device, queue, &d_wo_buf, dim * dim);
        let d_qkv = Self::read_buffer_f32(device, queue, &d_qkv_buf, seq_len * dim * 3);
        let d_input = Self::read_buffer_f32(device, queue, &d_input_buf, seq_len * dim);
        let d_o_b = Self::read_buffer_f32(device, queue, &d_o_b_buf, dim);
        // Unpack d_q, d_k, d_v from d_qkv
        let d_q = d_qkv[0..seq_len * dim].to_vec();
        let d_k = d_qkv[seq_len * dim..2 * seq_len * dim].to_vec();
        let d_v = d_qkv[2 * seq_len * dim..3 * seq_len * dim].to_vec();
        AttentionBackwardGradients {
            d_wo,
            d_q,
            d_k,
            d_v,
            d_input,
            d_o_b,
        }
    }
    /// Forward pass with intermediates (returns Q, K, V, attn_weights, output)
    #[allow(clippy::type_complexity)]
    pub fn forward_with_intermediates(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let dim = self.in_dim;
        // Read weights and biases
        let q_w = Self::read_buffer_f32(device, queue, &self.q_w_buf, dim * dim);
        let k_w = Self::read_buffer_f32(device, queue, &self.k_w_buf, dim * dim);
        let v_w = Self::read_buffer_f32(device, queue, &self.v_w_buf, dim * dim);
        let _o_w = Self::read_buffer_f32(device, queue, &self.o_w_buf, dim * dim);
        let biases = Self::read_buffer_f32(device, queue, &self.bias_buf, 3 * dim + dim);
        let (q_b, rest) = biases.split_at(dim);
        let (k_b, rest) = rest.split_at(dim);
        let (v_b, _o_b) = rest.split_at(dim);
        // Compute Q, K, V
        let mut q = vec![0.0; dim];
        let mut k = vec![0.0; dim];
        let mut v = vec![0.0; dim];
        for i in 0..dim {
            for j in 0..dim {
                q[i] += input[j] * q_w[i * dim + j];
                k[i] += input[j] * k_w[i * dim + j];
                v[i] += input[j] * v_w[i * dim + j];
            }
            q[i] += q_b[i];
            k[i] += k_b[i];
            v[i] += v_b[i];
        }
        // Compute attention scores and weights (single-token: score=dot(q,k)/sqrt(dim))
        let mut _attn_scores = 0.0;
        for i in 0..dim {
            _attn_scores += q[i] * k[i];
        }
        _attn_scores /= (dim as f32).sqrt();
        let attn_weights = vec![1.0]; // single-token: softmax([score]) = 1.0
        // Compute attn_out = attn_weights * V (single-token)
        let attn_out = v.clone();
        (q, k, v, attn_weights, attn_out)
    }
    /// Update all attention parameters using gradients from AttentionBackwardGradients
    pub fn update_parameters_from_backward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grads: &AttentionGradients,
        lr: f32,
    ) {
        let dim = self.in_dim;
        let mut o_w = Self::read_buffer_f32(device, queue, &self.o_w_buf, dim * dim);
        let mut v_w = Self::read_buffer_f32(device, queue, &self.v_w_buf, dim * dim);
        let mut q_w = Self::read_buffer_f32(device, queue, &self.q_w_buf, dim * dim);
        let mut k_w = Self::read_buffer_f32(device, queue, &self.k_w_buf, dim * dim);
        let mut biases = Self::read_buffer_f32(device, queue, &self.bias_buf, 3 * dim + dim);
        update_weights(&mut o_w, &grads.d_o_w, lr);
        update_weights(&mut v_w, &grads.d_v_w, lr);
        update_weights(&mut q_w, &grads.d_q_w, lr);
        update_weights(&mut k_w, &grads.d_k_w, lr);
        update_bias(biases.as_mut_slice(), &grads.d_v_b, lr);
        update_bias(biases.as_mut_slice(), &grads.d_q_b, lr);
        update_bias(biases.as_mut_slice(), &grads.d_k_b, lr);
        update_bias(biases.as_mut_slice(), &grads.d_o_b, lr);
        self.set_o_w(queue, &o_w);
        self.set_v_w(queue, biases.as_slice());
        self.set_q_w(queue, biases.as_slice());
        self.set_k_w(queue, biases.as_slice());
        self.set_biases(
            queue,
            biases.as_slice(),
            biases.as_slice(),
            biases.as_slice(),
            biases.as_slice(),
        );
    }
}
