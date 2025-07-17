use crate::gpu_transformer::blocks::linear::{BatchContext, OptimizerParams, VulkanLinearLayer};
use crate::gpu_transformer::compute_pipeline::VulkanComputePipeline;
use crate::gpu_transformer::vulkan_context::VulkanContext;
use ash::vk;
use std::error::Error;

pub struct VulkanGPUAttention {
    q_proj: VulkanLinearLayer,
    k_proj: VulkanLinearLayer,
    v_proj: VulkanLinearLayer,
    out_proj: VulkanLinearLayer,
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    context: VulkanContext,

    // GPU resources for forward pass
    scores_pipeline: VulkanComputePipeline,
    softmax_pipeline: VulkanComputePipeline,
    apply_pipeline: VulkanComputePipeline,

    // GPU resources for backward pass
    apply_backward_pipeline: VulkanComputePipeline,
    softmax_backward_pipeline: VulkanComputePipeline,
    scores_backward_pipeline: VulkanComputePipeline,

    // Buffers
    #[allow(dead_code)]
    scores_buffer: vk::Buffer,
    #[allow(dead_code)]
    scores_memory: vk::DeviceMemory,
    #[allow(dead_code)]
    softmax_buffer: vk::Buffer,
    #[allow(dead_code)]
    softmax_memory: vk::DeviceMemory,
    #[allow(dead_code)]
    attention_output_buffer: vk::Buffer,
    #[allow(dead_code)]
    attention_output_memory: vk::DeviceMemory,
}

impl VulkanGPUAttention {
    pub fn new(
        context: &VulkanContext,
        embed_dim: usize,
        num_heads: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let head_dim = embed_dim / num_heads;

        // Create linear projections
        let q_proj = VulkanLinearLayer::new(embed_dim, embed_dim)?;
        let k_proj = VulkanLinearLayer::new(embed_dim, embed_dim)?;
        let v_proj = VulkanLinearLayer::new(embed_dim, embed_dim)?;
        let out_proj = VulkanLinearLayer::new(embed_dim, embed_dim)?;

        // Create compute pipelines
        let scores_pipeline = VulkanComputePipeline::new(
            &context.ash_device,
            &context.ash_instance,
            context.physical_device,
        )?;

        let softmax_pipeline = VulkanComputePipeline::new(
            &context.ash_device,
            &context.ash_instance,
            context.physical_device,
        )?;

        let apply_pipeline = VulkanComputePipeline::new(
            &context.ash_device,
            &context.ash_instance,
            context.physical_device,
        )?;

        // Create backward compute pipelines
        let apply_backward_pipeline = VulkanComputePipeline::new(
            &context.ash_device,
            &context.ash_instance,
            context.physical_device,
        )?;

        let softmax_backward_pipeline = VulkanComputePipeline::new(
            &context.ash_device,
            &context.ash_instance,
            context.physical_device,
        )?;

        let scores_backward_pipeline = VulkanComputePipeline::new(
            &context.ash_device,
            &context.ash_instance,
            context.physical_device,
        )?;

        // Create placeholder buffers (will be recreated during forward pass)
        let (scores_buffer, scores_memory) = (vk::Buffer::null(), vk::DeviceMemory::null());
        let (softmax_buffer, softmax_memory) = (vk::Buffer::null(), vk::DeviceMemory::null());
        let (attention_output_buffer, attention_output_memory) =
            (vk::Buffer::null(), vk::DeviceMemory::null());

        Ok(VulkanGPUAttention {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            embed_dim,
            num_heads,
            head_dim,
            context: context.clone(),
            scores_pipeline,
            softmax_pipeline,
            apply_pipeline,
            apply_backward_pipeline,
            softmax_backward_pipeline,
            scores_backward_pipeline,
            scores_buffer,
            scores_memory,
            softmax_buffer,
            softmax_memory,
            attention_output_buffer,
            attention_output_memory,
        })
    }

    pub fn forward(
        &self,
        input: &[f32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        // Project to Q, K, V using existing linear layers
        let weights_q = self.q_proj.get_weights()?;
        let bias_q = self.q_proj.get_bias()?;
        let q = self
            .q_proj
            .forward(input, &weights_q, &bias_q, self.embed_dim, self.embed_dim)?;

        let weights_k = self.k_proj.get_weights()?;
        let bias_k = self.k_proj.get_bias()?;
        let k = self
            .k_proj
            .forward(input, &weights_k, &bias_k, self.embed_dim, self.embed_dim)?;

        let weights_v = self.v_proj.get_weights()?;
        let bias_v = self.v_proj.get_bias()?;
        let v = self
            .v_proj
            .forward(input, &weights_v, &bias_v, self.embed_dim, self.embed_dim)?;

        // Calculate buffer sizes
        let scores_size =
            batch_size * self.num_heads * seq_len * seq_len * std::mem::size_of::<f32>();
        let softmax_size = scores_size;
        let attention_output_size =
            batch_size * self.num_heads * seq_len * self.head_dim * std::mem::size_of::<f32>();

        // Create buffers using pipeline methods
        let scores_buffer = self.scores_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            scores_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let softmax_buffer = self.softmax_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            softmax_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let attention_output_buffer = self.apply_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            attention_output_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Create input buffers for Q, K, V
        let q_buffer = self.scores_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            (q.len() * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        q_buffer.upload_data(&self.context.ash_device, &q)?;

        let k_buffer = self.scores_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            (k.len() * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        k_buffer.upload_data(&self.context.ash_device, &k)?;

        let v_buffer = self.scores_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            (v.len() * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        v_buffer.upload_data(&self.context.ash_device, &v)?;

        // Step 1: Compute attention scores on GPU
        let scores_descriptor_set = self.scores_pipeline.create_descriptor_set()?;
        self.scores_pipeline.update_descriptor_set(
            scores_descriptor_set,
            0,
            q_buffer.buffer,
            0,
            q_buffer.size,
        );
        self.scores_pipeline.update_descriptor_set(
            scores_descriptor_set,
            1,
            k_buffer.buffer,
            0,
            k_buffer.size,
        );
        self.scores_pipeline.update_descriptor_set(
            scores_descriptor_set,
            2,
            scores_buffer.buffer,
            0,
            scores_buffer.size,
        );

        let (scores_cmd_buffer, scores_cmd_pool) = self.context.create_command_buffer()?;
        unsafe {
            self.context.ash_device.begin_command_buffer(
                scores_cmd_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            )?;
        }

        self.scores_pipeline.bind_pipeline(scores_cmd_buffer);
        self.scores_pipeline
            .bind_descriptor_sets(scores_cmd_buffer, scores_descriptor_set);

        // Push constants: batch_size, seq_len, num_heads, head_dim, embed_dim
        let scores_constants = [
            batch_size as u32,
            seq_len as u32,
            self.num_heads as u32,
            self.head_dim as u32,
            self.embed_dim as u32,
        ];
        self.scores_pipeline
            .push_constants_u32(scores_cmd_buffer, &scores_constants);

        let group_count = (batch_size * self.num_heads * seq_len * seq_len).div_ceil(256);
        self.scores_pipeline
            .dispatch(scores_cmd_buffer, group_count as u32, 1, 1);

        unsafe {
            self.context
                .ash_device
                .end_command_buffer(scores_cmd_buffer)?;
        }
        self.context.submit_and_wait(scores_cmd_buffer)?;
        self.context.destroy_command_pool(scores_cmd_pool);
        self.scores_pipeline
            .free_descriptor_set(scores_descriptor_set)?;

        // Step 2: Apply softmax on GPU
        let softmax_descriptor_set = self.softmax_pipeline.create_descriptor_set()?;
        self.softmax_pipeline.update_descriptor_set(
            softmax_descriptor_set,
            0,
            scores_buffer.buffer,
            0,
            scores_buffer.size,
        );
        self.softmax_pipeline.update_descriptor_set(
            softmax_descriptor_set,
            1,
            softmax_buffer.buffer,
            0,
            softmax_buffer.size,
        );

        let (softmax_cmd_buffer, softmax_cmd_pool) = self.context.create_command_buffer()?;
        unsafe {
            self.context.ash_device.begin_command_buffer(
                softmax_cmd_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            )?;
        }

        self.softmax_pipeline.bind_pipeline(softmax_cmd_buffer);
        self.softmax_pipeline
            .bind_descriptor_sets(softmax_cmd_buffer, softmax_descriptor_set);

        // Push constants: batch_size, seq_len, num_heads, head_dim, embed_dim
        self.softmax_pipeline
            .push_constants_u32(softmax_cmd_buffer, &scores_constants);

        let group_count = (batch_size * self.num_heads * seq_len * seq_len).div_ceil(256);
        self.softmax_pipeline
            .dispatch(softmax_cmd_buffer, group_count as u32, 1, 1);

        unsafe {
            self.context
                .ash_device
                .end_command_buffer(softmax_cmd_buffer)?;
        }
        self.context.submit_and_wait(softmax_cmd_buffer)?;
        self.context.destroy_command_pool(softmax_cmd_pool);
        self.softmax_pipeline
            .free_descriptor_set(softmax_descriptor_set)?;

        // Step 3: Apply attention weights to values on GPU
        let apply_descriptor_set = self.apply_pipeline.create_descriptor_set()?;
        self.apply_pipeline.update_descriptor_set(
            apply_descriptor_set,
            0,
            softmax_buffer.buffer,
            0,
            softmax_buffer.size,
        );
        self.apply_pipeline.update_descriptor_set(
            apply_descriptor_set,
            1,
            v_buffer.buffer,
            0,
            v_buffer.size,
        );
        self.apply_pipeline.update_descriptor_set(
            apply_descriptor_set,
            2,
            attention_output_buffer.buffer,
            0,
            attention_output_buffer.size,
        );

        let (apply_cmd_buffer, apply_cmd_pool) = self.context.create_command_buffer()?;
        unsafe {
            self.context.ash_device.begin_command_buffer(
                apply_cmd_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            )?;
        }

        self.apply_pipeline.bind_pipeline(apply_cmd_buffer);
        self.apply_pipeline
            .bind_descriptor_sets(apply_cmd_buffer, apply_descriptor_set);

        // Push constants: batch_size, seq_len, num_heads, head_dim, embed_dim
        self.apply_pipeline
            .push_constants_u32(apply_cmd_buffer, &scores_constants);

        let group_count = (batch_size * self.num_heads * seq_len * self.head_dim).div_ceil(256);
        self.apply_pipeline
            .dispatch(apply_cmd_buffer, group_count as u32, 1, 1);

        unsafe {
            self.context
                .ash_device
                .end_command_buffer(apply_cmd_buffer)?;
        }
        self.context.submit_and_wait(apply_cmd_buffer)?;
        self.context.destroy_command_pool(apply_cmd_pool);
        self.apply_pipeline
            .free_descriptor_set(apply_descriptor_set)?;

        // Read back attention output
        let mut attention_output =
            vec![0.0f32; batch_size * self.num_heads * seq_len * self.head_dim];
        attention_output_buffer.download_data(&self.context.ash_device, &mut attention_output)?;

        // Reshape attention output to match expected format
        let mut output = vec![0.0f32; batch_size * seq_len * self.embed_dim];
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for d in 0..self.head_dim {
                        let src_idx = b * self.num_heads * seq_len * self.head_dim
                            + h * seq_len * self.head_dim
                            + i * self.head_dim
                            + d;
                        let dst_idx = b * seq_len * self.embed_dim
                            + i * self.embed_dim
                            + h * self.head_dim
                            + d;
                        output[dst_idx] = attention_output[src_idx];
                    }
                }
            }
        }

        // Final output projection
        let weights_o = self.out_proj.get_weights()?;
        let bias_o = self.out_proj.get_bias()?;
        let final_out =
            self.out_proj
                .forward(&output, &weights_o, &bias_o, self.embed_dim, self.embed_dim)?;

        Ok(final_out)
    }

    pub fn train_batch_adam(
        &mut self,
        input: &[f32],
        target: &[f32],
        opt: &OptimizerParams,
        batch: &BatchContext,
    ) -> Result<f32, Box<dyn Error>> {
        // Forward pass using GPU attention
        let output = self.forward(input, batch.batch_size, batch.seq_len)?;

        // Compute loss (MSE)
        let mut loss = 0.0;
        let mut output_grad = vec![0.0f32; output.len()];
        for i in 0..output.len() {
            let diff = output[i] - target[i];
            loss += diff * diff;
            output_grad[i] = 2.0 * diff; // MSE gradient
        }
        loss /= output.len() as f32;

        // Get current Q, K, V projections for backward pass
        let weights_q = self.q_proj.get_weights()?;
        let bias_q = self.q_proj.get_bias()?;
        let q = self
            .q_proj
            .forward(input, &weights_q, &bias_q, self.embed_dim, self.embed_dim)?;

        let weights_k = self.k_proj.get_weights()?;
        let bias_k = self.k_proj.get_bias()?;
        let k = self
            .k_proj
            .forward(input, &weights_k, &bias_k, self.embed_dim, self.embed_dim)?;

        let weights_v = self.v_proj.get_weights()?;
        let bias_v = self.v_proj.get_bias()?;
        let v = self
            .v_proj
            .forward(input, &weights_v, &bias_v, self.embed_dim, self.embed_dim)?;

        // Calculate buffer sizes for backward pass
        let scores_size = batch.batch_size
            * self.num_heads
            * batch.seq_len
            * batch.seq_len
            * std::mem::size_of::<f32>();
        let softmax_size = scores_size;
        let attention_output_size = batch.batch_size
            * self.num_heads
            * batch.seq_len
            * self.head_dim
            * std::mem::size_of::<f32>();
        let qkv_size =
            batch.batch_size * batch.seq_len * self.embed_dim * std::mem::size_of::<f32>();

        // Create buffers for backward pass
        let grad_attn_out_buffer = self.apply_backward_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            attention_output_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        grad_attn_out_buffer.upload_data(&self.context.ash_device, &output_grad)?;

        let grad_softmax_buffer = self.apply_backward_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            softmax_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let zeros_softmax =
            vec![0.0f32; batch.batch_size * self.num_heads * batch.seq_len * batch.seq_len];
        grad_softmax_buffer.upload_data(&self.context.ash_device, &zeros_softmax)?;

        let grad_v_buffer = self.apply_backward_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            qkv_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let zeros_v = vec![0.0f32; batch.batch_size * batch.seq_len * self.embed_dim];
        grad_v_buffer.upload_data(&self.context.ash_device, &zeros_v)?;

        let grad_scores_buffer = self.softmax_backward_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            scores_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let zeros_scores =
            vec![0.0f32; batch.batch_size * self.num_heads * batch.seq_len * batch.seq_len];
        grad_scores_buffer.upload_data(&self.context.ash_device, &zeros_scores)?;

        let grad_q_buffer = self.scores_backward_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            qkv_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let zeros_q = vec![0.0f32; batch.batch_size * batch.seq_len * self.embed_dim];
        grad_q_buffer.upload_data(&self.context.ash_device, &zeros_q)?;

        let grad_k_buffer = self.scores_backward_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            qkv_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let zeros_k = vec![0.0f32; batch.batch_size * batch.seq_len * self.embed_dim];
        grad_k_buffer.upload_data(&self.context.ash_device, &zeros_k)?;

        // Create input buffers for Q, K, V
        let q_buffer = self.scores_backward_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            qkv_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        q_buffer.upload_data(&self.context.ash_device, &q)?;

        let k_buffer = self.scores_backward_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            qkv_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        k_buffer.upload_data(&self.context.ash_device, &k)?;

        let v_buffer = self.apply_backward_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            qkv_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        v_buffer.upload_data(&self.context.ash_device, &v)?;

        // Recreate forward pass buffers for backward computation
        let scores_buffer = self.scores_backward_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            scores_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let softmax_buffer = self.softmax_backward_pipeline.create_buffer(
            &self.context.ash_instance,
            self.context.physical_device,
            softmax_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Re-run forward pass to get intermediate values for backward pass
        // Step 1: Compute attention scores
        let scores_descriptor_set = self.scores_pipeline.create_descriptor_set()?;
        self.scores_pipeline.update_descriptor_set(
            scores_descriptor_set,
            0,
            q_buffer.buffer,
            0,
            q_buffer.size,
        );
        self.scores_pipeline.update_descriptor_set(
            scores_descriptor_set,
            1,
            k_buffer.buffer,
            0,
            k_buffer.size,
        );
        self.scores_pipeline.update_descriptor_set(
            scores_descriptor_set,
            2,
            scores_buffer.buffer,
            0,
            scores_buffer.size,
        );

        let (scores_cmd_buffer, scores_cmd_pool) = self.context.create_command_buffer()?;
        unsafe {
            self.context.ash_device.begin_command_buffer(
                scores_cmd_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            )?;
        }

        self.scores_pipeline.bind_pipeline(scores_cmd_buffer);
        self.scores_pipeline
            .bind_descriptor_sets(scores_cmd_buffer, scores_descriptor_set);

        let scores_constants = [
            batch.batch_size as u32,
            batch.seq_len as u32,
            self.num_heads as u32,
            self.head_dim as u32,
            self.embed_dim as u32,
        ];
        self.scores_pipeline
            .push_constants_u32(scores_cmd_buffer, &scores_constants);

        let group_count =
            (batch.batch_size * self.num_heads * batch.seq_len * batch.seq_len).div_ceil(256);
        self.scores_pipeline
            .dispatch(scores_cmd_buffer, group_count as u32, 1, 1);

        unsafe {
            self.context
                .ash_device
                .end_command_buffer(scores_cmd_buffer)?;
        }
        self.context.submit_and_wait(scores_cmd_buffer)?;
        self.context.destroy_command_pool(scores_cmd_pool);
        self.scores_pipeline
            .free_descriptor_set(scores_descriptor_set)?;

        // Step 2: Apply softmax
        let softmax_descriptor_set = self.softmax_pipeline.create_descriptor_set()?;
        self.softmax_pipeline.update_descriptor_set(
            softmax_descriptor_set,
            0,
            scores_buffer.buffer,
            0,
            scores_buffer.size,
        );
        self.softmax_pipeline.update_descriptor_set(
            softmax_descriptor_set,
            1,
            softmax_buffer.buffer,
            0,
            softmax_buffer.size,
        );

        let (softmax_cmd_buffer, softmax_cmd_pool) = self.context.create_command_buffer()?;
        unsafe {
            self.context.ash_device.begin_command_buffer(
                softmax_cmd_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            )?;
        }

        self.softmax_pipeline.bind_pipeline(softmax_cmd_buffer);
        self.softmax_pipeline
            .bind_descriptor_sets(softmax_cmd_buffer, softmax_descriptor_set);
        self.softmax_pipeline
            .push_constants_u32(softmax_cmd_buffer, &scores_constants);

        let group_count =
            (batch.batch_size * self.num_heads * batch.seq_len * batch.seq_len).div_ceil(256);
        self.softmax_pipeline
            .dispatch(softmax_cmd_buffer, group_count as u32, 1, 1);

        unsafe {
            self.context
                .ash_device
                .end_command_buffer(softmax_cmd_buffer)?;
        }
        self.context.submit_and_wait(softmax_cmd_buffer)?;
        self.context.destroy_command_pool(softmax_cmd_pool);
        self.softmax_pipeline
            .free_descriptor_set(softmax_descriptor_set)?;

        // Now run backward pass
        // Step 1: Backward through attention apply
        let apply_backward_descriptor_set = self.apply_backward_pipeline.create_descriptor_set()?;
        self.apply_backward_pipeline.update_descriptor_set(
            apply_backward_descriptor_set,
            0,
            grad_attn_out_buffer.buffer,
            0,
            grad_attn_out_buffer.size,
        );
        self.apply_backward_pipeline.update_descriptor_set(
            apply_backward_descriptor_set,
            1,
            softmax_buffer.buffer,
            0,
            softmax_buffer.size,
        );
        self.apply_backward_pipeline.update_descriptor_set(
            apply_backward_descriptor_set,
            2,
            v_buffer.buffer,
            0,
            v_buffer.size,
        );
        self.apply_backward_pipeline.update_descriptor_set(
            apply_backward_descriptor_set,
            3,
            grad_softmax_buffer.buffer,
            0,
            grad_softmax_buffer.size,
        );
        self.apply_backward_pipeline.update_descriptor_set(
            apply_backward_descriptor_set,
            4,
            grad_v_buffer.buffer,
            0,
            grad_v_buffer.size,
        );

        let (apply_backward_cmd_buffer, apply_backward_cmd_pool) =
            self.context.create_command_buffer()?;
        unsafe {
            self.context.ash_device.begin_command_buffer(
                apply_backward_cmd_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            )?;
        }

        self.apply_backward_pipeline
            .bind_pipeline(apply_backward_cmd_buffer);
        self.apply_backward_pipeline
            .bind_descriptor_sets(apply_backward_cmd_buffer, apply_backward_descriptor_set);
        self.apply_backward_pipeline
            .push_constants_u32(apply_backward_cmd_buffer, &scores_constants);

        let group_count =
            (batch.batch_size * self.num_heads * batch.seq_len * self.head_dim).div_ceil(256);
        self.apply_backward_pipeline
            .dispatch(apply_backward_cmd_buffer, group_count as u32, 1, 1);

        unsafe {
            self.context
                .ash_device
                .end_command_buffer(apply_backward_cmd_buffer)?;
        }
        self.context.submit_and_wait(apply_backward_cmd_buffer)?;
        self.context.destroy_command_pool(apply_backward_cmd_pool);
        self.apply_backward_pipeline
            .free_descriptor_set(apply_backward_descriptor_set)?;

        // Step 2: Backward through softmax
        let softmax_backward_descriptor_set =
            self.softmax_backward_pipeline.create_descriptor_set()?;
        self.softmax_backward_pipeline.update_descriptor_set(
            softmax_backward_descriptor_set,
            0,
            grad_softmax_buffer.buffer,
            0,
            grad_softmax_buffer.size,
        );
        self.softmax_backward_pipeline.update_descriptor_set(
            softmax_backward_descriptor_set,
            1,
            scores_buffer.buffer,
            0,
            scores_buffer.size,
        );
        self.softmax_backward_pipeline.update_descriptor_set(
            softmax_backward_descriptor_set,
            2,
            grad_scores_buffer.buffer,
            0,
            grad_scores_buffer.size,
        );

        let (softmax_backward_cmd_buffer, softmax_backward_cmd_pool) =
            self.context.create_command_buffer()?;
        unsafe {
            self.context.ash_device.begin_command_buffer(
                softmax_backward_cmd_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            )?;
        }

        self.softmax_backward_pipeline
            .bind_pipeline(softmax_backward_cmd_buffer);
        self.softmax_backward_pipeline
            .bind_descriptor_sets(softmax_backward_cmd_buffer, softmax_backward_descriptor_set);
        self.softmax_backward_pipeline
            .push_constants_u32(softmax_backward_cmd_buffer, &scores_constants);

        let group_count =
            (batch.batch_size * self.num_heads * batch.seq_len * batch.seq_len).div_ceil(256);
        self.softmax_backward_pipeline.dispatch(
            softmax_backward_cmd_buffer,
            group_count as u32,
            1,
            1,
        );

        unsafe {
            self.context
                .ash_device
                .end_command_buffer(softmax_backward_cmd_buffer)?;
        }
        self.context.submit_and_wait(softmax_backward_cmd_buffer)?;
        self.context.destroy_command_pool(softmax_backward_cmd_pool);
        self.softmax_backward_pipeline
            .free_descriptor_set(softmax_backward_descriptor_set)?;

        // Step 3: Backward through scores
        let scores_backward_descriptor_set =
            self.scores_backward_pipeline.create_descriptor_set()?;
        self.scores_backward_pipeline.update_descriptor_set(
            scores_backward_descriptor_set,
            0,
            grad_scores_buffer.buffer,
            0,
            grad_scores_buffer.size,
        );
        self.scores_backward_pipeline.update_descriptor_set(
            scores_backward_descriptor_set,
            1,
            q_buffer.buffer,
            0,
            q_buffer.size,
        );
        self.scores_backward_pipeline.update_descriptor_set(
            scores_backward_descriptor_set,
            2,
            k_buffer.buffer,
            0,
            k_buffer.size,
        );
        self.scores_backward_pipeline.update_descriptor_set(
            scores_backward_descriptor_set,
            3,
            grad_q_buffer.buffer,
            0,
            grad_q_buffer.size,
        );
        self.scores_backward_pipeline.update_descriptor_set(
            scores_backward_descriptor_set,
            4,
            grad_k_buffer.buffer,
            0,
            grad_k_buffer.size,
        );

        let (scores_backward_cmd_buffer, scores_backward_cmd_pool) =
            self.context.create_command_buffer()?;
        unsafe {
            self.context.ash_device.begin_command_buffer(
                scores_backward_cmd_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            )?;
        }

        self.scores_backward_pipeline
            .bind_pipeline(scores_backward_cmd_buffer);
        self.scores_backward_pipeline
            .bind_descriptor_sets(scores_backward_cmd_buffer, scores_backward_descriptor_set);
        self.scores_backward_pipeline
            .push_constants_u32(scores_backward_cmd_buffer, &scores_constants);

        let group_count =
            (batch.batch_size * self.num_heads * batch.seq_len * batch.seq_len).div_ceil(256);
        self.scores_backward_pipeline.dispatch(
            scores_backward_cmd_buffer,
            group_count as u32,
            1,
            1,
        );

        unsafe {
            self.context
                .ash_device
                .end_command_buffer(scores_backward_cmd_buffer)?;
        }
        self.context.submit_and_wait(scores_backward_cmd_buffer)?;
        self.context.destroy_command_pool(scores_backward_cmd_pool);
        self.scores_backward_pipeline
            .free_descriptor_set(scores_backward_descriptor_set)?;

        // Download gradients
        let mut grad_q = vec![0.0f32; batch.batch_size * batch.seq_len * self.embed_dim];
        grad_q_buffer.download_data(&self.context.ash_device, &mut grad_q)?;

        let mut grad_k = vec![0.0f32; batch.batch_size * batch.seq_len * self.embed_dim];
        grad_k_buffer.download_data(&self.context.ash_device, &mut grad_k)?;

        let mut grad_v = vec![0.0f32; batch.batch_size * batch.seq_len * self.embed_dim];
        grad_v_buffer.download_data(&self.context.ash_device, &mut grad_v)?;

        // Train all projections with proper gradients
        let batch = BatchContext {
            batch_size: batch.batch_size * batch.seq_len,
            seq_len: 1,
        };
        let opt = OptimizerParams {
            learning_rate: opt.learning_rate,
            beta1: opt.beta1,
            beta2: opt.beta2,
            epsilon: opt.epsilon,
        };

        self.out_proj
            .train_batch_adam(input, target, &opt, &batch)?;

        // Use computed gradients for Q, K, V projections
        self.q_proj.train_batch_adam(input, &grad_q, &opt, &batch)?;
        self.k_proj.train_batch_adam(input, &grad_k, &opt, &batch)?;
        self.v_proj.train_batch_adam(input, &grad_v, &opt, &batch)?;

        Ok(loss)
    }
}
