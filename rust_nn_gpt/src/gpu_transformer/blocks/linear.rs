use crate::gpu_transformer::compute_pipeline::{
    VulkanBatchedPipeline, VulkanBuffer, VulkanComputePipeline,
};
use crate::gpu_transformer::vulkan_context::VulkanContext;
use ash::vk;
use std::error::Error;

pub struct OptimizerParams {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

pub struct BatchContext {
    pub batch_size: usize,
    pub seq_len: usize,
}

pub struct VulkanLinearLayer {
    pipeline: VulkanComputePipeline,
    forward_pipeline: VulkanBatchedPipeline,
    adam_pipeline: VulkanBatchedPipeline,
    context: VulkanContext,
    // Persistent GPU buffers for parameters and Adam state
    weights: VulkanBuffer, // [out_dim, in_dim]
    bias: VulkanBuffer,    // [out_dim]
    m_w: VulkanBuffer,     // Adam m for weights
    v_w: VulkanBuffer,     // Adam v for weights
    m_b: VulkanBuffer,     // Adam m for bias
    v_b: VulkanBuffer,     // Adam v for bias
    in_dim: usize,
    out_dim: usize,
    timestep: f32, // Adam timestep
}

impl VulkanLinearLayer {
    pub fn new(in_dim: usize, out_dim: usize) -> Result<Self, Box<dyn Error>> {
        let context = VulkanContext::new()?;
        let pipeline = VulkanComputePipeline::new(
            &context.ash_device,
            &context.ash_instance,
            context.physical_device,
        )?;

        // Create batched pipelines for training
        let forward_pipeline = VulkanBatchedPipeline::new(
            &context.ash_device,
            &context.ash_instance,
            context.physical_device,
            4, // input, weights, bias, output
            "linear_batched_forward.comp",
            16, // 3 u32s + padding = 16 bytes
        )?;

        let adam_pipeline = VulkanBatchedPipeline::new(
            &context.ash_device,
            &context.ash_instance,
            context.physical_device,
            8, // weights, bias, weight_grad, bias_grad, weight_m, weight_v, bias_m, bias_v
            "adam_update.comp",
            32, // 8 floats = 32 bytes
        )?;

        let instance = &context.ash_instance;
        let physical_device = context.physical_device;
        let device = &context.ash_device;
        // Initialize weights and bias to small random values, Adam state to zero
        let w_size = out_dim * in_dim;
        let b_size = out_dim;
        let mut w_init = vec![0.0f32; w_size];
        for w in &mut w_init {
            *w = (rand::random::<f32>() - 0.5) * 0.1;
        }
        let b_init = vec![0.0f32; b_size];
        let zeros_w = vec![0.0f32; w_size];
        let zeros_b = vec![0.0f32; b_size];
        let weights = pipeline.create_buffer(
            instance,
            physical_device,
            (w_size * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        weights.upload_data(device, &w_init)?;
        let bias = pipeline.create_buffer(
            instance,
            physical_device,
            (b_size * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        bias.upload_data(device, &b_init)?;
        let m_w = pipeline.create_buffer(
            instance,
            physical_device,
            (w_size * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        m_w.upload_data(device, &zeros_w)?;
        let v_w = pipeline.create_buffer(
            instance,
            physical_device,
            (w_size * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        v_w.upload_data(device, &zeros_w)?;
        let m_b = pipeline.create_buffer(
            instance,
            physical_device,
            (b_size * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        m_b.upload_data(device, &zeros_b)?;
        let v_b = pipeline.create_buffer(
            instance,
            physical_device,
            (b_size * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        v_b.upload_data(device, &zeros_b)?;
        Ok(VulkanLinearLayer {
            pipeline,
            forward_pipeline,
            adam_pipeline,
            context,
            weights,
            bias,
            m_w,
            v_w,
            m_b,
            v_b,
            in_dim,
            out_dim,
            timestep: 0.0,
        })
    }

    pub fn forward(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        input_size: usize,
        output_size: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Calculate batch size from input length
        let batch_size = input.len() / input_size;
        let total_output_size = batch_size * output_size;

        // For now, fall back to CPU implementation for the entire batch
        let mut output = vec![0.0f32; total_output_size];

        for batch_idx in 0..batch_size {
            let input_offset = batch_idx * input_size;
            let output_offset = batch_idx * output_size;

            for i in 0..output_size {
                output[output_offset + i] = bias[i];
                for j in 0..input_size {
                    output[output_offset + i] +=
                        input[input_offset + j] * weights[i * input_size + j];
                }
            }
        }

        Ok(output)
    }

    /// Get the current weights from GPU memory
    pub fn get_weights(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut weights = vec![0.0f32; self.in_dim * self.out_dim];
        self.weights
            .download_data(&self.context.ash_device, &mut weights)?;
        Ok(weights)
    }

    /// Get the current bias from GPU memory
    pub fn get_bias(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut bias = vec![0.0f32; self.out_dim];
        self.bias
            .download_data(&self.context.ash_device, &mut bias)?;
        Ok(bias)
    }

    pub fn train_batch_adam(
        &mut self,
        input: &[f32],
        target: &[f32],
        opt: &OptimizerParams,
        batch: &BatchContext,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let in_dim = input.len() / batch.batch_size;
        let out_dim = target.len() / batch.batch_size;
        let physical_device = self.context.physical_device;
        let instance = &self.context.ash_instance;
        let device = &self.context.ash_device;

        // Create temporary buffers for this batch
        let input_buffer = self.pipeline.create_buffer(
            instance,
            physical_device,
            std::mem::size_of_val(input) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        input_buffer.upload_data(device, input)?;

        let target_buffer = self.pipeline.create_buffer(
            instance,
            physical_device,
            std::mem::size_of_val(target) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        target_buffer.upload_data(device, target)?;

        let output_buffer = self.pipeline.create_buffer(
            instance,
            physical_device,
            std::mem::size_of_val(target) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Create gradient buffers
        let weight_grad_buffer = self.pipeline.create_buffer(
            instance,
            physical_device,
            (self.in_dim * self.out_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        // Initialize weight gradients to zero
        let zeros_w = vec![0.0f32; self.in_dim * self.out_dim];
        weight_grad_buffer.upload_data(device, &zeros_w)?;

        let bias_grad_buffer = self.pipeline.create_buffer(
            instance,
            physical_device,
            (self.out_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        // Initialize bias gradients to zero
        let zeros_b = vec![0.0f32; self.out_dim];
        bias_grad_buffer.upload_data(device, &zeros_b)?;

        // Step 1: Forward pass
        let forward_descriptor_set = self.forward_pipeline.create_descriptor_set()?;
        self.forward_pipeline.update_descriptor_set(
            forward_descriptor_set,
            0,
            input_buffer.buffer,
            0,
            input_buffer.size,
        );
        self.forward_pipeline.update_descriptor_set(
            forward_descriptor_set,
            1,
            self.weights.buffer,
            0,
            self.weights.size,
        );
        self.forward_pipeline.update_descriptor_set(
            forward_descriptor_set,
            2,
            self.bias.buffer,
            0,
            self.bias.size,
        );
        self.forward_pipeline.update_descriptor_set(
            forward_descriptor_set,
            3,
            output_buffer.buffer,
            0,
            output_buffer.size,
        );

        let (forward_cmd_buffer, forward_cmd_pool) = self.context.create_command_buffer()?;
        unsafe {
            self.context.ash_device.begin_command_buffer(
                forward_cmd_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            )?;
        }

        self.forward_pipeline.bind_pipeline(forward_cmd_buffer);
        self.forward_pipeline
            .bind_descriptor_sets(forward_cmd_buffer, forward_descriptor_set);

        // Push constants: input_size, output_size, batch_size (as u32)
        let forward_constants = [in_dim as u32, out_dim as u32, batch.batch_size as u32];
        self.forward_pipeline
            .push_constants_batched_u32(forward_cmd_buffer, &forward_constants);

        let group_size = 256;
        let group_count = (batch.batch_size * out_dim).div_ceil(group_size);
        self.forward_pipeline
            .dispatch(forward_cmd_buffer, group_count as u32, 1, 1);

        unsafe {
            self.context
                .ash_device
                .end_command_buffer(forward_cmd_buffer)?;
        }
        self.context.submit_and_wait(forward_cmd_buffer)?;
        self.context.destroy_command_pool(forward_cmd_pool);
        self.forward_pipeline
            .free_descriptor_set(forward_descriptor_set)?;

        // Step 2: Compute loss and gradients
        let mut output = vec![0.0f32; target.len()];
        output_buffer.download_data(device, &mut output)?;

        // Compute loss and gradients for the entire batch
        let mut loss = 0.0;
        let mut output_grad = vec![0.0f32; target.len()];

        for i in 0..target.len() {
            let diff = output[i] - target[i];
            loss += diff * diff;
            output_grad[i] = 2.0 * diff; // MSE gradient
        }
        loss /= target.len() as f32; // Average loss over all samples

        // Step 3: Compute gradients on CPU
        let mut weight_grads = vec![0.0f32; self.in_dim * self.out_dim];
        let mut bias_grads = vec![0.0f32; self.out_dim];

        // Download current weights and bias for gradient computation
        let mut current_weights = vec![0.0f32; self.in_dim * self.out_dim];
        let mut current_bias = vec![0.0f32; self.out_dim];
        self.weights.download_data(device, &mut current_weights)?;
        self.bias.download_data(device, &mut current_bias)?;

        // Compute gradients for the entire batch
        for batch_idx in 0..batch.batch_size {
            let input_offset = batch_idx * in_dim;
            let output_offset = batch_idx * out_dim;

            for i in 0..out_dim {
                // Bias gradient (accumulate across batch)
                bias_grads[i] += output_grad[output_offset + i];

                // Weight gradients (accumulate across batch)
                for j in 0..in_dim {
                    let weight_idx = i * self.in_dim + j;
                    weight_grads[weight_idx] +=
                        output_grad[output_offset + i] * input[input_offset + j];
                }
            }
        }

        // Average gradients over batch size
        for grad in &mut weight_grads {
            *grad /= batch.batch_size as f32;
        }
        for grad in &mut bias_grads {
            *grad /= batch.batch_size as f32;
        }

        // Upload gradients to GPU buffers
        weight_grad_buffer.upload_data(device, &weight_grads)?;
        bias_grad_buffer.upload_data(device, &bias_grads)?;

        // Step 4: Adam update
        self.timestep += 1.0;
        let adam_descriptor_set = self.adam_pipeline.create_descriptor_set()?;
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            0,
            self.weights.buffer,
            0,
            self.weights.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            1,
            self.bias.buffer,
            0,
            self.bias.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            2,
            weight_grad_buffer.buffer,
            0,
            weight_grad_buffer.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            3,
            bias_grad_buffer.buffer,
            0,
            bias_grad_buffer.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            4,
            self.m_w.buffer,
            0,
            self.m_w.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            5,
            self.v_w.buffer,
            0,
            self.v_w.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            6,
            self.m_b.buffer,
            0,
            self.m_b.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            7,
            self.v_b.buffer,
            0,
            self.v_b.size,
        );

        let (adam_cmd_buffer, adam_cmd_pool) = self.context.create_command_buffer()?;
        unsafe {
            self.context.ash_device.begin_command_buffer(
                adam_cmd_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            )?;
        }

        self.adam_pipeline.bind_pipeline(adam_cmd_buffer);
        self.adam_pipeline
            .bind_descriptor_sets(adam_cmd_buffer, adam_descriptor_set);

        // Push constants: weight_size, bias_size, learning_rate, beta1, beta2, epsilon, t
        let adam_constants = [
            (self.in_dim * self.out_dim) as f32,
            self.out_dim as f32,
            opt.learning_rate,
            opt.beta1,
            opt.beta2,
            opt.epsilon,
            self.timestep,
            0.0,
        ];
        self.adam_pipeline
            .push_constants_batched(adam_cmd_buffer, &adam_constants);

        let group_count = (self.in_dim * self.out_dim).div_ceil(group_size);
        self.adam_pipeline
            .dispatch(adam_cmd_buffer, group_count as u32, 1, 1);

        unsafe {
            self.context
                .ash_device
                .end_command_buffer(adam_cmd_buffer)?;
        }
        self.context.submit_and_wait(adam_cmd_buffer)?;
        self.context.destroy_command_pool(adam_cmd_pool);
        self.adam_pipeline
            .free_descriptor_set(adam_descriptor_set)?;

        Ok(loss)
    }

    pub fn backward(
        &mut self,
        input: &[f32],
        grad_output: &[f32],
        opt: &OptimizerParams,
        batch: &BatchContext,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let in_dim = input.len() / batch.batch_size;
        let out_dim = grad_output.len() / batch.batch_size;

        // Download current weights for gradient computation
        let mut current_weights = vec![0.0f32; self.in_dim * self.out_dim];
        self.weights
            .download_data(&self.context.ash_device, &mut current_weights)?;

        // Compute input gradients: grad_input = grad_output * weights^T
        let mut grad_input = vec![0.0f32; input.len()];

        for batch_idx in 0..batch.batch_size {
            let input_offset = batch_idx * in_dim;
            let output_offset = batch_idx * out_dim;

            for i in 0..in_dim {
                for j in 0..out_dim {
                    grad_input[input_offset + i] +=
                        grad_output[output_offset + j] * current_weights[j * self.in_dim + i];
                }
            }
        }

        // Update weights using Adam
        // We need to compute weight gradients properly
        let mut weight_grads = vec![0.0f32; self.in_dim * self.out_dim];
        let mut bias_grads = vec![0.0f32; self.out_dim];

        for batch_idx in 0..batch.batch_size {
            let input_offset = batch_idx * in_dim;
            let output_offset = batch_idx * out_dim;

            for i in 0..out_dim {
                // Bias gradient (accumulate across batch)
                bias_grads[i] += grad_output[output_offset + i];

                // Weight gradients (accumulate across batch)
                for j in 0..in_dim {
                    let weight_idx = i * self.in_dim + j;
                    weight_grads[weight_idx] +=
                        grad_output[output_offset + i] * input[input_offset + j];
                }
            }
        }

        // Average gradients over batch size
        for grad in &mut weight_grads {
            *grad /= batch.batch_size as f32;
        }
        for grad in &mut bias_grads {
            *grad /= batch.batch_size as f32;
        }

        // Apply Adam update manually
        self.timestep += 1.0;
        let physical_device = self.context.physical_device;
        let instance = &self.context.ash_instance;
        let device = &self.context.ash_device;

        // Create gradient buffers
        let weight_grad_buffer = self.pipeline.create_buffer(
            instance,
            physical_device,
            (self.in_dim * self.out_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        weight_grad_buffer.upload_data(device, &weight_grads)?;

        let bias_grad_buffer = self.pipeline.create_buffer(
            instance,
            physical_device,
            (self.out_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        bias_grad_buffer.upload_data(device, &bias_grads)?;

        // Adam update
        let adam_descriptor_set = self.adam_pipeline.create_descriptor_set()?;
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            0,
            self.weights.buffer,
            0,
            self.weights.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            1,
            self.bias.buffer,
            0,
            self.bias.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            2,
            weight_grad_buffer.buffer,
            0,
            weight_grad_buffer.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            3,
            bias_grad_buffer.buffer,
            0,
            bias_grad_buffer.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            4,
            self.m_w.buffer,
            0,
            self.m_w.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            5,
            self.v_w.buffer,
            0,
            self.v_w.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            6,
            self.m_b.buffer,
            0,
            self.m_b.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            7,
            self.v_b.buffer,
            0,
            self.v_b.size,
        );

        let (adam_cmd_buffer, adam_cmd_pool) = self.context.create_command_buffer()?;
        unsafe {
            self.context.ash_device.begin_command_buffer(
                adam_cmd_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            )?;
        }

        self.adam_pipeline.bind_pipeline(adam_cmd_buffer);
        self.adam_pipeline
            .bind_descriptor_sets(adam_cmd_buffer, adam_descriptor_set);

        // Push constants: weight_size, bias_size, learning_rate, beta1, beta2, epsilon, t
        let adam_constants = [
            (self.in_dim * self.out_dim) as f32,
            self.out_dim as f32,
            opt.learning_rate,
            opt.beta1,
            opt.beta2,
            opt.epsilon,
            self.timestep,
            0.0,
        ];
        self.adam_pipeline
            .push_constants_batched(adam_cmd_buffer, &adam_constants);

        let group_size = 256;
        let group_count = (self.in_dim * self.out_dim).div_ceil(group_size);
        self.adam_pipeline
            .dispatch(adam_cmd_buffer, group_count as u32, 1, 1);

        unsafe {
            self.context
                .ash_device
                .end_command_buffer(adam_cmd_buffer)?;
        }
        self.context.submit_and_wait(adam_cmd_buffer)?;
        self.context.destroy_command_pool(adam_cmd_pool);
        self.adam_pipeline
            .free_descriptor_set(adam_descriptor_set)?;

        Ok(grad_input)
    }
}
