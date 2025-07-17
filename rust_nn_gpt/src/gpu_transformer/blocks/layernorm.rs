use crate::gpu_transformer::blocks::linear::{BatchContext, OptimizerParams};
use crate::gpu_transformer::compute_pipeline::{VulkanBatchedPipeline, VulkanBuffer};
use crate::gpu_transformer::vulkan_context::VulkanContext;
use ash::vk;
use std::error::Error;

pub struct VulkanLayerNorm {
    forward_pipeline: VulkanBatchedPipeline,
    adam_pipeline: VulkanBatchedPipeline,
    context: VulkanContext,
    // Persistent GPU buffers for parameters and Adam state
    gamma: VulkanBuffer,   // [feature_dim]
    beta: VulkanBuffer,    // [feature_dim]
    m_gamma: VulkanBuffer, // Adam m for gamma
    v_gamma: VulkanBuffer, // Adam v for gamma
    m_beta: VulkanBuffer,  // Adam m for beta
    v_beta: VulkanBuffer,  // Adam v for beta
    feature_dim: usize,
    eps: f32,
    timestep: f32, // Adam timestep
}

impl VulkanLayerNorm {
    pub fn new(feature_dim: usize) -> Result<Self, Box<dyn Error>> {
        let context = VulkanContext::new()?;

        // Create batched pipelines for forward pass and training
        let forward_pipeline = VulkanBatchedPipeline::new(
            &context.ash_device,
            &context.ash_instance,
            context.physical_device,
            4, // input, gamma, beta, output
            "layernorm_forward.comp",
            16, // 2 u32s + 1 float + padding = 16 bytes
        )?;

        let adam_pipeline = VulkanBatchedPipeline::new(
            &context.ash_device,
            &context.ash_instance,
            context.physical_device,
            8, // gamma, beta, gamma_grad, beta_grad, gamma_m, gamma_v, beta_m, beta_v
            "layernorm_adam_update.comp",
            32, // 1 u32 + 4 floats + padding = 32 bytes
        )?;

        let instance = &context.ash_instance;
        let physical_device = context.physical_device;
        let device = &context.ash_device;

        // Initialize gamma to 1.0, beta to 0.0, Adam state to zero
        let gamma_init = vec![1.0f32; feature_dim];
        let beta_init = vec![0.0f32; feature_dim];
        let zeros = vec![0.0f32; feature_dim];

        let gamma = forward_pipeline.create_buffer(
            instance,
            physical_device,
            (feature_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        gamma.upload_data(device, &gamma_init)?;

        let beta = forward_pipeline.create_buffer(
            instance,
            physical_device,
            (feature_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        beta.upload_data(device, &beta_init)?;

        let m_gamma = forward_pipeline.create_buffer(
            instance,
            physical_device,
            (feature_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        m_gamma.upload_data(device, &zeros)?;

        let v_gamma = forward_pipeline.create_buffer(
            instance,
            physical_device,
            (feature_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        v_gamma.upload_data(device, &zeros)?;

        let m_beta = forward_pipeline.create_buffer(
            instance,
            physical_device,
            (feature_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        m_beta.upload_data(device, &zeros)?;

        let v_beta = forward_pipeline.create_buffer(
            instance,
            physical_device,
            (feature_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        v_beta.upload_data(device, &zeros)?;

        Ok(VulkanLayerNorm {
            forward_pipeline,
            adam_pipeline,
            context,
            gamma,
            beta,
            m_gamma,
            v_gamma,
            m_beta,
            v_beta,
            feature_dim,
            eps: 1e-5,
            timestep: 0.0,
        })
    }

    pub fn forward(
        &self,
        input: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let total_size = batch_size * self.feature_dim;
        if input.len() != total_size {
            return Err("Input size doesn't match batch_size * feature_dim".into());
        }

        let physical_device = self.context.physical_device;
        let instance = &self.context.ash_instance;
        let device = &self.context.ash_device;

        // Create temporary buffers
        let input_buffer = self.forward_pipeline.create_buffer(
            instance,
            physical_device,
            std::mem::size_of_val(input) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        input_buffer.upload_data(device, input)?;

        let output_buffer = self.forward_pipeline.create_buffer(
            instance,
            physical_device,
            (total_size * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Set up descriptor set
        let descriptor_set = self.forward_pipeline.create_descriptor_set()?;
        self.forward_pipeline.update_descriptor_set(
            descriptor_set,
            0,
            input_buffer.buffer,
            0,
            input_buffer.size,
        );
        self.forward_pipeline.update_descriptor_set(
            descriptor_set,
            1,
            self.gamma.buffer,
            0,
            self.gamma.size,
        );
        self.forward_pipeline.update_descriptor_set(
            descriptor_set,
            2,
            self.beta.buffer,
            0,
            self.beta.size,
        );
        self.forward_pipeline.update_descriptor_set(
            descriptor_set,
            3,
            output_buffer.buffer,
            0,
            output_buffer.size,
        );

        // Execute forward pass
        let (cmd_buffer, cmd_pool) = self.context.create_command_buffer()?;
        unsafe {
            self.context.ash_device.begin_command_buffer(
                cmd_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            )?;
        }

        self.forward_pipeline.bind_pipeline(cmd_buffer);
        self.forward_pipeline
            .bind_descriptor_sets(cmd_buffer, descriptor_set);

        // Push constants: feature_dim, batch_size, epsilon
        let constants = [self.feature_dim as u32, batch_size as u32];
        self.forward_pipeline
            .push_constants_batched_u32(cmd_buffer, &constants);

        let group_size = 256;
        let group_count = total_size.div_ceil(group_size);
        self.forward_pipeline
            .dispatch(cmd_buffer, group_count as u32, 1, 1);

        unsafe {
            self.context.ash_device.end_command_buffer(cmd_buffer)?;
        }
        self.context.submit_and_wait(cmd_buffer)?;
        self.context.destroy_command_pool(cmd_pool);
        self.forward_pipeline.free_descriptor_set(descriptor_set)?;

        // Download result
        let mut output = vec![0.0f32; total_size];
        output_buffer.download_data(device, &mut output)?;

        Ok(output)
    }

    /// Get the current gamma parameters from GPU memory
    pub fn get_gamma(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut gamma = vec![0.0f32; self.feature_dim];
        self.gamma
            .download_data(&self.context.ash_device, &mut gamma)?;
        Ok(gamma)
    }

    /// Get the current beta parameters from GPU memory
    pub fn get_beta(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut beta = vec![0.0f32; self.feature_dim];
        self.beta
            .download_data(&self.context.ash_device, &mut beta)?;
        Ok(beta)
    }

    pub fn train_batch_adam(
        &mut self,
        input: &[f32],
        target: &[f32],
        opt: &OptimizerParams,
        batch: &BatchContext,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let total_size = batch.batch_size * self.feature_dim;
        if input.len() != total_size || target.len() != total_size {
            return Err("Input/target size doesn't match batch_size * feature_dim".into());
        }
        let batch_size = batch.batch_size;
        let learning_rate = opt.learning_rate;
        let beta1 = opt.beta1;
        let beta2 = opt.beta2;
        let epsilon = opt.epsilon;

        let physical_device = self.context.physical_device;
        let instance = &self.context.ash_instance;
        let device = &self.context.ash_device;

        // Create temporary buffers
        let input_buffer = self.forward_pipeline.create_buffer(
            instance,
            physical_device,
            std::mem::size_of_val(input) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        input_buffer.upload_data(device, input)?;

        let target_buffer = self.forward_pipeline.create_buffer(
            instance,
            physical_device,
            std::mem::size_of_val(target) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        target_buffer.upload_data(device, target)?;

        let output_buffer = self.forward_pipeline.create_buffer(
            instance,
            physical_device,
            (total_size * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

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
            self.gamma.buffer,
            0,
            self.gamma.size,
        );
        self.forward_pipeline.update_descriptor_set(
            forward_descriptor_set,
            2,
            self.beta.buffer,
            0,
            self.beta.size,
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

        let forward_constants = [self.feature_dim as u32, batch_size as u32];
        self.forward_pipeline
            .push_constants_batched_u32(forward_cmd_buffer, &forward_constants);

        let group_size = 256;
        let group_count = total_size.div_ceil(group_size);
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

        let mut loss = 0.0;
        let mut output_grad = vec![0.0f32; target.len()];

        for i in 0..target.len() {
            let diff = output[i] - target[i];
            loss += diff * diff;
            output_grad[i] = 2.0 * diff; // MSE gradient
        }
        loss /= target.len() as f32;

        // Step 3: Compute gradients on CPU (simplified - in practice this would be done on GPU)
        let mut gamma_grads = vec![0.0f32; self.feature_dim];
        let mut beta_grads = vec![0.0f32; self.feature_dim];

        // Download current parameters for gradient computation
        let mut current_gamma = vec![0.0f32; self.feature_dim];
        let mut current_beta = vec![0.0f32; self.feature_dim];
        self.gamma.download_data(device, &mut current_gamma)?;
        self.beta.download_data(device, &mut current_beta)?;

        // Compute gradients for the entire batch
        for batch_idx in 0..batch_size {
            let input_offset = batch_idx * self.feature_dim;
            let output_offset = batch_idx * self.feature_dim;

            // Calculate mean and variance for this batch element
            let mut sum = 0.0;
            for i in 0..self.feature_dim {
                sum += input[input_offset + i];
            }
            let mean = sum / self.feature_dim as f32;

            let mut var_sum = 0.0;
            for i in 0..self.feature_dim {
                let diff = input[input_offset + i] - mean;
                var_sum += diff * diff;
            }
            let variance = var_sum / self.feature_dim as f32;
            let std = (variance + self.eps).sqrt();

            // Compute gradients
            for i in 0..self.feature_dim {
                let normalized = (input[input_offset + i] - mean) / std;
                gamma_grads[i] += output_grad[output_offset + i] * normalized;
                beta_grads[i] += output_grad[output_offset + i];
            }
        }

        // Average gradients over batch size
        for grad in &mut gamma_grads {
            *grad /= batch_size as f32;
        }
        for grad in &mut beta_grads {
            *grad /= batch_size as f32;
        }

        // Create gradient buffers
        let gamma_grad_buffer = self.forward_pipeline.create_buffer(
            instance,
            physical_device,
            (self.feature_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        gamma_grad_buffer.upload_data(device, &gamma_grads)?;

        let beta_grad_buffer = self.forward_pipeline.create_buffer(
            instance,
            physical_device,
            (self.feature_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        beta_grad_buffer.upload_data(device, &beta_grads)?;

        // Step 4: Adam update
        self.timestep += 1.0;
        let adam_descriptor_set = self.adam_pipeline.create_descriptor_set()?;
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            0,
            self.gamma.buffer,
            0,
            self.gamma.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            1,
            self.beta.buffer,
            0,
            self.beta.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            2,
            gamma_grad_buffer.buffer,
            0,
            gamma_grad_buffer.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            3,
            beta_grad_buffer.buffer,
            0,
            beta_grad_buffer.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            4,
            self.m_gamma.buffer,
            0,
            self.m_gamma.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            5,
            self.v_gamma.buffer,
            0,
            self.v_gamma.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            6,
            self.m_beta.buffer,
            0,
            self.m_beta.size,
        );
        self.adam_pipeline.update_descriptor_set(
            adam_descriptor_set,
            7,
            self.v_beta.buffer,
            0,
            self.v_beta.size,
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

        let adam_constants = [
            self.feature_dim as f32,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            self.timestep,
            0.0,
            0.0,
        ];
        self.adam_pipeline
            .push_constants_batched(adam_cmd_buffer, &adam_constants);

        let group_count = self.feature_dim.div_ceil(group_size);
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
        let batch_size = batch.batch_size;
        let learning_rate = opt.learning_rate;
        let beta1 = opt.beta1;
        let beta2 = opt.beta2;
        let epsilon = opt.epsilon;

        // Forward pass to get normalized output
        let normalized = self.forward(input, batch_size)?;

        // Compute gradients for gamma and beta
        let mut gamma_grad = vec![0.0f32; self.feature_dim];
        let mut beta_grad = vec![0.0f32; self.feature_dim];

        for batch_idx in 0..batch_size {
            let offset = batch_idx * self.feature_dim;
            for i in 0..self.feature_dim {
                gamma_grad[i] += grad_output[offset + i] * normalized[offset + i];
                beta_grad[i] += grad_output[offset + i];
            }
        }

        // Average gradients over batch
        for grad in &mut gamma_grad {
            *grad /= batch_size as f32;
        }
        for grad in &mut beta_grad {
            *grad /= batch_size as f32;
        }

        // Update parameters using Adam
        self.timestep += 1.0;
        let physical_device = self.context.physical_device;
        let instance = &self.context.ash_instance;
        let device = &self.context.ash_device;

        // Create gradient buffers
        let gamma_grad_buffer = self.forward_pipeline.create_buffer(
            instance,
            physical_device,
            (self.feature_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        gamma_grad_buffer.upload_data(device, &gamma_grad)?;

        let beta_grad_buffer = self.forward_pipeline.create_buffer(
            instance,
            physical_device,
            (self.feature_dim * std::mem::size_of::<f32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        beta_grad_buffer.upload_data(device, &beta_grad)?;

        // Adam update
        let descriptor_set = self.adam_pipeline.create_descriptor_set()?;
        self.adam_pipeline.update_descriptor_set(
            descriptor_set,
            0,
            self.gamma.buffer,
            0,
            self.gamma.size,
        );
        self.adam_pipeline.update_descriptor_set(
            descriptor_set,
            1,
            self.beta.buffer,
            0,
            self.beta.size,
        );
        self.adam_pipeline.update_descriptor_set(
            descriptor_set,
            2,
            gamma_grad_buffer.buffer,
            0,
            gamma_grad_buffer.size,
        );
        self.adam_pipeline.update_descriptor_set(
            descriptor_set,
            3,
            beta_grad_buffer.buffer,
            0,
            beta_grad_buffer.size,
        );
        self.adam_pipeline.update_descriptor_set(
            descriptor_set,
            4,
            self.m_gamma.buffer,
            0,
            self.m_gamma.size,
        );
        self.adam_pipeline.update_descriptor_set(
            descriptor_set,
            5,
            self.v_gamma.buffer,
            0,
            self.v_gamma.size,
        );
        self.adam_pipeline.update_descriptor_set(
            descriptor_set,
            6,
            self.m_beta.buffer,
            0,
            self.m_beta.size,
        );
        self.adam_pipeline.update_descriptor_set(
            descriptor_set,
            7,
            self.v_beta.buffer,
            0,
            self.v_beta.size,
        );

        let (cmd_buffer, cmd_pool) = self.context.create_command_buffer()?;
        unsafe {
            self.context.ash_device.begin_command_buffer(
                cmd_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            )?;
        }

        self.adam_pipeline.bind_pipeline(cmd_buffer);
        self.adam_pipeline
            .bind_descriptor_sets(cmd_buffer, descriptor_set);

        // Push constants: feature_dim, learning_rate, beta1, beta2, epsilon, t
        let constants = [
            self.feature_dim as f32,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            self.timestep,
            0.0,
            0.0,
        ];
        self.adam_pipeline
            .push_constants_batched(cmd_buffer, &constants);

        let group_size = 256;
        let group_count = self.feature_dim.div_ceil(group_size);
        self.adam_pipeline
            .dispatch(cmd_buffer, group_count as u32, 1, 1);

        unsafe {
            self.context.ash_device.end_command_buffer(cmd_buffer)?;
        }
        self.context.submit_and_wait(cmd_buffer)?;
        self.context.destroy_command_pool(cmd_pool);
        self.adam_pipeline.free_descriptor_set(descriptor_set)?;

        // Compute input gradients
        let mut grad_input = vec![0.0f32; input.len()];

        // Download current gamma and beta
        let mut current_gamma = vec![0.0f32; self.feature_dim];
        let mut current_beta = vec![0.0f32; self.feature_dim];
        self.gamma.download_data(device, &mut current_gamma)?;
        self.beta.download_data(device, &mut current_beta)?;

        // Compute input gradients for each sample in the batch
        for batch_idx in 0..batch_size {
            let offset = batch_idx * self.feature_dim;

            // Compute mean and variance for this sample
            let mut mean = 0.0;
            let mut var = 0.0;
            for i in 0..self.feature_dim {
                mean += input[offset + i];
            }
            mean /= self.feature_dim as f32;

            for i in 0..self.feature_dim {
                let diff = input[offset + i] - mean;
                var += diff * diff;
            }
            var /= self.feature_dim as f32;
            let std_dev = (var + self.eps).sqrt();

            // Compute gradients
            for i in 0..self.feature_dim {
                // Gradient with respect to input
                let grad_x = grad_output[offset + i] * current_gamma[i] / std_dev;
                grad_input[offset + i] = grad_x;
            }
        }

        Ok(grad_input)
    }
}
