use crate::gpu_transformer::compute_pipeline::VulkanBatchedPipeline;
use crate::gpu_transformer::vulkan_context::VulkanContext;
use ash::vk;
use std::error::Error;

pub struct VulkanGELU {
    pipeline: VulkanBatchedPipeline,
    context: VulkanContext,
}

impl VulkanGELU {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let context = VulkanContext::new()?;

        let pipeline = VulkanBatchedPipeline::new(
            &context.ash_device,
            &context.ash_instance,
            context.physical_device,
            2, // input, output
            "gelu.comp",
            8, // 1 u32 = 4 bytes + padding = 8 bytes
        )?;

        Ok(VulkanGELU { pipeline, context })
    }

    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let size = input.len();
        let physical_device = self.context.physical_device;
        let instance = &self.context.ash_instance;
        let device = &self.context.ash_device;

        // Create buffers
        let input_buffer = self.pipeline.create_buffer(
            instance,
            physical_device,
            std::mem::size_of_val(input) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        input_buffer.upload_data(device, input)?;

        let output_buffer = self.pipeline.create_buffer(
            instance,
            physical_device,
            std::mem::size_of_val(input) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Set up descriptor set
        let descriptor_set = self.pipeline.create_descriptor_set()?;
        self.pipeline.update_descriptor_set(
            descriptor_set,
            0,
            input_buffer.buffer,
            0,
            input_buffer.size,
        );
        self.pipeline.update_descriptor_set(
            descriptor_set,
            1,
            output_buffer.buffer,
            0,
            output_buffer.size,
        );

        // Execute
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

        self.pipeline.bind_pipeline(cmd_buffer);
        self.pipeline
            .bind_descriptor_sets(cmd_buffer, descriptor_set);

        let constants = [size as u32];
        self.pipeline
            .push_constants_batched_u32(cmd_buffer, &constants);

        let group_size = 256;
        let group_count = size.div_ceil(group_size);
        self.pipeline.dispatch(cmd_buffer, group_count as u32, 1, 1);

        unsafe {
            self.context.ash_device.end_command_buffer(cmd_buffer)?;
        }
        self.context.submit_and_wait(cmd_buffer)?;
        self.context.destroy_command_pool(cmd_pool);
        self.pipeline.free_descriptor_set(descriptor_set)?;

        // Download result
        let mut output = vec![0.0f32; size];
        output_buffer.download_data(device, &mut output)?;

        Ok(output)
    }
}
