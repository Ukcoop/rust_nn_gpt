use crate::gpu_transformer::shader_loader;
use ash::Device;
use ash::Instance;
use ash::vk;
use std::ptr;
use std::sync::Arc;

pub struct VulkanComputePipeline {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
}

pub struct VulkanBatchedPipeline {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
}

pub struct VulkanBuffer {
    pub device: Arc<Device>,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub mapped_ptr: Option<*mut std::ffi::c_void>,
}

impl VulkanBuffer {
    pub fn new(
        device: &Arc<Device>,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo {
                    s_type: vk::StructureType::BUFFER_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::BufferCreateFlags::empty(),
                    size,
                    usage,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    queue_family_index_count: 0,
                    p_queue_family_indices: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                None,
            )?
        };

        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let memory_type_index = unsafe {
            instance
                .get_physical_device_memory_properties(physical_device)
                .memory_types
                .iter()
                .enumerate()
                .position(|(i, memory_type)| {
                    (memory_requirements.memory_type_bits & (1 << i)) != 0
                        && memory_type.property_flags.contains(memory_properties)
                })
                .ok_or("No suitable memory type found")?
        };

        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                    p_next: std::ptr::null(),
                    allocation_size: memory_requirements.size,
                    memory_type_index: memory_type_index as u32,
                    _marker: std::marker::PhantomData,
                },
                None,
            )?
        };

        unsafe {
            device.bind_buffer_memory(buffer, memory, 0)?;
        }

        let mapped_ptr = if memory_properties.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            Some(unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())? })
        } else {
            None
        };

        Ok(VulkanBuffer {
            device: device.clone(),
            buffer,
            memory,
            size,
            mapped_ptr,
        })
    }

    pub fn upload_data<T>(
        &self,
        _device: &Arc<Device>,
        data: &[T],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mapped_ptr) = self.mapped_ptr {
            let data_size = std::mem::size_of_val(data);
            if data_size as vk::DeviceSize <= self.size {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const std::ffi::c_void,
                        mapped_ptr,
                        data_size,
                    );
                }
                Ok(())
            } else {
                Err("Data too large for buffer".into())
            }
        } else {
            Err("Buffer not host visible".into())
        }
    }

    pub fn download_data<T>(
        &self,
        _device: &Arc<Device>,
        data: &mut [T],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mapped_ptr) = self.mapped_ptr {
            let data_size = std::mem::size_of_val(data);
            if data_size as vk::DeviceSize <= self.size {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        mapped_ptr,
                        data.as_mut_ptr() as *mut std::ffi::c_void,
                        data_size,
                    );
                }
                Ok(())
            } else {
                Err("Data too large for buffer".into())
            }
        } else {
            Err("Buffer not host visible".into())
        }
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            if self.mapped_ptr.is_some() {
                self.device.unmap_memory(self.memory);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl VulkanComputePipeline {
    pub fn new(
        device: &Arc<Device>,
        _instance: &Instance,
        _physical_device: vk::PhysicalDevice,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create descriptor set layout
        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: ptr::null(),
                _marker: std::marker::PhantomData,
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: ptr::null(),
                _marker: std::marker::PhantomData,
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: ptr::null(),
                _marker: std::marker::PhantomData,
            },
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: ptr::null(),
                _marker: std::marker::PhantomData,
            },
        ];

        let layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            binding_count: 4,
            p_bindings: bindings.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_create_info, None)? };

        // Create pipeline layout with push constants
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 8, // 2 uints: input_size, output_size
        };

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            _marker: std::marker::PhantomData,
        };

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None)? };

        // Create descriptor pool
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 400, // 4 bindings * 100 sets (much larger)
        }];

        let pool_create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            max_sets: 100, // Increased from 10 to 100
            pool_size_count: 1,
            p_pool_sizes: pool_sizes.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_create_info, None)? };

        // Create compute shader using the compiled SPIR-V
        let shader_module = Self::create_shader_module(device)?;

        // Create compute pipeline
        let shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: c"main".as_ptr(),
            p_specialization_info: ptr::null(),
            _marker: std::marker::PhantomData,
        };

        let pipeline_create_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage: shader_stage_create_info,
            layout: pipeline_layout,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: 0,
            _marker: std::marker::PhantomData,
        };

        let pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .map_err(|(_, e)| e)?[0]
        };

        // Clean up shader module
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(VulkanComputePipeline {
            device: device.clone(),
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
        })
    }

    fn create_shader_module(
        device: &Arc<Device>,
    ) -> Result<vk::ShaderModule, Box<dyn std::error::Error>> {
        // Load the compiled SPIR-V shader
        let shader_code = shader_loader::load_linear_shader()?;

        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: shader_code.len() * std::mem::size_of::<u32>(),
            p_code: shader_code.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        unsafe { Ok(device.create_shader_module(&shader_module_create_info, None)?) }
    }

    pub fn create_buffer(
        &self,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<VulkanBuffer, Box<dyn std::error::Error>> {
        VulkanBuffer::new(
            &self.device,
            instance,
            physical_device,
            size,
            usage,
            memory_properties,
        )
    }

    pub fn create_descriptor_set(&self) -> Result<vk::DescriptorSet, Box<dyn std::error::Error>> {
        let layout = [self.descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: self.descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: layout.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let descriptor_sets = unsafe { self.device.allocate_descriptor_sets(&allocate_info)? };
        let descriptor_set = descriptor_sets[0];
        Ok(descriptor_set)
    }

    pub fn update_descriptor_set(
        &self,
        descriptor_set: vk::DescriptorSet,
        binding: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) {
        let buffer_info = vk::DescriptorBufferInfo {
            buffer,
            offset,
            range,
        };

        let write_descriptor_set = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: descriptor_set,
            dst_binding: binding,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: &buffer_info,
            p_texel_buffer_view: ptr::null(),
            _marker: std::marker::PhantomData,
        };

        unsafe {
            self.device
                .update_descriptor_sets(&[write_descriptor_set], &[]);
        }
    }

    pub fn dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        unsafe {
            self.device
                .cmd_dispatch(command_buffer, group_count_x, group_count_y, group_count_z);
        }
    }

    pub fn bind_pipeline(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
        }
    }

    pub fn bind_descriptor_sets(
        &self,
        command_buffer: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
    ) {
        // Validate handles before binding
        if self.pipeline_layout == vk::PipelineLayout::null() {
            return;
        }
        if descriptor_set == vk::DescriptorSet::null() {
            return;
        }

        unsafe {
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
        }
    }

    pub fn push_constants(
        &self,
        command_buffer: vk::CommandBuffer,
        input_size: u32,
        output_size: u32,
    ) {
        let constants = [input_size, output_size];
        let bytes = bytemuck::cast_slice(&constants);
        unsafe {
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
        }
    }

    pub fn push_constants_u32(&self, command_buffer: vk::CommandBuffer, constants: &[u32]) {
        let bytes = bytemuck::cast_slice(constants);
        unsafe {
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
        }
    }

    pub fn free_descriptor_set(
        &self,
        descriptor_set: vk::DescriptorSet,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.device.free_descriptor_sets(
                self.descriptor_pool,
                std::slice::from_ref(&descriptor_set),
            )?;
        }
        Ok(())
    }
}

impl VulkanBatchedPipeline {
    pub fn new(
        _device: &Arc<Device>,
        _instance: &Instance,
        _physical_device: vk::PhysicalDevice,
        binding_count: u32,
        shader_name: &str,
        push_constant_size: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create descriptor set layout with variable number of bindings
        let mut bindings = Vec::new();
        for i in 0..binding_count {
            bindings.push(vk::DescriptorSetLayoutBinding {
                binding: i,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: ptr::null(),
                _marker: std::marker::PhantomData,
            });
        }

        let layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            binding_count,
            p_bindings: bindings.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let descriptor_set_layout =
            unsafe { _device.create_descriptor_set_layout(&layout_create_info, None)? };

        // Create pipeline layout with push constants
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: push_constant_size,
        };

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            _marker: std::marker::PhantomData,
        };

        let pipeline_layout =
            unsafe { _device.create_pipeline_layout(&pipeline_layout_create_info, None)? };

        // Create descriptor pool
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: binding_count * 100, // binding_count * 100 sets
        }];

        let pool_create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            max_sets: 100,
            pool_size_count: 1,
            p_pool_sizes: pool_sizes.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let descriptor_pool = unsafe { _device.create_descriptor_pool(&pool_create_info, None)? };

        // Create compute shader using the specified shader
        let shader_module = Self::create_shader_module(_device, shader_name)?;

        // Create compute pipeline
        let shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: c"main".as_ptr(),
            p_specialization_info: ptr::null(),
            _marker: std::marker::PhantomData,
        };

        let pipeline_create_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage: shader_stage_create_info,
            layout: pipeline_layout,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: 0,
            _marker: std::marker::PhantomData,
        };

        let pipeline = unsafe {
            _device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .map_err(|(_, e)| e)?[0]
        };

        // Clean up shader module
        unsafe { _device.destroy_shader_module(shader_module, None) };

        Ok(VulkanBatchedPipeline {
            device: _device.clone(),
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
        })
    }

    fn create_shader_module(
        device: &Arc<Device>,
        shader_name: &str,
    ) -> Result<vk::ShaderModule, Box<dyn std::error::Error>> {
        // Load the compiled SPIR-V shader
        let shader_code = shader_loader::load_shader(shader_name)?;

        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: shader_code.len() * std::mem::size_of::<u32>(),
            p_code: shader_code.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        unsafe { Ok(device.create_shader_module(&shader_module_create_info, None)?) }
    }

    pub fn create_buffer(
        &self,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<VulkanBuffer, Box<dyn std::error::Error>> {
        VulkanBuffer::new(
            &self.device,
            instance,
            physical_device,
            size,
            usage,
            memory_properties,
        )
    }

    pub fn create_descriptor_set(&self) -> Result<vk::DescriptorSet, Box<dyn std::error::Error>> {
        let layout = [self.descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: self.descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: layout.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let descriptor_sets = unsafe { self.device.allocate_descriptor_sets(&allocate_info)? };
        let descriptor_set = descriptor_sets[0];
        Ok(descriptor_set)
    }

    pub fn update_descriptor_set(
        &self,
        descriptor_set: vk::DescriptorSet,
        binding: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) {
        let buffer_info = vk::DescriptorBufferInfo {
            buffer,
            offset,
            range,
        };

        let write_descriptor_set = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: descriptor_set,
            dst_binding: binding,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: &buffer_info,
            p_texel_buffer_view: ptr::null(),
            _marker: std::marker::PhantomData,
        };

        unsafe {
            self.device
                .update_descriptor_sets(&[write_descriptor_set], &[]);
        }
    }

    pub fn dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        unsafe {
            self.device
                .cmd_dispatch(command_buffer, group_count_x, group_count_y, group_count_z);
        }
    }

    pub fn bind_pipeline(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
        }
    }

    pub fn bind_descriptor_sets(
        &self,
        command_buffer: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
    ) {
        // Validate handles before binding
        if self.pipeline_layout == vk::PipelineLayout::null() {
            return;
        }
        if descriptor_set == vk::DescriptorSet::null() {
            return;
        }

        unsafe {
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
        }
    }

    pub fn push_constants_batched(&self, command_buffer: vk::CommandBuffer, constants: &[f32]) {
        let bytes = bytemuck::cast_slice(constants);
        unsafe {
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
        }
    }
    pub fn push_constants_batched_u32(&self, command_buffer: vk::CommandBuffer, constants: &[u32]) {
        let bytes = bytemuck::cast_slice(constants);
        unsafe {
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
        }
    }

    pub fn free_descriptor_set(
        &self,
        descriptor_set: vk::DescriptorSet,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.device.free_descriptor_sets(
                self.descriptor_pool,
                std::slice::from_ref(&descriptor_set),
            )?;
        }
        Ok(())
    }
}

impl Drop for VulkanBatchedPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
