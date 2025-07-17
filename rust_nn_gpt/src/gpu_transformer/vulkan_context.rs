use ash::vk;
use ash::{Device, Entry, Instance};
use std::sync::Arc;

#[derive(Clone)]
pub struct VulkanContext {
    pub entry: Arc<Entry>,
    pub physical_device: vk::PhysicalDevice,
    pub compute_queue: vk::Queue,
    pub compute_queue_family: u32,
    pub ash_instance: Arc<Instance>,
    pub ash_device: Arc<Device>,
}

impl VulkanContext {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Create entry point (exactly like minimal example)
        let entry = unsafe { ash::Entry::load()? };

        // Create instance (exactly like minimal example)
        let app_name = std::ffi::CString::new("Rust NN GPT")?;
        let engine_name = std::ffi::CString::new("No Engine")?;

        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::API_VERSION_1_0,
            _marker: std::marker::PhantomData,
        };

        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo {
                    s_type: vk::StructureType::INSTANCE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::InstanceCreateFlags::empty(),
                    p_application_info: &app_info,
                    // Deprecated fields, required by struct
                    enabled_layer_count: 0,
                    pp_enabled_layer_names: std::ptr::null(),
                    enabled_extension_count: 0,
                    pp_enabled_extension_names: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                None,
            )?
        };

        // Get physical device (exactly like minimal example)
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let physical_device = physical_devices
            .first()
            .ok_or("No physical devices found")?;

        // Create device (exactly like minimal example)
        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(*physical_device)
                .iter()
                .enumerate()
                .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or("No compute queue family found")?
                .0 as u32
        };

        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let device_features = vk::PhysicalDeviceFeatures {
            robust_buffer_access: 0,
            full_draw_index_uint32: 0,
            image_cube_array: 0,
            independent_blend: 0,
            geometry_shader: 0,
            tessellation_shader: 0,
            sample_rate_shading: 0,
            dual_src_blend: 0,
            logic_op: 0,
            multi_draw_indirect: 0,
            draw_indirect_first_instance: 0,
            depth_clamp: 0,
            depth_bias_clamp: 0,
            fill_mode_non_solid: 0,
            depth_bounds: 0,
            wide_lines: 0,
            large_points: 0,
            alpha_to_one: 0,
            multi_viewport: 0,
            sampler_anisotropy: 0,
            texture_compression_etc2: 0,
            texture_compression_astc_ldr: 0,
            texture_compression_bc: 0,
            occlusion_query_precise: 0,
            pipeline_statistics_query: 0,
            vertex_pipeline_stores_and_atomics: 0,
            fragment_stores_and_atomics: 0,
            shader_tessellation_and_geometry_point_size: 0,
            shader_image_gather_extended: 0,
            shader_storage_image_extended_formats: 0,
            shader_storage_image_multisample: 0,
            shader_storage_image_read_without_format: 0,
            shader_storage_image_write_without_format: 0,
            shader_uniform_buffer_array_dynamic_indexing: 0,
            shader_sampled_image_array_dynamic_indexing: 0,
            shader_storage_buffer_array_dynamic_indexing: 0,
            shader_storage_image_array_dynamic_indexing: 0,
            shader_clip_distance: 0,
            shader_cull_distance: 0,
            shader_float64: 0,
            shader_int64: 0,
            shader_int16: 0,
            shader_resource_residency: 0,
            shader_resource_min_lod: 0,
            sparse_binding: 0,
            sparse_residency_buffer: 0,
            sparse_residency_image2_d: 0,
            sparse_residency_image3_d: 0,
            sparse_residency2_samples: 0,
            sparse_residency4_samples: 0,
            sparse_residency8_samples: 0,
            sparse_residency16_samples: 0,
            sparse_residency_aliased: 0,
            variable_multisample_rate: 0,
            inherited_queries: 0,
        };

        let device = unsafe {
            instance.create_device(
                *physical_device,
                &vk::DeviceCreateInfo {
                    s_type: vk::StructureType::DEVICE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::DeviceCreateFlags::empty(),
                    queue_create_info_count: 1,
                    p_queue_create_infos: &queue_create_info,
                    // Deprecated fields, required by struct
                    #[allow(deprecated)]
                    enabled_layer_count: 0,
                    #[allow(deprecated)]
                    pp_enabled_layer_names: std::ptr::null(),
                    enabled_extension_count: 0,
                    pp_enabled_extension_names: std::ptr::null(),
                    p_enabled_features: &device_features,
                    _marker: std::marker::PhantomData,
                },
                None,
            )?
        };

        // Get queue
        let compute_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Ok(VulkanContext {
            entry: Arc::new(entry),
            physical_device: *physical_device,
            compute_queue,
            compute_queue_family: queue_family_index,
            ash_instance: Arc::new(instance),
            ash_device: Arc::new(device),
        })
    }

    pub fn create_command_buffer(
        &self,
    ) -> Result<(vk::CommandBuffer, vk::CommandPool), Box<dyn std::error::Error>> {
        // Create command pool on demand (minimal example pattern)
        let command_pool = unsafe {
            self.ash_device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index: self.compute_queue_family,
                    _marker: std::marker::PhantomData,
                },
                None,
            )?
        };

        let command_buffers = unsafe {
            self.ash_device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                    p_next: std::ptr::null(),
                    command_pool,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: 1,
                    _marker: std::marker::PhantomData,
                })?
        };

        let command_buffer = command_buffers
            .first()
            .ok_or("Failed to allocate command buffer")?;

        Ok((*command_buffer, command_pool))
    }

    pub fn destroy_command_pool(&self, command_pool: vk::CommandPool) {
        unsafe {
            self.ash_device.destroy_command_pool(command_pool, None);
        }
    }

    pub fn submit_and_wait(
        &self,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        let fence = unsafe {
            self.ash_device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };

        unsafe {
            self.ash_device
                .queue_submit(self.compute_queue, &[submit_info], fence)?;
            self.ash_device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.ash_device.destroy_fence(fence, None);
        }

        Ok(())
    }
}
