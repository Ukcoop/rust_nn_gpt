pub async fn initialize_wgpu() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::default();
    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
    {
        Ok(adapter) => adapter,
        Err(e) => return Err(format!("Failed to find an appropriate adapter: {e}")),
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(|e| format!("Failed to create device: {e}"))?;
    Ok((device, queue))
}

use wgpu::util::DeviceExt;

pub fn create_buffer_init<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

pub fn create_storage_buffer(
    device: &wgpu::Device,
    label: &str,
    size: usize,
    usage: Option<wgpu::BufferUsages>,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage: usage.unwrap_or(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
        mapped_at_creation: false,
    })
}

pub fn create_uniform_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

pub fn create_staging_buffer(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn create_storage_buffer_with_data<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    extra_usage: Option<wgpu::BufferUsages>,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    if let Some(extra) = extra_usage {
        usage |= extra;
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

pub fn write_to_buffer<T: bytemuck::Pod>(queue: &wgpu::Queue, buffer: &wgpu::Buffer, data: &[T]) {
    queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
}

/// Helper to create a bind group layout from a list of entries.
pub fn create_bind_group_layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

/// Helper to create a bind group from a layout and a list of buffers (as entire bindings).
pub fn create_bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    buffers: &[(&wgpu::Buffer, u32)], // (buffer, binding)
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .map(|(buffer, binding)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// Helper to generate a vector of BindGroupLayoutEntry for buffers, with automatic binding numbers and compute visibility.
pub fn make_bgl_entries(descs: &[BufferDesc]) -> Vec<wgpu::BindGroupLayoutEntry> {
    descs
        .iter()
        .enumerate()
        .map(|(i, desc)| {
            let ty = if desc.is_uniform {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            } else {
                wgpu::BindingType::Buffer {
                    ty: if desc.read_only {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: false }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            };
            wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            }
        })
        .collect()
}

/// Description for a buffer in a bind group layout.
#[derive(Clone, Copy)]
pub struct BufferDesc {
    pub read_only: bool,
    pub is_uniform: bool,
}

/// Read a buffer as a Vec<f32> (blocking, for CPU-side access)
pub fn read_buffer_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Vec<f32> {
    use futures::executor::block_on;
    let size = (len * std::mem::size_of::<f32>()) as u64;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ReadBufferF32 Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ReadBufferF32 Encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
    queue.submit(Some(encoder.finish()));
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        let _ = sender.send(v);
    });
    let _ = device.poll(wgpu::MaintainBase::Wait);
    block_on(receiver.receive());
    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();
    result
}

/// Specification for a buffer to be created for a layer.
pub struct BufferSpec<'a> {
    pub label: &'a str,
    pub size: usize,
    pub usage: Option<wgpu::BufferUsages>,
}

/// Holds the created buffers, bind group, and layout for a layer.
pub struct LayerGpuResources {
    pub buffers: Vec<wgpu::Buffer>,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

/// Create all buffers, bind group layout, and bind group for a layer in one call.
///
/// - `device`: The wgpu device
/// - `buffer_specs`: List of buffer specs (label, size, usage)
/// - `bgl_descs`: List of BufferDesc for the bind group layout (must match buffer_specs order)
///   Returns a struct with all buffers, the bind group layout, and the bind group.
pub fn create_layer_gpu_resources(
    device: &wgpu::Device,
    buffer_specs: &[BufferSpec],
    bgl_descs: &[BufferDesc],
    bind_group_label: &str,
    layout_label: &str,
) -> LayerGpuResources {
    assert_eq!(buffer_specs.len(), bgl_descs.len());
    let buffers: Vec<wgpu::Buffer> = buffer_specs
        .iter()
        .map(|spec| create_storage_buffer(device, spec.label, spec.size, spec.usage))
        .collect();
    let bgl_entries = make_bgl_entries(bgl_descs);
    let bind_group_layout = create_bind_group_layout(device, layout_label, &bgl_entries);
    let bind_group = create_bind_group(
        device,
        bind_group_label,
        &bind_group_layout,
        &buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| (buf, i as u32))
            .collect::<Vec<_>>(),
    );
    LayerGpuResources {
        buffers,
        bind_group_layout,
        bind_group,
    }
}
