//! Shared GPU neural network utilities for parameter updates.

/// Update a weights vector with gradients and learning rate.
pub fn update_weights(weights: &mut [f32], grad: &[f32], lr: f32) {
    for (w, g) in weights.iter_mut().zip(grad) {
        *w -= lr * g;
    }
}

/// Update a bias vector with gradients and learning rate.
pub fn update_bias(bias: &mut [f32], grad: &[f32], lr: f32) {
    for (b, g) in bias.iter_mut().zip(grad) {
        *b -= lr * g;
    }
}

/// Pack multiple slices into a single Vec<f32> for GPU buffer operations.
pub fn pack_slices(slices: &[&[f32]]) -> Vec<f32> {
    let total_len: usize = slices.iter().map(|s| s.len()).sum();
    let mut packed = Vec::with_capacity(total_len);
    for s in slices {
        packed.extend_from_slice(s);
    }
    packed
}

/// Unpack a flat slice into multiple Vec<f32> according to the provided lengths.
pub fn unpack_slices(flat: &[f32], lengths: &[usize]) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(lengths.len());
    let mut offset = 0;
    for &len in lengths {
        result.push(flat[offset..offset + len].to_vec());
        offset += len;
    }
    result
}

/// Write a slice of f32 to a wgpu buffer at offset 0.
pub fn write_f32_slice(queue: &wgpu::Queue, buffer: &wgpu::Buffer, data: &[f32]) {
    let cast_slice: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    queue.write_buffer(buffer, 0, cast_slice);
}
