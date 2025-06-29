use crate::gpu_transformer::gpu_utils::{update_weights, update_bias, pack_slices, unpack_slices};

// Pack inputs: input | d_output
let packed_inputs = pack_slices(&[input, grad_output]); 

// Unpack output buffer: d_input | d_weights | d_bias
let unpacked = unpack_slices(
    &output_buffer,
    &[self.in_dim, self.in_dim * self.out_dim, self.out_dim]
);
let d_input = &unpacked[0];
let d_weights = &unpacked[1];
let d_bias = &unpacked[2]; 