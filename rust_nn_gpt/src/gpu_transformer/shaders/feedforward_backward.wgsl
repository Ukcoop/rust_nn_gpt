// FeedForward Backward Pass Shader (GELU, packed buffers)
// Binding 0: packed_inputs = input | pre_hidden | hidden | d_output
// Binding 1: packed_weights = w2 | w1
// Binding 2: packed_grads = d_w1 | d_b1 | d_w2 | d_b2 | d_input

struct PackedInputs { data: array<f32>, } // input | pre_hidden | hidden | d_output
struct PackedWeights { data: array<f32>, } // w2 | w1
struct PackedGrads { data: array<f32>, } // d_w1 | d_b1 | d_w2 | d_b2 | d_input

struct FFWDims {
    in_dim: u32,
    hidden_dim: u32,
    out_dim: u32,
};
@group(0) @binding(3) var<uniform> dims: FFWDims;

@group(0) @binding(0) var<storage, read> packed_inputs: PackedInputs;
@group(0) @binding(1) var<storage, read> packed_weights: PackedWeights;
@group(0) @binding(2) var<storage, read_write> packed_grads: PackedGrads;

// Helper: GELU derivative
fn gelu_deriv(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608;
    let x3 = x * x * x;
    let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
    let tanh_inner = tanh(inner);
    let left = 0.5 * (1.0 + tanh_inner);
    let right = 0.5 * x * (1.0 - tanh_inner * tanh_inner) * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
    return left + right;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let in_dim = dims.in_dim;
    let hidden_dim = dims.hidden_dim;
    let out_dim = dims.out_dim;
    // Offsets for packed buffers
    let input_offset = 0u;
    let pre_hidden_offset = input_offset + in_dim;
    let hidden_offset = pre_hidden_offset + hidden_dim;
    let d_output_offset = hidden_offset + hidden_dim;
    let w2_offset = 0u;
    let w1_offset = w2_offset + out_dim * hidden_dim;
    let d_w1_offset = 0u;
    let d_b1_offset = d_w1_offset + hidden_dim * in_dim;
    let d_w2_offset = d_b1_offset + hidden_dim;
    let d_b2_offset = d_w2_offset + out_dim * hidden_dim;
    let d_input_offset = d_b2_offset + out_dim;

    // d_W2, d_b2, d_hidden
    if (i < out_dim) {
        for (var j: u32 = 0u; j < hidden_dim; j = j + 1u) {
            let idx = i * hidden_dim + j;
            let h = packed_inputs.data[hidden_offset + j];
            let d_out = packed_inputs.data[d_output_offset + i];
            packed_grads.data[d_w2_offset + idx] = d_out * h;
        }
        packed_grads.data[d_b2_offset + i] = packed_inputs.data[d_output_offset + i];
    }
    // d_hidden (for all hidden units)
    var d_hidden: array<f32, 256>;
    if (i < hidden_dim) {
        var sum: f32 = 0.0;
        for (var k: u32 = 0u; k < out_dim; k = k + 1u) {
            let w2_idx = k * hidden_dim + i;
            let d_out = packed_inputs.data[d_output_offset + k];
            let w2 = packed_weights.data[w2_offset + w2_idx];
            sum = sum + d_out * w2;
        }
        let x = packed_inputs.data[pre_hidden_offset + i];
        d_hidden[i] = sum * gelu_deriv(x);
        // d_W1, d_b1
        for (var j: u32 = 0u; j < in_dim; j = j + 1u) {
            let idx = i * in_dim + j;
            let inp = packed_inputs.data[input_offset + j];
            packed_grads.data[d_w1_offset + idx] = d_hidden[i] * inp;
        }
        packed_grads.data[d_b1_offset + i] = d_hidden[i];
    }
    // d_input
    if (i < in_dim) {
        var sum: f32 = 0.0;
        for (var j: u32 = 0u; j < hidden_dim; j = j + 1u) {
            let w1_idx = j * in_dim + i;
            let d_h = d_hidden[j];
            let w1 = packed_weights.data[w1_offset + w1_idx];
            sum = sum + d_h * w1;
        }
        packed_grads.data[d_input_offset + i] = sum;
    }
} 