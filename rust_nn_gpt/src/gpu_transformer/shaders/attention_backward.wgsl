// MultiHeadAttention Backward Pass Shader (single-head, single-sample, full gradients)
// input_buf: [Q (dim) | K (dim) | V (dim) | d_output (dim) | w_o (dim*dim)]
// output_buf: [d_wo (dim*dim) | d_qkv (3*dim) | d_input (dim) | d_o_b (dim)]

struct DimUniform { dim: u32, };
@group(0) @binding(0) var<storage, read> input_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;
@group(0) @binding(2) var<uniform> dim_uniform: DimUniform;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let dim = dim_uniform.dim;
    let q_offset = 0u;
    let k_offset = q_offset + dim;
    let v_offset = k_offset + dim;
    let d_output_offset = v_offset + dim;
    let w_o_offset = d_output_offset + dim;
    // Output offsets
    let d_wo_offset = 0u;
    let d_qkv_offset = d_wo_offset + dim * dim;
    let d_input_offset = d_qkv_offset + 3u * dim;
    let d_o_b_offset = d_input_offset + dim;
    let d_v_w_offset = d_o_b_offset + dim;
    let d_q_w_offset = d_v_w_offset + dim * dim;
    let d_k_w_offset = d_q_w_offset + dim * dim;
    if (i >= dim) { return; }
    // Unpack Q, K, V
    let q = input_buf[q_offset + i];
    let k = input_buf[k_offset + i];
    let v = input_buf[v_offset + i];
    // d_output
    let d_output = input_buf[d_output_offset + i];
    // --- Forward pass reference ---
    // attn_scores = dot(Q, K) / sqrt(dim)
    var attn_scores: f32 = 0.0;
    for (var j: u32 = 0u; j < dim; j = j + 1u) {
        attn_scores = attn_scores + input_buf[q_offset + j] * input_buf[k_offset + j];
    }
    attn_scores = attn_scores / sqrt(f32(dim));
    // attn_weights = 1.0 (single sample)
    let attn_weights = 1.0;
    // attn_out = V (single sample)
    let attn_out = v;
    // --- Backward pass ---
    // d_attn_out = d_output * W_O^T
    var d_attn_out: f32 = 0.0;
    for (var j: u32 = 0u; j < dim; j = j + 1u) {
        let w_idx = w_o_offset + j * dim + i; // W_O^T
        d_attn_out = d_attn_out + input_buf[d_output_offset + j] * input_buf[w_idx];
    }
    // dV = d_attn_out (since attn_weights = 1.0)
    output_buf[d_qkv_offset + 2u * dim + i] = d_attn_out;
    // dQ, dK: for single-token, softmax grad is zero, so these are zero
    output_buf[d_qkv_offset + i] = 0.0;
    output_buf[d_qkv_offset + dim + i] = 0.0;
    // d_input = dV (since only V path is nonzero)
    output_buf[d_input_offset + i] = d_attn_out;
    // d_WO = outer(attn_out, d_output)
    for (var j: u32 = 0u; j < dim; j = j + 1u) {
        let idx = d_wo_offset + i * dim + j;
        output_buf[idx] = attn_out * input_buf[d_output_offset + j];
    }
    // Compute true output bias gradient
    output_buf[d_o_b_offset + i] = d_output;
    // Compute d_v_w, d_q_w, d_k_w
    for (var j: u32 = 0u; j < dim; j = j + 1u) {
        let v_idx = d_v_w_offset + i * dim + j;
        let q_idx = d_q_w_offset + i * dim + j;
        let k_idx = d_k_w_offset + i * dim + j;
        output_buf[v_idx] = d_attn_out * input_buf[j]; // d_v = d_attn_out
        output_buf[q_idx] = 0.0; // d_q = 0.0 for single-token
        output_buf[k_idx] = 0.0; // d_k = 0.0 for single-token
    }
} 