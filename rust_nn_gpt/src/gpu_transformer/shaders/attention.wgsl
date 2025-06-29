// Single-head self-attention (minimal, 1 token)
// Input: x [in_dim]
// Q, K, V: [in_dim][in_dim], b: [in_dim]
// O: [out_dim][in_dim], b: [out_dim]
// Output: y [out_dim]

struct Input { x: array<f32>, };
struct QW { w: array<f32>, };
struct KW { w: array<f32>, };
struct VW { w: array<f32>, };
struct OW { w: array<f32>, };
struct Output { y: array<f32>, };

@group(0) @binding(0) var<storage, read> input: Input;
@group(0) @binding(1) var<storage, read> q_w: QW;
@group(0) @binding(2) var<storage, read> k_w: KW;
@group(0) @binding(3) var<storage, read> v_w: VW;
@group(0) @binding(4) var<storage, read> o_w: OW;
@group(0) @binding(5) var<storage, read> bias_buf: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: Output;

// For now, assume batch size = 1, sequence length = 1 (no real attention, just projections)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let out_dim = arrayLength(&output.y);
    let in_dim = arrayLength(&input.x);
    if (i >= out_dim) { return; }
    // Project input to Q, K, V
    var q: array<f32, 256>;
    var k: array<f32, 256>;
    var v: array<f32, 256>;
    for (var j: u32 = 0u; j < in_dim; j = j + 1u) {
        var sum_q: f32 = 0.0;
        var sum_k: f32 = 0.0;
        var sum_v: f32 = 0.0;
        for (var k2: u32 = 0u; k2 < in_dim; k2 = k2 + 1u) {
            sum_q = sum_q + input.x[k2] * q_w.w[j * in_dim + k2];
            sum_k = sum_k + input.x[k2] * k_w.w[j * in_dim + k2];
            sum_v = sum_v + input.x[k2] * v_w.w[j * in_dim + k2];
        }
        let q_b_idx = j;
        let k_b_idx = in_dim + j;
        let v_b_idx = 2u * in_dim + j;
        q[j] = sum_q + bias_buf[q_b_idx];
        k[j] = sum_k + bias_buf[k_b_idx];
        v[j] = sum_v + bias_buf[v_b_idx];
    }
    // Attention score (dot product, softmax is 1.0 for single token)
    var attn_score: f32 = 0.0;
    for (var j: u32 = 0u; j < in_dim; j = j + 1u) {
        attn_score = attn_score + q[j] * k[j];
    }
    // Normally: attn = softmax(score), but with 1 token, attn = 1.0
    // Weighted sum
    var attn_out: array<f32, 256>;
    for (var j: u32 = 0u; j < in_dim; j = j + 1u) {
        attn_out[j] = v[j];
    }
    // Output projection
    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < in_dim; j = j + 1u) {
        sum = sum + attn_out[j] * o_w.w[i * in_dim + j];
    }
    let o_b_idx = 3u * in_dim + i;
    output.y[i] = sum + bias_buf[o_b_idx];
} 