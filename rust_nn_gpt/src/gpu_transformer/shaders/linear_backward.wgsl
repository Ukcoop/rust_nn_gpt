// Linear Layer Backward Pass Shader
// Input: x [in_dim], d_output [out_dim], W [out_dim][in_dim]
// Output: d_W [out_dim][in_dim], d_b [out_dim], d_input [in_dim]

struct Input { x: array<f32> }
struct DOutput { d: array<f32> }
struct Weights { w: array<f32> } // row-major: out_dim * in_dim
struct D_W { dw: array<f32> }
struct D_b { db: array<f32> }
struct DInput { dx: array<f32> }

@group(0) @binding(0) var<storage, read> input: Input;
@group(0) @binding(1) var<storage, read> d_output: DOutput;
@group(0) @binding(2) var<storage, read> weights: Weights;
@group(0) @binding(3) var<storage, read_write> d_w: D_W;
@group(0) @binding(4) var<storage, read_write> d_b: D_b;
@group(0) @binding(5) var<storage, read_write> d_input: DInput;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let in_dim = arrayLength(&input.x);
    let out_dim = arrayLength(&d_output.d);
    // d_W and d_b
    if (i < out_dim) {
        for (var j: u32 = 0u; j < in_dim; j = j + 1u) {
            let idx = i * in_dim + j;
            d_w.dw[idx] = d_output.d[i] * input.x[j];
        }
        d_b.db[i] = d_output.d[i];
    }
    // d_input
    if (i < in_dim) {
        var sum: f32 = 0.0;
        for (var j: u32 = 0u; j < out_dim; j = j + 1u) {
            let w_idx = j * in_dim + i;
            sum = sum + d_output.d[j] * weights.w[w_idx];
        }
        d_input.dx[i] = sum;
    }
} 