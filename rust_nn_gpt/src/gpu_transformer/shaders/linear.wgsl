// WGSL: Simple linear layer (y = x * W^T + b)
// Input: x [in_dim], W [out_dim][in_dim], b [out_dim]
// Output: y [out_dim]

struct LinearInput {
    x: array<f32>,
};

struct LinearWeights {
    w: array<f32>, // row-major: out_dim * in_dim
};

struct LinearBias {
    b: array<f32>,
};

struct LinearOutput {
    y: array<f32>,
};

@group(0) @binding(0) var<storage, read> input: LinearInput;
@group(0) @binding(1) var<storage, read> weights: LinearWeights;
@group(0) @binding(2) var<storage, read> bias: LinearBias;
@group(0) @binding(3) var<storage, read_write> output: LinearOutput;

@group(0) @binding(4) var<uniform> in_dim: u32;
@group(0) @binding(5) var<uniform> out_dim: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= out_dim) { return; }
    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < in_dim; j = j + 1u) {
        let w_idx = i * in_dim + j;
        sum = sum + input.x[j] * weights.w[w_idx];
    }
    output.y[i] = sum + bias.b[i];
} 