// 2-layer MLP with ReLU
// Input: x [in_dim]
// W1: [hidden_dim][in_dim], b1: [hidden_dim]
// W2: [out_dim][hidden_dim], b2: [out_dim]
// Output: y [out_dim]

struct Input {
    x: array<f32>,
};
struct W1 {
    w: array<f32>, // row-major: hidden_dim * in_dim
};
struct B1 {
    b: array<f32>,
};
struct W2 {
    w: array<f32>, // row-major: out_dim * hidden_dim
};
struct B2 {
    b: array<f32>,
};
struct Output {
    y: array<f32>,
};

@group(0) @binding(0) var<storage, read> input: Input;
@group(0) @binding(1) var<storage, read> w1: W1;
@group(0) @binding(2) var<storage, read> b1: B1;
@group(0) @binding(3) var<storage, read> w2: W2;
@group(0) @binding(4) var<storage, read> b2: B2;
@group(0) @binding(5) var<storage, read_write> output: Output;

// TODO: pass in in_dim, hidden_dim, out_dim as uniforms if needed
// For now, assume these are known at compile time or use arrayLength

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let out_dim = arrayLength(&output.y);
    let hidden_dim = arrayLength(&b1.b);
    let in_dim = arrayLength(&input.x);
    if (i >= out_dim) { return; }
    // Compute hidden = max(0, x * W1^T + b1)
    var hidden: array<f32, 256>;
    for (var j: u32 = 0u; j < hidden_dim; j = j + 1u) {
        var sum: f32 = 0.0;
        for (var k: u32 = 0u; k < in_dim; k = k + 1u) {
            let w_idx = j * in_dim + k;
            sum = sum + input.x[k] * w1.w[w_idx];
        }
        // GELU activation
        let x = sum + b1.b[j];
        let gelu = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * pow(x, 3.0))));
        hidden[j] = gelu;
    }
    // Compute output = hidden * W2^T + b2
    var sum2: f32 = 0.0;
    for (var j: u32 = 0u; j < hidden_dim; j = j + 1u) {
        let w2_idx = i * hidden_dim + j;
        sum2 = sum2 + hidden[j] * w2.w[w2_idx];
    }
    output.y[i] = sum2 + b2.b[i];
} 