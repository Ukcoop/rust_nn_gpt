// LayerNorm Backward Pass Shader
// Input: x [dim], mean [1], var [1], gamma [dim], d_output [dim]
// Output: d_gamma [dim], d_beta [dim], d_input [dim]

struct Input { x: array<f32> }
struct Mean { mean: f32 }
struct Variance { variance: f32 }
struct Gamma { gamma: array<f32> }
struct DOutput { d: array<f32> }
struct D_Gamma { dgamma: array<f32> }
struct D_Beta { dbeta: array<f32> }
struct DInput { dx: array<f32> }

@group(0) @binding(0) var<storage, read> input: Input;
@group(0) @binding(1) var<storage, read> mean: Mean;
@group(0) @binding(2) var<storage, read> variance: Variance;
@group(0) @binding(3) var<storage, read> gamma: Gamma;
@group(0) @binding(4) var<storage, read> d_output: DOutput;
@group(0) @binding(5) var<storage, read_write> d_gamma: D_Gamma;
@group(0) @binding(6) var<storage, read_write> d_beta: D_Beta;
@group(0) @binding(7) var<storage, read_write> d_input: DInput;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let dim = arrayLength(&input.x);
    if (i >= dim) { return; }
    let eps = 1e-6;
    let mean_val = mean.mean;
    let var_val = variance.variance;
    let stdev = sqrt(var_val + eps);
    let x_hat = (input.x[i] - mean_val) / stdev;
    // d_gamma and d_beta
    d_gamma.dgamma[i] = d_output.d[i] * x_hat;
    d_beta.dbeta[i] = d_output.d[i];
    // d_input (see LayerNorm backward formula)
    // First, sum d_output * gamma and d_output * gamma * x_hat
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    for (var j: u32 = 0u; j < dim; j = j + 1u) {
        let x_hat_j = (input.x[j] - mean_val) / stdev;
        sum1 = sum1 + d_output.d[j] * gamma.gamma[j];
        sum2 = sum2 + d_output.d[j] * gamma.gamma[j] * x_hat_j;
    }
    let dx = (1.0 / stdev) * gamma.gamma[i] * (d_output.d[i]
        - (1.0 / f32(dim)) * sum1
        - (x_hat / f32(dim)) * sum2);
    d_input.dx[i] = dx;
} 