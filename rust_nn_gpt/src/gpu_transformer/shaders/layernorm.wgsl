// Proper LayerNorm WGSL: input, gamma, beta, output
struct Input { x: array<f32>, }
struct Gamma { g: array<f32>, }
struct Beta { b: array<f32>, }
struct Output { y: array<f32>, }

@group(0) @binding(0) var<storage, read> input: Input;
@group(0) @binding(1) var<storage, read> gamma: Gamma;
@group(0) @binding(2) var<storage, read> beta: Beta;
@group(0) @binding(3) var<storage, read_write> output: Output;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim = arrayLength(&input.x);
    let i = global_id.x;
    if (i >= dim) { return; }

    // Compute mean
    var mean: f32 = 0.0;
    for (var j: u32 = 0u; j < dim; j = j + 1u) {
        mean = mean + input.x[j];
    }
    mean = mean / f32(dim);

    // Compute variance
    var variance: f32 = 0.0;
    for (var j: u32 = 0u; j < dim; j = j + 1u) {
        let diff = input.x[j] - mean;
        variance = variance + diff * diff;
    }
    variance = variance / f32(dim);

    let eps: f32 = 1e-5;
    let normed = (input.x[i] - mean) / sqrt(variance + eps);
    output.y[i] = gamma.g[i] * normed + beta.b[i];
} 