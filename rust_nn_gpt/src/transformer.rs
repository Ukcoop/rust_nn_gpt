use crate::cpu_transformer::CpuTransformer;

pub struct TransformerConfig {
    pub input_dim: usize,
    pub model_dim: usize,
    pub output_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    // Add more config options as needed
}

pub struct Transformer {
    cpu: CpuTransformer,
}

impl Transformer {
    pub fn new(config: TransformerConfig) -> Self {
        Self {
            cpu: CpuTransformer::new(&config),
        }
    }

    pub fn train(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
    ) -> Result<f32, Box<dyn std::error::Error>> {
        self.cpu.train_batch(inputs, targets)
    }

    pub fn predict(&mut self, input: &[f32]) -> Vec<f32> {
        self.cpu.forward(input)
    }
}
