use crate::cpu_transformer::CpuTransformer;
use crate::transformer_weights::TransformerWeights;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct TransformerConfig {
    pub input_dim: usize,
    pub model_dim: usize,
    pub output_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    // Add more config options as needed
}

pub struct Transformer {
    pub cpu: CpuTransformer,
    pub best_loss: Option<f32>,
    pub config: TransformerConfig,
}

#[derive(Serialize, Deserialize)]
struct TransformerSaveData {
    pub config: TransformerConfig,
    pub weights: TransformerWeights,
    pub best_loss: Option<f32>,
}

impl Transformer {
    pub fn new(config: TransformerConfig) -> Self {
        Self {
            cpu: CpuTransformer::new(&config),
            best_loss: None,
            config,
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

    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        let save_data = TransformerSaveData {
            config: self.config.clone(),
            weights: TransformerWeights::from(&self.cpu),
            best_loss: self.best_loss,
        };
        serde_json::to_writer(writer, &save_data).map_err(std::io::Error::other)
    }

    pub fn load_json(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let save_data: TransformerSaveData = serde_json::from_reader(reader).map_err(std::io::Error::other)?;
        let cpu = CpuTransformer::from_weights(&save_data.weights, &save_data.config);
        Ok(Self {
            cpu,
            best_loss: save_data.best_loss,
            config: save_data.config,
        })
    }
}
