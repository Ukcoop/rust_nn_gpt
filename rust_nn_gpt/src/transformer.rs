use crate::cpu_transformer::CpuTransformer;
use crate::gpu_transformer::VulkanTransformer;
use crate::gpu_transformer::blocks::linear::{BatchContext, OptimizerParams};
use crate::transformer_weights::TransformerWeights;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Backend {
    Cpu,
    Gpu,
    Vulkan,
    Auto,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TransformerConfig {
    pub input_dim: usize,
    pub model_dim: usize,
    pub output_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub backend: Backend,
}

pub struct Transformer {
    pub cpu: Option<CpuTransformer>,
    pub gpu: Option<VulkanTransformer>,
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
    pub fn new(config: TransformerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        match config.backend {
            Backend::Cpu => {
                let cpu = CpuTransformer::new(&config);
                Ok(Self {
                    cpu: Some(cpu),
                    gpu: None,
                    best_loss: None,
                    config,
                })
            }
            Backend::Gpu | Backend::Vulkan => match VulkanTransformer::new(&config) {
                Ok(gpu) => Ok(Self {
                    cpu: None,
                    gpu: Some(gpu),
                    best_loss: None,
                    config,
                }),
                Err(e) => {
                    eprintln!("Failed to initialize GPU transformer: {e}");
                    eprintln!("Falling back to CPU transformer...");
                    let cpu = CpuTransformer::new(&config);
                    Ok(Self {
                        cpu: Some(cpu),
                        gpu: None,
                        best_loss: None,
                        config: TransformerConfig {
                            backend: Backend::Cpu,
                            ..config
                        },
                    })
                }
            },
            Backend::Auto => {
                // Try GPU first, fall back to CPU if it fails
                match VulkanTransformer::new(&config) {
                    Ok(gpu) => {
                        println!("Successfully initialized GPU transformer");
                        Ok(Self {
                            cpu: None,
                            gpu: Some(gpu),
                            best_loss: None,
                            config: TransformerConfig {
                                backend: Backend::Vulkan,
                                ..config
                            },
                        })
                    }
                    Err(e) => {
                        println!("Failed to initialize GPU transformer: {e}");
                        println!("Falling back to CPU transformer...");
                        let cpu = CpuTransformer::new(&config);
                        Ok(Self {
                            cpu: Some(cpu),
                            gpu: None,
                            best_loss: None,
                            config: TransformerConfig {
                                backend: Backend::Cpu,
                                ..config
                            },
                        })
                    }
                }
            }
        }
    }

    pub fn train(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
    ) -> Result<f32, Box<dyn std::error::Error>> {
        match &mut self.gpu {
            Some(gpu) => {
                // Use the actual GPU training pipeline
                let batch_size = inputs.len();
                let input_dim = self.config.input_dim;
                let output_dim = self.config.output_dim;

                // Flatten inputs and targets for batch processing
                let mut flat_inputs = Vec::with_capacity(batch_size * input_dim);
                let mut flat_targets = Vec::with_capacity(batch_size * output_dim);

                for input in inputs {
                    flat_inputs.extend(input);
                }
                for target in targets {
                    flat_targets.extend(target);
                }

                let opt = OptimizerParams {
                    learning_rate: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                };
                let batch = BatchContext {
                    batch_size,
                    seq_len: 1,
                };
                return gpu.train_batch_adam(&flat_inputs, &flat_targets, &opt, &batch);
            }
            None => {
                if let Some(cpu) = &mut self.cpu {
                    cpu.train_batch(inputs, targets)
                } else {
                    Err("No transformer backend available".into())
                }
            }
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        match &mut self.gpu {
            Some(gpu) => {
                // Use the GPU transformer's forward method with batch size 1
                gpu.forward(input, 1)
            }
            None => {
                if let Some(cpu) = &mut self.cpu {
                    Ok(cpu.forward(input))
                } else {
                    Err("No transformer backend available".into())
                }
            }
        }
    }

    pub fn forward_batch(
        &mut self,
        flat_inputs: &[f32],
        batch_size: usize,
        _training: bool,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        match &mut self.gpu {
            Some(gpu) => gpu.forward(flat_inputs, batch_size),
            None => {
                // Reshape flat inputs into batch
                let input_dim = self.config.input_dim;
                let inputs: Vec<Vec<f32>> = flat_inputs
                    .chunks(input_dim)
                    .take(batch_size)
                    .map(|chunk| chunk.to_vec())
                    .collect();

                // Process each input in the batch
                let mut outputs = Vec::new();
                for input in inputs {
                    let output = self.forward(&input)?;
                    outputs.extend(output);
                }

                Ok(outputs)
            }
        }
    }

    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        let weights = match &self.gpu {
            Some(_gpu) => {
                // TODO: Implement weights extraction from Vulkan transformer
                // For now, create a dummy CPU transformer to get weights structure
                let dummy_cpu = CpuTransformer::new(&self.config);
                TransformerWeights::from(&dummy_cpu)
            }
            None => {
                if let Some(cpu) = &self.cpu {
                    TransformerWeights::from(cpu)
                } else {
                    return Err(std::io::Error::other("No transformer backend available"));
                }
            }
        };
        let save_data = TransformerSaveData {
            config: self.config.clone(),
            weights,
            best_loss: self.best_loss,
        };
        serde_json::to_writer(writer, &save_data).map_err(std::io::Error::other)
    }

    pub fn load_json(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let save_data: TransformerSaveData =
            serde_json::from_reader(reader).map_err(std::io::Error::other)?;

        let (cpu, gpu) = match save_data.config.backend {
            Backend::Cpu => {
                let cpu = CpuTransformer::from_weights(&save_data.weights, &save_data.config);
                (Some(cpu), None)
            }
            Backend::Gpu | Backend::Vulkan => {
                match VulkanTransformer::new(&save_data.config) {
                    Ok(gpu) => (None, Some(gpu)),
                    Err(_) => {
                        // Fallback to CPU if GPU loading fails
                        let cpu =
                            CpuTransformer::from_weights(&save_data.weights, &save_data.config);
                        (Some(cpu), None)
                    }
                }
            }
            Backend::Auto => {
                // For Auto backend, try GPU first, fall back to CPU
                match VulkanTransformer::new(&save_data.config) {
                    Ok(gpu) => (None, Some(gpu)),
                    Err(_) => {
                        // Fallback to CPU if GPU loading fails
                        let cpu =
                            CpuTransformer::from_weights(&save_data.weights, &save_data.config);
                        (Some(cpu), None)
                    }
                }
            }
        };

        Ok(Self {
            cpu,
            gpu,
            best_loss: save_data.best_loss,
            config: save_data.config,
        })
    }
}
