use crate::cpu_transformer as cpu;
use crate::cpu_transformer::CpuTransformer;
use crate::gpu_transformer::GpuTransformer;
pub use cpu::TransformerConfig;

/// Selects which backend to use for the transformer (CPU or GPU)
pub enum TransformerBackend {
    Cpu,
    Gpu,
    Auto,
}

/// A wrapper for either a CPU or GPU transformer
pub struct Transformer {
    backend: TransformerImpl,
    config: TransformerConfig,
}

/// Internal enum for backend implementations (boxed for size parity)
enum TransformerImpl {
    Cpu(Box<CpuTransformer>),
    Gpu(Box<GpuTransformer>),
}

impl Transformer {
    /// Create a new Transformer with the specified backend and config
    pub async fn new(
        backend: TransformerBackend,
        config: TransformerConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let backend = match backend {
            TransformerBackend::Cpu => TransformerImpl::Cpu(Box::new(CpuTransformer::new(&config))),
            TransformerBackend::Gpu => {
                let gpu = GpuTransformer::new(&config).await?;
                TransformerImpl::Gpu(Box::new(gpu))
            }
            TransformerBackend::Auto => {
                if Self::gpu_available().await {
                    let gpu = GpuTransformer::new(&config).await?;
                    TransformerImpl::Gpu(Box::new(gpu))
                } else {
                    TransformerImpl::Cpu(Box::new(CpuTransformer::new(&config)))
                }
            }
        };
        Ok(Self { backend, config })
    }

    /// Forward pass: 1D input -> 1D output
    pub fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        match &self.backend {
            TransformerImpl::Cpu(cpu) => cpu.forward(&input),
            TransformerImpl::Gpu(gpu) => gpu.forward(&input),
        }
    }

    /// Training: input and target are 1D vectors (single instance)
    pub fn train(&mut self, input: Vec<f32>, target: Vec<f32>) -> f32 {
        match &mut self.backend {
            TransformerImpl::Cpu(cpu) => cpu.train(&input, &target, self.config.learning_rate),
            TransformerImpl::Gpu(gpu) => gpu.train(&input, &target, self.config.learning_rate),
        }
    }

    /// Predict: input is a 1D vector (single instance)
    pub fn predict(&self, input: Vec<f32>) -> Vec<f32> {
        self.forward(input)
    }

    /// Helper function to check if a GPU is available for wgpu
    async fn gpu_available() -> bool {
        let adapters = wgpu::Instance::default().enumerate_adapters(wgpu::Backends::all());
        for adapter in adapters {
            let info = adapter.get_info();
            if info.device_type != wgpu::DeviceType::Cpu {
                return true;
            }
        }
        false
    }
}
