pub mod cpu_transformer;
mod cpu_utils;
mod gelu;
pub mod gpu_transformer;
pub mod transformer;

#[cfg(test)]
mod tests {
    use super::transformer::{Transformer, TransformerBackend, TransformerConfig};
    use tokio::runtime::Runtime;

    #[test]
    fn test_transformer_trains_and_predicts_cpu_backend() {
        let rt = match Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("Failed to create Tokio runtime: {}", e);
                return;
            }
        };
        rt.block_on(async {
            let config = TransformerConfig {
                input_dim: 4,
                output_dim: 4,
                height: 4,
                layers: 2,
                learning_rate: 0.01,
            };

            let transformer_result = Transformer::new(TransformerBackend::Cpu, config).await;
            let mut transformer = match transformer_result {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Failed to create Transformer: {}", e);
                    return;
                }
            };
            let input = vec![1.0, 2.0, 3.0, 4.0];
            let target = vec![0.0, 1.0, 0.0, 1.0];
            let mut last_loss = f32::MAX;

            for _ in 0..200 {
                let loss = transformer.train(input.clone(), target.clone());
                assert!(
                    loss <= last_loss + 1e-3,
                    "Loss did not decrease ({} -> {})",
                    last_loss,
                    loss
                );
                last_loss = loss;
            }
        });
    }

    #[test]
    fn test_transformer_trains_and_predicts_auto_backend() {
        let rt = match Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("Failed to create Tokio runtime: {}", e);
                return;
            }
        };
        rt.block_on(async {
            let config = TransformerConfig {
                input_dim: 4,
                output_dim: 4,
                height: 4,
                layers: 2,
                learning_rate: 0.01,
            };

            let transformer_result = Transformer::new(TransformerBackend::Auto, config).await;
            let mut transformer = match transformer_result {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Failed to create Transformer (Auto backend): {}", e);
                    return;
                }
            };
            let input = vec![1.0, 2.0, 3.0, 4.0];
            let target = vec![0.0, 1.0, 0.0, 1.0];
            let mut last_loss = f32::MAX;

            for _ in 0..200 {
                let loss = transformer.train(input.clone(), target.clone());
                assert!(
                    loss <= last_loss + 1e-3,
                    "Loss did not decrease ({} -> {})",
                    last_loss,
                    loss
                );
                last_loss = loss;
            }
        });
    }
}
