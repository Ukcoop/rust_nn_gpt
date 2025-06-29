use rust_nn_gpt::transformer::{Transformer, TransformerBackend, TransformerConfig};

#[tokio::main]
async fn main() {
    // Select backend: TransformerBackend::Cpu or TransformerBackend::Gpu
    let backend = TransformerBackend::Auto; // Automatically select GPU if available, otherwise CPU
    // let backend = TransformerBackend::Gpu; // Uncomment to test GPU (will panic on unimplemented)

    // Define model config for a larger model
    let config = TransformerConfig {
        input_dim: 8,
        output_dim: 8,
        height: 8,
        layers: 4,
        learning_rate: 0.1,
    };

    let mut transformer = match Transformer::new(backend, config.clone()).await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to initialize transformer: {}", e);
            return;
        }
    };

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let target = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let epochs = 1000;

    for epoch in 0..epochs {
        let loss = transformer.train(input.clone(), target.clone());
        println!("Epoch {}: Loss = {}", epoch + 1, loss);
    }

    let prediction = transformer.predict(input);
    println!("Final Prediction: {:?}", prediction);
}
