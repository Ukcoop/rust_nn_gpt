use rand::Rng;
use rust_nn_gpt::transformer::{Backend, Transformer, TransformerConfig};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Model parameters
    let input_dim = 16;
    let model_dim = 32;
    let output_dim = 16;
    let num_heads = 4;
    let num_layers = 2;
    let batch_size = 8;
    let epochs = 100;

    // Generate random data
    let mut rng = rand::thread_rng();
    let inputs: Vec<Vec<f32>> = (0..batch_size)
        .map(|_| (0..input_dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let targets: Vec<Vec<f32>> = (0..batch_size)
        .map(|_| (0..output_dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    // Create config with Auto backend - will try GPU first, fall back to CPU
    let config = TransformerConfig {
        input_dim,
        model_dim,
        output_dim,
        num_heads,
        num_layers,
        backend: Backend::Auto,
    };

    println!("Initializing transformer with Auto backend...");
    let mut transformer = Transformer::new(config)?;

    // Check which backend was actually used
    match transformer.config.backend {
        Backend::Cpu => println!("Using CPU backend"),
        Backend::Vulkan => println!("Using GPU (Vulkan) backend"),
        Backend::Gpu => println!("Using GPU backend"),
        Backend::Auto => println!("Auto backend selected"),
    }

    println!("Training for {epochs} epochs, batch size {batch_size}");

    for epoch in 0..epochs {
        let loss = transformer.train(&inputs, &targets)?;
        println!("Epoch {:2}: Loss = {:8.6}", epoch + 1, loss);
    }

    Ok(())
}
