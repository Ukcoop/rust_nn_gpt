use rand::Rng;
use rust_nn_gpt::transformer::{Transformer, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = TransformerConfig {
        input_dim: 32,
        model_dim: 64,
        output_dim: 32,
        num_heads: 2,
        num_layers: 4,
    };
    let mut rng = rand::thread_rng();
    let batch_size = 64;
    let mut inputs: Vec<Vec<f32>> = (0..batch_size)
        .map(|_| {
            (0..config.input_dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect();
    let mut targets: Vec<Vec<f32>> = (0..batch_size)
        .map(|_| {
            (0..config.output_dim)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect()
        })
        .collect();
    // Create Transformer
    let mut transformer = Transformer::new(config);
    // Training loop
    for epoch in 0..1000 {
        let loss = transformer.train(&inputs, &targets)?;
        println!("Epoch {}: loss = {}", epoch, loss);
    }
    Ok(())
}
