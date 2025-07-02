use rand::Rng;
use rand::SeedableRng;
use rust_nn_gpt::transformer::{Transformer, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = TransformerConfig {
        input_dim: 32,
        model_dim: 64,
        output_dim: 32,
        num_heads: 2,
        num_layers: 4,
    };
    let batch_size = 64;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42); // Fixed seed for reproducibility
    let inputs: Vec<Vec<f32>> = (0..batch_size)
        .map(|_| {
            (0..config.input_dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect();
    let targets: Vec<Vec<f32>> = (0..batch_size)
        .map(|_| {
            (0..config.output_dim)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect()
        })
        .collect();
    // Create Transformer
    let mut transformer = Transformer::new(config);
    // Try to load previous best model
    if let Ok(loaded) = Transformer::load_json("best_model.json") {
        transformer = loaded;
        println!("Loaded model from best_model.json");
    } else {
        println!("No previous model found, starting fresh.");
    }
    // Use best_loss from transformer
    let mut best_loss = transformer.best_loss.unwrap_or(f32::INFINITY);
    // Training loop
    for epoch in 0..1000 {
        let loss = transformer.train(&inputs, &targets)?;
        println!("Epoch {}: loss = {}", epoch, loss);
        if loss < best_loss {
            best_loss = loss;
            transformer.best_loss = Some(best_loss);
            transformer.save_json("best_model.json")?;
        }
    }
    Ok(())
}
