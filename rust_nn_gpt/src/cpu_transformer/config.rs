//! Configuration for a general-purpose transformer
#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub height: usize, // used for embed_dim and ff_dim
    pub layers: usize, // number of transformer blocks
    pub learning_rate: f32,
}
