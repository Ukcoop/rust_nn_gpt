use crate::cpu_transformer::attention::MultiHeadAttention;
use crate::cpu_transformer::feedforward::FeedForward;
use crate::cpu_transformer::layernorm::LayerNorm;

/// A single transformer block: (Attn -> Add & Norm -> FF -> Add & Norm)
pub struct TransformerBlock {
    pub attn: MultiHeadAttention,
    pub norm1: LayerNorm,
    pub ff: FeedForward,
    pub norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(config: &crate::cpu_transformer::config::TransformerConfig) -> Self {
        Self {
            attn: MultiHeadAttention::new(config),
            norm1: LayerNorm::new(config),
            ff: FeedForward::new(config),
            norm2: LayerNorm::new(config),
        }
    }
    /// Forward pass for a single block
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Attention + Add & Norm
        let attn_out = self.attn.forward(input);
        let x = input
            .iter()
            .zip(attn_out.iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();

        let normed1 = self.norm1.forward(&x);

        // FeedForward + Add & Norm
        let ff_out = self.ff.forward(&normed1);
        let x2 = normed1
            .iter()
            .zip(ff_out.iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();

        self.norm2.forward(&x2)
    }
}
