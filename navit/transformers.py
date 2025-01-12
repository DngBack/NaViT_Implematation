from typing import Optional
from torch import Tensor, nn

from utils import LayerNorm, Attention, FeedForward

class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of multi-head attention and a feedforward network.

    Args:
        dim (int): The dimension of the input and output embeddings.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mlp_dim (int): The dimension of the feedforward network.
        dropout (float): Dropout probability.
    """
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm = LayerNorm(dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for the transformer block.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, num_patches, dim)`.
            mask (Optional[Tensor]): Mask for padding.
            attn_mask (Optional[Tensor]): Mask for attention.

        Returns:
            Tensor: Output tensor of shape `(batch_size, num_patches, dim)`.
        """
        x = self.attn(x, mask=mask, attn_mask=attn_mask) + x
        x = self.ff(x) + x
        return self.norm(x)
    
class Transformer(nn.Module):
    """
    A stack of transformer blocks.

    Args:
        dim (int): The dimension of the input and output embeddings.
        depth (int): The number of transformer blocks.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mlp_dim (int): The dimension of the feedforward network.
        dropout (float): Dropout probability.
    """
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = LayerNorm(dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for the transformer.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, num_patches, dim)`.
            mask (Optional[Tensor]): Mask for padding.
            attn_mask (Optional[Tensor]): Mask for attention.

        Returns:
            Tensor: Output tensor of shape `(batch_size, num_patches, dim)`.
        """
        for layer in self.layers:
            x = layer(x, mask=mask, attn_mask=attn_mask)
        return self.norm(x)
