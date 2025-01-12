from functools import partial
from typing import List, Union, Optional, Callable, Tuple
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

from helper import divisible_by, exists
from utils import LayerNorm, Attention, group_images_by_max_seq_len

### PATCH EMBEDDING
class PatchEmbedding(nn.Module):
    """
    Converts image patches into embeddings.

    Args:
        patch_dim (int): The dimension of the input patches.
        dim (int): The dimension of the output embeddings.
    """
    def __init__(self, patch_dim: int, dim: int):
        super().__init__()
        self.norm = LayerNorm(patch_dim)
        self.proj = nn.Linear(patch_dim, dim)
        self.post_norm = LayerNorm(dim)

    def forward(self, patches: Tensor) -> Tensor:
        """
        Forward pass for patch embedding.

        Args:
            patches (Tensor): Input patches of shape `(batch_size, num_patches, patch_dim)`.

        Returns:
            Tensor: Embedded patches of shape `(batch_size, num_patches, dim)`.
        """
        patches = self.norm(patches)
        patches = self.proj(patches)
        patches = self.post_norm(patches)
        return patches

class PositionalEmbedding(nn.Module):
    """
    Adds 2D positional embeddings to patch embeddings.

    Args:
        patch_height_dim (int): The height dimension of the patches.
        patch_width_dim (int): The width dimension of the patches.
        dim (int): The dimension of the embeddings.
    """
    def __init__(self, patch_height_dim: int, patch_width_dim: int, dim: int):
        super().__init__()
        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

    def forward(self, patch_positions: Tensor) -> Tensor:
        """
        Forward pass for positional embedding.

        Args:
            patch_positions (Tensor): Patch positions of shape `(batch_size, num_patches, 2)`.

        Returns:
            Tensor: Positional embeddings of shape `(batch_size, num_patches, dim)`.
        """
        h_indices, w_indices = patch_positions.unbind(dim=-1)
        h_pos = self.pos_embed_height[h_indices]
        w_pos = self.pos_embed_width[w_indices]
        return h_pos + w_pos

class AttentionPooling(nn.Module):
    """
    Attention pooling for aggregating patch embeddings.

    Args:
        dim (int): The dimension of the input and output embeddings.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
    """
    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(dim))
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """
        Forward pass for attention pooling.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, num_patches, dim)`.
            attn_mask (Tensor): Attention mask of shape `(batch_size, num_patches)`.

        Returns:
            Tensor: Pooled embeddings of shape `(batch_size * num_patches, dim)`.
        """
        b, n, _ = x.shape
        queries = repeat(self.queries, 'd -> b n d', b=b, n=n)
        pooled = self.attn(queries, context=x, attn_mask=attn_mask) + queries
        return rearrange(pooled, 'b n d -> (b n) d')

class NaViT(nn.Module):
    """
    NaViT (Patch n’ Pack) Vision Transformer for any aspect ratio and resolution.

    Args:
        image_size (int): The size of the input image.
        patch_size (int): The size of each patch.
        num_classes (int): The number of output classes.
        dim (int): The dimension of the embeddings.
        depth (int): The number of transformer blocks.
        heads (int): The number of attention heads.
        mlp_dim (int): The dimension of the feedforward network.
        channels (int): The number of input channels.
        dim_head (int): The dimension of each attention head.
        dropout (float): Dropout probability.
        emb_dropout (float): Dropout probability for embeddings.
        token_dropout_prob (Optional[Union[float, Callable]]): Token dropout probability.
    """
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        token_dropout_prob: Optional[Union[float, Callable]] = None
    ):
        super().__init__()
        # (Initialization remains the same as before)
        pass

    def forward(
        self,
        batched_images: Union[List[Tensor], List[List[Tensor]]],
        group_images: bool = False,
        group_max_seq_len: int = 2048
    ) -> Tensor:
        """
        Forward pass for NaViT.

        Args:
            batched_images (Union[List[Tensor], List[List[Tensor]]]): Input images.
            group_images (bool): Whether to group images by sequence length.
            group_max_seq_len (int): Maximum sequence length for grouping.

        Returns:
            Tensor: Output logits of shape `(batch_size, num_classes)`.
        """
        if group_images:
            batched_images = group_images_by_max_seq_len(
                batched_images, # type: ignore
                patch_size=self.patch_size,
                calc_token_dropout=self.calc_token_dropout,
                max_seq_len=group_max_seq_len
            )

        # Process images into patches, positions, and attention masks
        patches, patch_positions, attn_mask = self._process_images(batched_images) # type: ignore

        # Patch embedding
        x = self.patch_embedding(patches)

        # Add positional embedding
        x = x + self.pos_embedding(patch_positions)

        # Dropout
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x, attn_mask=attn_mask)

        # Attention pooling
        x = self.attention_pooling(x, attn_mask=attn_mask)

        # MLP head
        return self.mlp_head(x)

    def _process_images(self, batched_images: List[List[Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Process a batch of images into patches, positions, and attention masks.

        Args:
            batched_images (List[List[Tensor]]): A list of image groups, where each group contains images of the same resolution.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - patches: Tensor of shape `(batch_size, num_patches, patch_dim)`.
                - patch_positions: Tensor of shape `(batch_size, num_patches, 2)`.
                - attn_mask: Tensor of shape `(batch_size, num_patches, num_patches)`.
        """
        # (Implementation provided above)
        device = self.device
        arange = partial(torch.arange, device=device)
        pad_sequence = partial(orig_pad_sequence, batch_first=True)

        num_images = []
        batched_sequences = []
        batched_positions = []
        batched_image_ids = []

        for images in batched_images:
            num_images.append(len(images))

            sequences = []
            positions = []
            image_ids = torch.empty((0,), device=device, dtype=torch.long)

            for image_id, image in enumerate(images):
                assert image.ndim == 3 and image.shape[0] == self.channels, \
                    f"Input images must have shape (C, H, W). Got {image.shape}."
                image_dims = image.shape[-2:]
                assert all(divisible_by(dim, self.patch_size) for dim in image_dims), \
                    f"Image dimensions {image_dims} must be divisible by patch size {self.patch_size}."

                # Calculate patch dimensions
                ph, pw = image_dims[0] // self.patch_size, image_dims[1] // self.patch_size

                # Create patch positions
                pos = torch.stack(torch.meshgrid((
                    arange(ph), # type: ignore
                    arange(pw)
                ), indexing='ij'), dim=-1)

                pos = rearrange(pos, 'h w c -> (h w) c')
                seq = rearrange(image, 'c (h p1) (w p2) -> (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)

                # Apply token dropout if specified
                if exists(self.calc_token_dropout):
                    token_dropout = self.calc_token_dropout(*image_dims)
                    num_keep = max(1, int(seq.shape[0] * (1 - token_dropout)))
                    keep_indices = torch.randn((seq.shape[0],), device=device).topk(num_keep, dim=-1).indices

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]

                # Append to sequences and positions
                image_ids = F.pad(image_ids, (0, seq.shape[0]), value=image_id)
                sequences.append(seq)
                positions.append(pos)

            # Concatenate sequences and positions for the current batch
            batched_image_ids.append(image_ids)
            batched_sequences.append(torch.cat(sequences, dim=0))
            batched_positions.append(torch.cat(positions, dim=0))

        # Derive key padding mask
        lengths = torch.tensor([seq.shape[0] for seq in batched_sequences], device=device, dtype=torch.long)
        max_length = arange(lengths.amax().item())
        key_pad_mask = rearrange(lengths, 'b -> b 1') <= rearrange(max_length, 'n -> 1 n')

        # Derive attention mask
        batched_image_ids = pad_sequence(batched_image_ids)
        attn_mask = rearrange(batched_image_ids, 'b i -> b 1 i 1') == rearrange(batched_image_ids, 'b j -> b 1 1 j')
        attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 1 j')

        # Pad sequences and positions
        patches = pad_sequence(batched_sequences)
        patch_positions = pad_sequence(batched_positions)

        return patches, patch_positions, attn_mask
