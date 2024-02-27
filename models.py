# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    """    Modulate the input tensor by applying shift and scale.

    Args:
        x (tensor): Input tensor to be modulated.
        shift (tensor): Shift tensor to be applied.
        scale (tensor): Scale tensor to be applied.

    Returns:
        tensor: The modulated tensor.
    """

    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        """        Initialize the model with the given hidden size and frequency embedding size.

        Args:
            hidden_size (int): The size of the hidden layer.
            frequency_embedding_size (int?): The size of the frequency embedding. Defaults to 256.
        """

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """        Create sinusoidal timestep embeddings.

        This function creates sinusoidal timestep embeddings based on the input indices and dimension.

        Args:
            t (Tensor): A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim (int): The dimension of the output.
            max_period (int?): Controls the minimum frequency of the embeddings. Defaults to 10000.

        Returns:
            Tensor: An (N, D) Tensor of positional embeddings.
                Reference:
            https: //github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """        Forward pass of the model to compute the timestep embedding.

        Args:
            t (tensor): Input tensor representing the timestep.

        Returns:
            tensor: The computed timestep embedding.
        """

        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        """        Initialize the model with the specified parameters.

        Args:
            num_classes (int): The number of classes for the model.
            hidden_size (int): The size of the hidden layer.
            dropout_prob (float): The probability of dropout.
        """

        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """        Drops labels to enable classifier-free guidance.

        This function drops labels to enable classifier-free guidance. It generates drop_ids based on the dropout probability
        and updates the labels accordingly.

        Args:
            labels (tensor): The input labels.
            force_drop_ids (tensor?): A tensor indicating the force drop ids.

        Returns:
            tensor: The updated labels after dropping.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        """        Forward pass through the neural network model.

        This method takes input labels and performs a forward pass through the neural network model. If training and dropout is enabled, or if force_drop_ids is provided, it applies token dropout to the input labels. Then, it retrieves the embeddings for the modified labels from the embedding table.

        Args:
            labels (tensor): Input labels for the forward pass.
            train (bool): A flag indicating whether the model is in training mode.
            force_drop_ids (list?): A list of indices to force dropout on.

        Returns:
            tensor: The embeddings obtained after the forward pass.
        """

        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        """        Initializes the model with the given parameters.

        Args:
            hidden_size (int): The dimension of the hidden state.
            num_heads (int): The number of attention heads.
            mlp_ratio (float?): The ratio of the hidden size for the MLP. Defaults to 4.0.
            **block_kwargs: Additional keyword arguments for the attention block.


        Note:
            This method initializes the model with layer normalization, attention mechanism, multi-layer perceptron (MLP),
            and adaptive layer normalization modulation.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """        Apply forward pass of the network with adaptive layer normalization modulation.

        This function applies the forward pass of the network with adaptive layer normalization modulation. It first computes the modulation parameters and then applies the modulation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            c (torch.Tensor): The conditioning tensor.

        Returns:
            torch.Tensor: The output tensor after applying the forward pass.
        """

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        """        Initialize the model with given parameters.

        Args:
            hidden_size (int): The size of the hidden layer.
            patch_size (int): The size of the patch.
            out_channels (int): The number of output channels.
        """

        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """        Apply forward pass through the neural network with adaptive layer normalization modulation.

        This function applies the forward pass through the neural network with adaptive layer normalization modulation.
        It first calculates the shift and scale using the adaLN_modulation method, then modulates the input x using the calculated shift and scale.
        Finally, it applies a linear transformation to the modulated input.

        Args:
            x (tensor): The input tensor to the neural network.
            c (tensor): The conditioning tensor for adaptive layer normalization modulation.

        Returns:
            tensor: The output tensor after the forward pass.
        """

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        """        Initializes the DiT (Data-efficient image Transformer) model with the given parameters.

        Args:
            input_size (int): The input size of the image.
            patch_size (int): The size of the image patches.
            in_channels (int): The number of input channels.
            hidden_size (int): The hidden size for the transformer model.
            depth (int): The depth of the transformer model.
            num_heads (int): The number of attention heads.
            mlp_ratio (float): The ratio of the hidden size in the feedforward network to the hidden size in the attention mechanism.
            class_dropout_prob (float): The probability of dropout for the class token.
            num_classes (int): The number of output classes.
            learn_sigma (bool): Whether to learn the sigma parameter.
        """

        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        """        Initialize the weights of the transformer model.

        This function initializes the weights of the transformer model including transformer layers, position embeddings,
        patch embeddings, label embedding table, timestep embedding MLP, and modulation layers in DiT blocks.

        Args:
            self: The transformer model instance.
        """

        # Initialize transformer layers:
        def _basic_init(module):
            """            Initialize the transformer layers with Xavier uniform initialization for weights and constant initialization for bias.

            Args:
                module (torch.nn.Module): The module to be initialized.
            """

            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """        Unpatchify the input tensor to reconstruct the original images.

        Args:
            x (tensor): Input tensor of shape (N, T, patch_size**2 * C), where N is the batch size,
                T is the number of patches, patch_size is the size of each patch, and C is the number of channels.

        Returns:
            tensor: Reconstructed images of shape (N, H, W, C), where H and W are the height and width of the original images.

        Raises:
            AssertionError: If the input tensor cannot be reshaped into a valid image grid.
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """        Forward pass of DiT.

        Args:
            x (torch.Tensor): (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
            t (torch.Tensor): (N,) tensor of diffusion timesteps
            y (torch.Tensor): (N,) tensor of class labels

        Returns:
            torch.Tensor: (N, out_channels, H, W) tensor representing the output of the forward pass
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.

        This method performs a forward pass of DiT while also batching the unconditional forward pass for classifier-free guidance.

        Args:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The target tensor.
            y (torch.Tensor): The output tensor.
            cfg_scale (float): The scale factor for classifier-free guidance.

        Returns:
            torch.Tensor: The concatenated tensor of epsilon and rest.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """    Get 2D sine/cosine positional embedding.

    This function calculates the 2D sine/cosine positional embedding based on the input parameters.

    Args:
        embed_dim (int): The dimension of the embedding.
        grid_size (int): The grid height and width.
        cls_token (bool?): Whether to include a cls_token. Defaults to False.
        extra_tokens (int?): The number of extra tokens. Defaults to 0.

    Returns:
        numpy.ndarray: The positional embedding array of shape [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (with or without cls_token).
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """    Get 2D sinusoidal positional embeddings from a grid.

    This function takes the embedding dimension and a 2D grid as input and returns the 2D sinusoidal positional embeddings.

    Args:
        embed_dim (int): The embedding dimension, must be divisible by 2.
        grid (tuple): A tuple containing the 2D grid dimensions.

    Returns:
        numpy.ndarray: The 2D sinusoidal positional embeddings.
    """

    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """    Encode positions into a 1D sine-cosine positional embedding grid.

    Args:
        embed_dim (int): Output dimension for each position.
        pos (ndarray): A list of positions to be encoded: size (M,).

    Returns:
        ndarray: A 2D array of shape (M, D) representing the positional embeddings.

    Raises:
        AssertionError: If embed_dim is not divisible by 2.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    """    DiT model with extra large configuration and patch size of 2.

    This function returns a DiT model with a depth of 28, hidden size of 1152, patch size of 2, and 16 attention heads.

    Args:
        **kwargs: Additional keyword arguments for the DiT model.

    Returns:
        DiT: A DiT model with the specified configuration.
    """

    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    """    Create a DiT (Data-efficient image Transformer) model with extra large configuration and patch size 4.

    This function creates a Data-efficient image Transformer (DiT) model with an extra large configuration, including a depth of 28, a hidden size of 1152, and 4x4 patch size, along with additional keyword arguments.

    Args:
        **kwargs: Additional keyword arguments for model customization.

    Returns:
        DiT: A Data-efficient image Transformer model with the specified configuration.
    """

    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    """    Create a DiT (Diversity Transformer) model with extra large configuration.

    This function creates a Diversity Transformer (DiT) model with the following configuration:
    - Depth: 28
    - Hidden size: 1152
    - Patch size: 8
    - Number of heads: 16

    Args:
        **kwargs: Additional keyword arguments for the DiT model.

    Returns:
        DiT: A Diversity Transformer model with the specified configuration.
    """

    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    """    Create a Diverse Transformer with a depth of 24, hidden size of 1024, patch size of 2, and 16 attention heads.

    Args:
        **kwargs: Additional keyword arguments for the Diverse Transformer.

    Returns:
        DiverseTransformer: An instance of the Diverse Transformer model.
    """

    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    """    Create a Diverse Transformer (DiT) model with a depth of 24, hidden size of 1024, patch size of 4, and 16 attention heads.

    Args:
        **kwargs: Additional keyword arguments for the DiT model.

    Returns:
        DiT: A Diverse Transformer model with the specified configuration.
    """

    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    """    Create a Distransformer (DiT) model with a depth of 24, hidden size of 1024, patch size of 8, and 16 attention heads.

    Args:
        **kwargs: Additional keyword arguments for the DiT model.

    Returns:
        DiT: A Distransformer model with the specified configuration.
    """

    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    """    Create a Diverse Transformer (DiT) model with a specific configuration.

    This function creates a Diverse Transformer (DiT) model with the specified configuration parameters.

    Args:
        **kwargs: Additional keyword arguments to be passed to the DiT model.

    Returns:
        DiT: A Diverse Transformer model with the specified configuration.
    """

    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    """    Create a Diverse Transformer (DiT) model with a specified depth, hidden size, patch size, and number of heads.

    Args:
        **kwargs: Additional keyword arguments for the DiT model.

    Returns:
        DiT: A Diverse Transformer model with the specified configuration.
    """

    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    """    Create a Diverse Transformer (DiT) model with a patch size of 8.

    This function creates a Diverse Transformer (DiT) model with the specified parameters, including depth, hidden size, and number of heads. The patch size is set to 8 by default.

    Args:
        **kwargs: Additional keyword arguments for configuring the DiT model.

    Returns:
        DiT: A Diverse Transformer model with the specified parameters.
    """

    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    """    Create a DiT model with specific configuration.

    This function creates a Distransformer (DiT) model with the specified depth, hidden size, patch size, and number of heads, along with any additional keyword arguments provided.

    Args:
        **kwargs: Additional keyword arguments for configuring the DiT model.

    Returns:
        DiT: A Distransformer model with the specified configuration.
    """

    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    """    Create a DiT model with specific configuration parameters.

    This function creates a Distransformer (DiT) model with the specified configuration parameters.
    The DiT model is created with a depth of 12, hidden size of 384, patch size of 4, and 6 attention heads.

    Args:
        **kwargs: Additional keyword arguments for configuring the DiT model.

    Returns:
        DiT: A Distransformer model with the specified configuration.
    """

    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    """    Create a DiT model with specific configuration.

    This function creates a DiT (Data-efficient image Transformer) model with the specified configuration parameters.

    Args:
        **kwargs: Additional keyword arguments for configuring the DiT model.

    Returns:
        DiT: A Data-efficient image Transformer model.
    """

    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
