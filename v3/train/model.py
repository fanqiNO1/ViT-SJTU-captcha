import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """The FFN layer in ViT.

    Args:
        hidden_size (int): The hidden dimension of the FFN layer.
        intermediate_size (int): The intermediate size of the FFN layer.
        act_fn (nn.Module): The activation function to be used.
        Defaults to nn.SiLU().
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 act_fn: nn.Module = nn.SiLU()):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(hidden_state))
        up = self.up_proj(hidden_state)
        hidden_state = self.down_proj(gate * up)
        return hidden_state


def rotate_half(x):
    """Rotate half the hidden dims of the input tensor."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(tensor: torch.Tensor,
                         freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to the input tensor."""
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 1, 2).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 1, 2).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


def repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if num_repeats == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     num_repeats, slen,
                                                     head_dim)
    hidden_states = hidden_states.reshape(batch,
                                          num_key_value_heads * num_repeats,
                                          slen, head_dim)
    return hidden_states


class Attention(nn.Module):
    """The multi-head self-attention layer in ViT.

    Args:
        hidden_size (int): The hidden dimension of the input tensor.
        num_attention_heads (int): The number of attention heads.
        num_key_value_heads (int): The number of key and value heads.
        dropout (float): The dropout probability. Defaults to 0.
    """

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 dropout: float = 0.):
        super().__init__()
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = dropout

        self.q_proj = nn.Linear(hidden_size,
                                num_attention_heads * self.head_dim,
                                bias=True)
        self.k_proj = nn.Linear(hidden_size,
                                num_key_value_heads * self.head_dim,
                                bias=True)
        self.v_proj = nn.Linear(hidden_size,
                                num_key_value_heads * self.head_dim,
                                bias=True)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim,
                                hidden_size,
                                bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor,
                rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        """The forward pass of the attention layer."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        q = apply_rotary_pos_emb(q, rotary_pos_emb)
        k = apply_rotary_pos_emb(k, rotary_pos_emb)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1,
                                 dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class RMSNorm(nn.Module):
    """Root mean square layer normalization.

    Args:
        hidden_size (int): The hidden dimension of the input tensor.
        eps (float): The epsilon value for numerical stability.
        Defaults to 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The forward pass of the RMSNorm layer."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        standard_deviation = torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * standard_deviation
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f'{tuple(self.weight.shape)}, eps={self.variance_epsilon}'


class PatchEmbed(nn.Module):
    """The patch embedding layer.

    Args:
        patch_size (int): The patch size to be used.
        in_channels (int): The number of input channels.
        embed_dim (int): The embedding dimension.
    """

    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [patch_size, patch_size]
        stride = [patch_size, patch_size]
        self.proj = nn.Conv2d(in_channels,
                              embed_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = self.proj(hidden_states.to(target_dtype))
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states


class RotaryEmbedding(nn.Module):
    """The rotary position embedding layer.

    Args:
        dim (int): The embedding dimension.
        theta (int): The theta value for the sinusoidal function.
        Defaults to 10000.
    """

    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        steps = torch.arange(0, dim, 2, dtype=torch.float)
        inv_freq = 1.0 / (theta**(steps / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        """The forward pass of the rotary position embedding layer."""
        seq = torch.arange(seqlen,
                           device=self.inv_freq.device,
                           dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq).unsqueeze(0)
        return freqs


class EncoderLayer(nn.Module):
    """The basic encoder layer in ViT.

    Args:
        hidden_size (int): The hidden dimension of the input tensor.
        num_attention_heads (int): The number of attention heads.
        num_key_value_heads (int): The number of key and value heads.
        intermediate_size (int): The intermediate size of the FFN layer.
        act_fn (nn.Module): The activation function to be used.
        dropout (float): The dropout probability. Defaults to 0.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int,
                 num_key_value_heads: int, intermediate_size: int,
                 act_fn: nn.Module, dropout: float):
        super().__init__()
        self.attn = Attention(hidden_size, num_attention_heads,
                              num_key_value_heads, dropout)
        self.norm1 = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size, act_fn)
        self.norm2 = RMSNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor,
                rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        """The forward pass of the encoder layer."""
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states),
                                                  rotary_pos_emb)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class ViTCaptcha(nn.Module):
    """The Vision Transformer model.

    Args:
        image_size (int): The size of the input image.
        num_layers (int): The number of encoder layers.
        num_classes (int): The number of output classes.
        patch_size (int): The patch size to be used.
        in_channels (int): The number of input channels.
        hidden_size (int): The hidden dimension of the input tensor.
        num_attention_heads (int): The number of attention heads.
        num_key_value_heads (int): The number of key and value heads.
        intermediate_size (int): The intermediate size of the FFN layer.
        act_fn (nn.Module): The activation function to be used.
        dropout (float): The dropout probability. Defaults to 0.
    """

    def __init__(self,
                 image_size: int,
                 num_layers: int,
                 num_classes: int,
                 patch_size: int,
                 in_channels: int,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 intermediate_size: int,
                 act_fn: nn.Module = nn.SiLU(),
                 dropout: float = 0.):
        super().__init__()
        self.params = dict(image_size=image_size,
                           num_layers=num_layers,
                           num_classes=num_classes,
                           patch_size=patch_size,
                           in_channels=in_channels,
                           hidden_size=hidden_size,
                           num_attention_heads=num_attention_heads,
                           num_key_value_heads=num_key_value_heads,
                           intermediate_size=intermediate_size,
                           act_fn=act_fn,
                           dropout=dropout)

        self.seqlen = (image_size // patch_size)**2 + 1
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)

        head_dim = hidden_size // num_attention_heads
        self.rotary_pos_emb = RotaryEmbedding(head_dim)

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_attention_heads, num_key_value_heads,
                         intermediate_size, act_fn, dropout)
            for _ in range(num_layers)
        ])

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Vision Transformer model."""
        batch = hidden_states.shape[0]
        hidden_states = self.patch_embed(hidden_states)
        cls_token = self.cls_token.expand(batch, -1, -1)
        hidden_states = torch.cat([cls_token, hidden_states], dim=1)
        rotary_pos_emb = self.rotary_pos_emb(self.seqlen)

        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary_pos_emb)

        hidden_states = self.head(hidden_states[:, 0])
        return hidden_states
