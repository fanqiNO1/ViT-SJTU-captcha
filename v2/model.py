import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim, mlp_dim):
        super(MLP, self).__init__()
        self.gate_proj = nn.Linear(dim, mlp_dim, bias=False)
        self.up_proj = nn.Linear(dim, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Attention(nn.Module):
    def __init__(self, dim, heads, kv_heads):
        super(Attention, self).__init__()
        self.head_dim = dim // heads
        self.kv_groups = heads // kv_heads
        self.scale = self.head_dim ** -0.5
    
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(heads * self.head_dim, dim, bias=True)
    
    def repeat_kv(self, k_or_v):
        b, kv_heads, n, head_dim = k_or_v.shape
        k_or_v = k_or_v[:, :, None, :, :].expand(b, kv_heads, self.kv_groups, n, head_dim)
        return k_or_v.reshape(b, kv_heads * self.kv_groups, n, head_dim)

    def forward(self, x):
        x_shape = x.shape[:-1]
        hidden_shape = (*x_shape, -1, self.head_dim)
        q = self.q_proj(x).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(x).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)
        k, v = self.repeat_kv(k), self.repeat_kv(v)
        attn = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(*x_shape, -1).contiguous()
        x = self.o_proj(x)
        return x
    

class RMSNorm(nn.Module):
    def __init__(self, dim, epsilon=1e-6):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.epsilon = epsilon

    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.epsilon)
        return self.weight * x


class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, kv_heads):
        super(EncoderLayer, self).__init__()
        self.input_norm = RMSNorm(dim)
        self.attn = Attention(dim, heads, kv_heads)
        self.mlp = MLP(dim, mlp_dim)
        self.output_norm = RMSNorm(dim)
    
    def forward(self, x):
        x = x + self.attn(self.input_norm(x))
        x = x + self.mlp(self.output_norm(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, channels, dim):
        super(PatchEmbed, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels

        self.proj = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ViTCaptcha(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, channels, dim, depth, heads, mlp_dim, kv_heads):
        super(ViTCaptcha, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(image_size, patch_size, channels, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.encoders = nn.ModuleList([
            EncoderLayer(dim, heads, mlp_dim, kv_heads) for _ in range(depth)
        ])

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        for encoder in self.encoders:
            x = encoder(x)
        x = x[:, 0]
        x = self.mlp_head(x)
        return x
