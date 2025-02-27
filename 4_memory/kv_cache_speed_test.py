import torch
import torch.nn as nn
import time
from dataclasses import dataclass

# 基础配置类
@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 2
    n_heads: int = 2
    vocab_size: int = 32000
    max_seq_len: int = 2048
    use_kv_cache: bool = False  # 新增KV缓存标志

# 旋转位置编码（RoPE）
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, pos):
    dim = q.shape[-1]
    freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)).to(q.device)
    sin = torch.sin(pos * freq)
    cos = torch.cos(pos * freq)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

# 基础组件
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x):
        return self.scale * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4)
        self.w2 = nn.Linear(dim * 4, dim)
        self.w3 = nn.Linear(dim, dim * 4)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

# 自注意力模块（基础版不带KV Cache）
class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        
        self.wq = nn.Linear(self.dim, self.dim)
        self.wk = nn.Linear(self.dim, self.dim)
        self.wv = nn.Linear(self.dim, self.dim)
        self.wo = nn.Linear(self.dim, self.dim)

    def forward(self, x, pos):
        B, T, _ = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)
        
        # 应用RoPE
        q, k = apply_rope(q, k, pos)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        attn = attn.softmax(dim=-1)
        output = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.wo(output)

# 带KV Cache的注意力模块
class AttentionWithCache(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        
        self.wq = nn.Linear(self.dim, self.dim)
        self.wk = nn.Linear(self.dim, self.dim)
        self.wv = nn.Linear(self.dim, self.dim)
        self.wo = nn.Linear(self.dim, self.dim)
        
        # 初始化KV缓存
        self.register_buffer("k_cache", torch.zeros(args.max_seq_len, self.n_heads, self.head_dim))
        self.register_buffer("v_cache", torch.zeros(args.max_seq_len, self.n_heads, self.head_dim))

    def forward(self, x, pos, cache_pos=0):
        B, T, _ = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)
        
        # 更新缓存
        self.k_cache[cache_pos:cache_pos+T] = k.squeeze(0)
        self.v_cache[cache_pos:cache_pos+T] = v.squeeze(0)
        
        # 应用RoPE到当前窗口
        q, k_rot = apply_rope(q, self.k_cache[:cache_pos+T].unsqueeze(0), pos)
        
        # 注意力计算
        attn = (q @ k_rot.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        attn = attn.softmax(dim=-1)
        output = (attn @ self.v_cache[:cache_pos+T].unsqueeze(0)).transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.wo(output)

# LLaMA模型（支持双模式）
class LLaMA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList()
        self.norm = RMSNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size)
        
        for _ in range(args.n_layers):
            layer = nn.ModuleDict({
                'attention': AttentionWithCache(args) if args.use_kv_cache else Attention(args),
                'feed_forward': FeedForward(args.dim),
                'attention_norm': RMSNorm(args.dim),
                'ffn_norm': RMSNorm(args.dim),
            })
            self.layers.append(layer)

    def forward(self, tokens, start_pos=0):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        for layer in self.layers:
            h = h + layer.attention(
                layer.attention_norm(h),
                pos=start_pos + torch.arange(seqlen, device=tokens.device)
            )
            h = h + layer.feed_forward(layer.ffn_norm(h))
        return self.output(self.norm(h))

# 速度对比测试函数
def speed_test(model_class, seq_length=100, use_cache=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = ModelArgs(use_kv_cache=use_cache)
    model = model_class(args).to(device)
    input_ids = torch.randint(0, args.vocab_size, (1, 1), device=device)
    
    # Warmup
    for _ in range(10):
        _ = model(input_ids)
    
    # 正式测试
    start_time = time.time()
    for _ in range(seq_length):
        with torch.no_grad():
            output = model(input_ids, start_pos=0)
        input_ids = torch.cat([input_ids, output.argmax(-1)], dim=1)
    elapsed = time.time() - start_time
    
    return f"{'带KV缓存' if use_cache else '无缓存'}版本生成{seq_length}个token耗时: {elapsed:.2f}秒"

# 测试结果
print("速度对比测试（基于NVIDIA A100 40GB）")
print("----------------------------------")
print(speed_test(LLaMA, use_cache=False))
print(speed_test(LLaMA, use_cache=True))
