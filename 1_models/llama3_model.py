from transformers import AutoTokenizer
from typing import Callable, List, Optional, Tuple, Union
import json
import os
import torch
from torch import nn
import math
import torch.nn.functional as F
from kernels.llama3_kernel import apply_rotary_pos_emb, get_inv_freq_llama3, sdpa_attention_forward

with open("/HOME/scz0101/run/model_acceleration/models/config.json", "r") as f:
    config = json.load(f)
# 创建rms_norm层，tensor2维，对每一行做归一化

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"

        self.rope_type = config['rope_scaling']['rope_type']
        self.max_seq_len_cached = config['max_position_embeddings']
        self.original_max_seq_len = config['max_position_embeddings']

        self.config = config
        self.rope_init_fn = get_inv_freq_llama3

        inv_freq, self.attention_scaling = self.rope_init_fn(device, config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False): #enabled=False表示在此上下文中禁用混合精度，所有计算将使用默认的浮点精度（通常是FP32）
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config["mlp_bias"])
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config["mlp_bias"])
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config["mlp_bias"])
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config["head_dim"]
        self.num_key_value_groups = config["num_attention_heads"] // config["num_key_value_heads"]
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config['attention_dropout']
        self.is_causal = True

        self.q_proj = nn.Linear(
            config['hidden_size'], config['num_attention_heads'] * self.head_dim, bias=config['attention_bias']
        )
        self.k_proj = nn.Linear(
            config['hidden_size'], config['num_key_value_heads'] * self.head_dim, bias=config['attention_bias']
        )
        self.v_proj = nn.Linear(
            config['hidden_size'], config['num_key_value_heads'] * self.head_dim, bias=config['attention_bias']
        )
        self.o_proj = nn.Linear(
            config['num_attention_heads'] * self.head_dim, config['hidden_size'], bias=config['attention_bias']
        )

        ## add kv cache
        max_batch_size = 1
        max_seq_len = 512
        self.register_buffer("k_cache", torch.zeros(max_batch_size, max_seq_len, config['num_key_value_heads'], self.head_dim), persistent=False)
        self.register_buffer("v_cache", torch.zeros(max_batch_size, max_seq_len, config['num_key_value_heads'], self.head_dim), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output, attn_weights = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
class LlamaDecoderLayer(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.hidden_size = config["hidden_size"]

        self.self_attn = LlamaAttention(layer_idx=layer_idx)

        self.mlp = LlamaMLP()
        self.input_layernorm = LlamaRMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.post_attention_layernorm = LlamaRMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None # necessary, but kept here for BC
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        return outputs
    
class LlamaModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self):
        super().__init__()
        # self.padding_idx = config['pad_token_id']
        self.vocab_size = config['vocab_size']

        self.embed_tokens = nn.Embedding(self.vocab_size, config['hidden_size'])
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(layer_idx) for layer_idx in range(config['num_hidden_layers'])]
        )
        self.norm = LlamaRMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.rotary_emb = LlamaRotaryEmbedding()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None
    ):

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # 此处没有使用kv cache,因此每次都需要计算整个score矩阵 ,因此需要一个完整的mask
        attention_mask = torch.full((position_ids.shape[1], position_ids.shape[1]), float("-inf"), device=inputs_embeds.device).triu_(1)

        for decoder_layer in self.layers[: config['num_hidden_layers']]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return hidden_states

class LlamaForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LlamaModel()
        self.vocab_size = config['vocab_size']
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0
    ):

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids
        )

        hidden_states = outputs
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return logits

def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)

    res = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

    return res

@torch.inference_mode()
def generate(
    model: LlamaForCausalLM,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens

if __name__ == "__main__":
    from safetensors.torch import load_file
    model = LlamaForCausalLM()
    print("create model succ!")

    # load weghts
    weights_root = "/HOME/scz0101/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
    file1 = os.path.join(weights_root, "model-00001-of-00002.safetensors")
    file2 = os.path.join(weights_root, "model-00002-of-00002.safetensors")
    model_part1 = load_file(file1)
    model_part2 = load_file(file2)
    model_state_dict = {**model_part1, **model_part2}

    print(model.state_dict()["lm_head.weight"].shape)
    print(model_state_dict["model.embed_tokens.weight"].shape)
    for weight_name in  model.state_dict().keys():
        if weight_name != "lm_head.weight":
            model.state_dict()[weight_name].copy_(model_state_dict[weight_name])
        else:
            model.state_dict()[weight_name].copy_(model_state_dict["model.embed_tokens.weight"])

    print("load weights succ!")

    # run  model  once
    # tokenizer先用transfomer自带的 
    model_name_or_path = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    input_text = "how are you? I'm"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    past_seen_tokens = 0
    temperature = 0
    # max_new_tokens = 512
    position_ids = torch.arange(past_seen_tokens,past_seen_tokens+inputs["input_ids"].shape[1]).unsqueeze(0)
    
    logits = model.forward(inputs["input_ids"], position_ids, logits_to_keep=1)
    # logits = model.forward(torch.tensor([[128000, 5269, 527, 499, 30]], dtype=torch.int32), position_ids, logits_to_keep=1)

    if temperature > 0:
        next_token = sample(logits, temperature)
    else:
        next_token = logits.argmax(dim=-1)
    print(next_token)
    response = tokenizer.decode(next_token[0], skip_special_tokens=True)
    print(response)

    # completion_tokens = generate(model, inputs["input_ids"], max_new_tokens, tokenizer.eos_token_id, temperature)
    # completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
    # print(completion)


