from transformers import AutoTokenizer
import json
import os
import torch
#### llama3.2 from scratch

# 读取并查看模型参数
with open("./config.json", "r") as f:
    config = json.load(f)
print(config)

# 读取模型权重
from safetensors import safe_open
weights_root = "/HOME/scz0101/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
# 文件路径
file1 = os.path.join(weights_root, "model-00001-of-00002.safetensors")
file2 = os.path.join(weights_root, "model-00002-of-00002.safetensors")

# 加载第一个文件
with safe_open(file1, framework="pt", device="cpu") as f:
    state_dict1 = {key: f.get_tensor(key) for key in f.keys()}
# 查看key和size
print(json.dumps(list(state_dict1.keys()), indent=4)) ## layer0~20
print("model.embed_tokens.weight", state_dict1["model.embed_tokens.weight"].shape) # vob_size*hidden_size
print("model.layers.0.input_layernorm.weight", state_dict1["model.layers.0.input_layernorm.weight"].shape) #hidden_size
print("model.layers.0.mlp.down_proj.weight", state_dict1["model.layers.0.mlp.down_proj.weight"].shape) #hidden_size * intermediate_size
print("model.layers.0.mlp.gate_proj.weight", state_dict1["model.layers.0.mlp.gate_proj.weight"].shape) # intermediate_size * hidden_size 
print("model.layers.0.mlp.up_proj.weight", state_dict1["model.layers.0.mlp.up_proj.weight"].shape) #intermediate_size * hidden_size 
print("model.layers.0.post_attention_layernorm.weight",state_dict1["model.layers.0.post_attention_layernorm.weight"].shape) #hidden_size
print("model.layers.0.self_attn.k_proj.weight",state_dict1["model.layers.0.self_attn.k_proj.weight"].shape) #(head_dim*num_key_value_heads)*hidden_size
print("model.layers.0.self_attn.o_proj.weight",state_dict1["model.layers.0.self_attn.o_proj.weight"].shape) #(head_dim*num_attention_heads)*hidden_size
print("model.layers.0.self_attn.q_proj.weight",state_dict1["model.layers.0.self_attn.q_proj.weight"].shape) #(head_dim*num_attention_heads)*hidden_size
print("model.layers.0.self_attn.v_proj.weight",state_dict1["model.layers.0.self_attn.v_proj.weight"].shape) #(head_dim*num_key_value_heads)*hidden_size

# # 加载第二个文件
# with safe_open(file2, framework="pt", device="cpu") as f:
#     state_dict2 = {key: f.get_tensor(key) for key in f.keys()}

# # 合并两个状态字典
# state_dict = {**state_dict1, **state_dict2}

# # 查看模型的权重key
# print(json.dumps(list(state_dict2.keys()), indent=4)) ## layer21~27

#先跳过tokenizer，可以直接使用autotokenizer，后续再来实现

# model_name_or_path = "meta-llama/Llama-3.2-3B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token

#开始创建embeding_layer

embedding_layer = torch.nn.Embedding(config['vocab_size'], config["hidden_size"])
embedding_layer.weight.data.copy_(state_dict1["model.embed_tokens.weight"])