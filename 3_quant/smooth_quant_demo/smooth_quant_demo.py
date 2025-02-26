'''
# SmoothQuant on Llama 2 7B

In this notebook, we use Llama-2-7B model to demonstrate SmoothQuant can use 8-bit for both weights and activations to achieve the similar perplexity as FP16 models.


1. 安装pytorch、transformers、Acceleatte
2. 安装smoothquant`cd srcs/`然后`python setup.py install`

'''

import os
import copy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_llama_like
import tqdm

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))

def selective_release(target_model):
    # 阶段1：解除引用
    del target_model
    # # 强制垃圾回收
    # gc.collect()
    torch.cuda.empty_cache()
    
    # 阶段3：验证释放结果
    print(f"当前显存占用：{torch.cuda.memory_allocated()/1024**3:.2f}GB")


from datasets import load_dataset
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
evaluator = Evaluator(dataset, tokenizer, "cuda")


model_fp16 = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.float16, device_map="auto"
)

ppl_fp16 = evaluator.evaluate(model_fp16)
print(f"Original model (fp16) perplexity: {ppl_fp16}")

## Naive W8A8 Quantized Model Perplexity
model_w8a8 = quantize_llama_like(model_fp16)
print(model_w8a8)
ppl_w8a8 = evaluator.evaluate(model_w8a8)
print(f"Naive W8A8 quantized model perplexity: {ppl_w8a8}")

## SmoothQuant W8A8 Quantized Model Perplexity
model_ori= LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.float16, device_map="auto"
)
act_scales = torch.load("./act_scales/llama_3.2_3b.pt")

for alpha in [0.5, 0.6, 0.7, 0.8]:
    model = copy.deepcopy(model_ori)
    smooth_lm(model, act_scales, alpha)
    model_smoothquant_w8a8 = quantize_llama_like(model)
    print(model_smoothquant_w8a8)

    ppl_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
    print(f"SmoothQuant W8A8 quantized with aplha {alpha} model perplexity: {ppl_smoothquant_w8a8}")
    selective_release(model)

'''

Original model (fp16) perplexity: 10.612541198730469
Naive W8A8 quantized model perplexity: 10.581432342529297
SmoothQuant W8A8 quantized with aplha 0.5 model perplexity: 10.594319343566895
SmoothQuant W8A8 quantized with aplha 0.6 model perplexity: 10.593501091003418
SmoothQuant W8A8 quantized with aplha 0.7 model perplexity: 10.585674285888672
SmoothQuant W8A8 quantized with aplha 0.8 model perplexity: 10.570932388305664
'''