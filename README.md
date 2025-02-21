# model_acceleration
项目介绍：大模型加速方法有很多，每个都去看的话，需要看很多工程，也不方便横向对比。那么能否用一个开源模型（比如llama3.2），将所有的加速方法都实现一遍，所有方法的实现都从头开始实现。

## 开发计划
### 1. llama3.2实现
- [ ] weights使用[llama3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
### 2. benchmark实现
- [ ] speed test (目标平台：cuda)
- [ ] memory test
- [ ] ppl
- [ ] task_eval
### 3. 量化方法实现
- [ ] naive quant（W8A8/W4A4）
- [ ] smooth quant
- [ ] awq
- [ ] kv cache量化
### 4. 内存优化方法实现
- [ ] kv cache
- [ ] paged attention
- [ ] flash attention
- [ ] MQA/GQA/MLA
### 5. 微调技术
- [ ] Lora
- [ ] QLora
- [ ] AdaLora
- [ ] fp8/fp4/nf4低显存微调
- [ ] spinquant
- [ ] (Optional) Prompt Tuning
- [ ] (Optional) Prefix Tuning
- [ ] (Optional) P-Tuning
### 6. 并行训练优化demo
- [ ] 数据并行
- [ ] 模型并行
- [ ] 流水线并行
- [ ] 张量并行
- [ ] 序列并行
- [ ] 多维混合并行
- [ ] 自动并行
- [ ] MOE
### 7. 算子优化
- [ ] 算子融合
- [ ] 高性能算子（deep seek的fp8_gemm）
- [ ] 子图融合和替换
### 8. 编译优化
- [ ] LLVM
- [ ] MLIR
- [ ] TVM
### 9. 服务级优化
- [ ] Continous Batching
- [ ] 动态批处理
- [ ] 异步tokenizer/detokenizer
### 10. 蒸馏
- [ ] Deepseek-R1 [蒸馏](https://github.com/agentica-project/deepscaler) 


## 环境配置

## 参考资料
- [ ] 入门：https://transformers.run/