from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_name_or_path = "meta-llama/Llama-3.2-3B-Instruct"
# 从预训练的模型中获取tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)

# 将模型转移到GPU上，如果有的话
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义一个聊天历史列表
chat_history = []

# 模拟用户输入
user_input = "Who are you?"

# 将用户输入添加到聊天历史
chat_history.append({"role": "user", "content": user_input})

# 将聊天历史转换为适合tokenizer的格式
# 这里我们假设每个消息都是独立的，并且我们只关心最后一条消息
input_text = chat_history[-1]["content"]

# 使用模型生成回复
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
print(inputs)

# 生成回复
generated_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=512
)

# 解码生成的回复
response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 将模型的回复添加到聊天历史
chat_history.append({"role": "assistant", "content": response})

# 打印聊天历史
for message in chat_history:
    print(f"{message['role']}: {message['content']}")