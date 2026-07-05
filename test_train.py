# -*- coding: utf-8 -*-
"""
Qwen2-VL 3B LoRA微调测试脚本
基于transformers+peft架构
使用float32加载模型（更稳定）
"""

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_JAX'] = '1'

import sys
import torch
import psutil

print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"可用内存: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")

model_dir = "models/qwen/Qwen2___5-VL-3B-Instruct"


print("\n" + "=" * 60)
print("Step 1: 加载Tokenizer")
print("=" * 60)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"Tokenizer加载完成，内存: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")


print("\n" + "=" * 60)
print("Step 2: 加载Processor")
print("=" * 60)
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
print(f"Processor加载完成，内存: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")


print("\n" + "=" * 60)
print("Step 3: 加载模型 (float32)")
print("=" * 60)
from transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
print(f"模型加载成功！内存: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")


print("\n" + "=" * 60)
print("Step 4: 配置LoRA")
print("=" * 60)
from peft import LoraConfig, get_peft_model, TaskType

model.enable_input_require_grads()

target_modules = []
for name, module in model.named_modules():
    if hasattr(module, 'weight'):
        if 'language_model' in name and any(keyword in name for keyword in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            target_modules.append(name)

print(f"目标模块数量: {len(target_modules)}")

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print(f"LoRA配置完成！内存: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")


print("\n" + "=" * 60)
print("Step 5: 加载数据")
print("=" * 60)
from PIL import Image
import pandas as pd

train_df = pd.read_excel("data/qwen-vl-train.xlsx")
conversations = []

for idx, row in train_df.iterrows():
    image_path = row.get("image", "")
    prompt = row.get("prompt", "")
    response = row.get("response", "")
    
    if not os.path.isabs(image_path):
        image_path = os.path.join("data", image_path)
    
    if os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
        conversations.append({
            "image": image,
            "prompt": prompt,
            "response": response
        })

print(f"加载了 {len(conversations)} 个训练样本")


print("\n" + "=" * 60)
print("Step 6: 准备训练数据")
print("=" * 60)
all_inputs = []
for conv in conversations:
    text = processor.apply_chat_template([
        {"role": "user", "content": [
            {"type": "text", "text": conv["prompt"]},
            {"type": "image"}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": conv["response"]}
        ]}
    ], tokenize=False, add_generation_prompt=False)

    inputs = processor(
        text=[text],
        images=[conv["image"]],
        return_tensors="pt",
    )
    
    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    item = {
        "input_ids": inputs["input_ids"].squeeze(0),
        "attention_mask": inputs["attention_mask"].squeeze(0),
        "pixel_values": inputs["pixel_values"].squeeze(0),
        "labels": labels.squeeze(0),
    }
    
    if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
        item["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
    
    all_inputs.append(item)

print(f"训练数据准备完成！内存: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")


print("\n" + "=" * 60)
print("Step 7: 开始训练")
print("=" * 60)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5)

model.train()
max_steps = 3

for step in range(max_steps):
    print(f"\nStep {step + 1}/{max_steps}")
    
    batch = all_inputs[step % len(all_inputs)]
    
    optimizer.zero_grad()
    
    model_kwargs = {
        "input_ids": batch["input_ids"].unsqueeze(0),
        "attention_mask": batch["attention_mask"].unsqueeze(0),
        "pixel_values": batch["pixel_values"].unsqueeze(0),
        "labels": batch["labels"].unsqueeze(0),
    }
    if "image_grid_thw" in batch:
        model_kwargs["image_grid_thw"] = batch["image_grid_thw"].unsqueeze(0)
    
    outputs = model(**model_kwargs)
    loss = outputs.loss
    print(f"loss: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()
    scheduler.step()


print("\n" + "=" * 60)
print("Step 8: 保存模型")
print("=" * 60)
model.save_pretrained("car_insurance_lora_model")
tokenizer.save_pretrained("car_insurance_lora_model")
processor.save_pretrained("car_insurance_lora_model")
print("模型保存成功！")


print("\n" + "=" * 60)
print("训练完成！")
print("=" * 60)
