# -*- coding: utf-8 -*-
"""
LoRA与基座合并权重脚本 - 将LoRA适配器权重合并到基座模型中
生成独立的完整模型，便于部署和推理
"""

import os
import sys
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)
from peft import PeftModel
from modelscope import snapshot_download


def load_config():
    import yaml
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    paths = config.get("paths", {})
    model_config = config.get("model", {})
    
    return {
        "base_dir": base_dir,
        "cache_dir": paths.get("cache_dir", os.path.join(base_dir, "models")),
        "lora_save_dir": paths.get("lora_save_dir", os.path.join(base_dir, "car_insurance_lora_model")),
        "merged_save_dir": os.path.join(base_dir, "car_insurance_merged_model"),
        "model_id": model_config.get("model_id", "qwen/Qwen2.5-VL-3B-Instruct"),
        "torch_dtype": model_config.get("torch_dtype", "float16"),
        "device_map": model_config.get("device_map", "auto"),
    }


def find_model_dir(config):
    base_dir = config["cache_dir"]
    model_name = config["model_id"].replace("/", os.sep)
    
    possible_paths = [
        os.path.join(base_dir, model_name),
        os.path.join(base_dir, config["model_id"].split("/")[1]),
        os.path.join(base_dir, "qwen", "Qwen2___5-VL-3B-Instruct"),
        os.path.join(base_dir, "qwen", "Qwen2.5-VL-3B-Instruct"),
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, "config.json")):
            return path
    return None


def main():
    print("=" * 60)
    print("LoRA与基座合并权重")
    print("=" * 60)
    
    config = load_config()
    
    adapter_path = os.path.join(config["lora_save_dir"], "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        print(f"错误：未找到 LoRA 适配器: {adapter_path}")
        print("请先运行训练脚本生成LoRA适配器")
        sys.exit(1)
    
    print(f"\n加载基座模型: {config['model_id']}")
    model_dir = find_model_dir(config)
    
    if model_dir is None:
        print("模型未找到，正在从ModelScope下载...")
        model_dir = snapshot_download(
            config["model_id"],
            cache_dir=config["cache_dir"]
        )
    
    print(f"模型目录: {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    
    print("加载基座模型...")
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(config["torch_dtype"], torch.float16)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True
    )
    
    print(f"加载 LoRA 适配器: {config['lora_save_dir']}")
    
    adapter_path = os.path.join(config["lora_save_dir"], "adapter_model.safetensors")
    load_path = config["lora_save_dir"]
    
    try:
        import safetensors.torch
        weights = safetensors.torch.load_file(adapter_path)
        
        first_key = list(weights.keys())[0]
        print(f"权重键格式: {first_key}")
        
        if "base_model.model.model." in first_key:
            print("检测到多层包装的权重键，正在修复...")
            new_weights = {}
            for old_key, value in weights.items():
                new_key = old_key.replace("base_model.model.model.", "base_model.model.")
                new_weights[new_key] = value
            
            import shutil
            temp_dir = os.path.join(config["lora_save_dir"], "fixed_adapter")
            os.makedirs(temp_dir, exist_ok=True)
            
            safetensors.torch.save_file(new_weights, os.path.join(temp_dir, "adapter_model.safetensors"))
            
            import json
            adapter_config_path = os.path.join(config["lora_save_dir"], "adapter_config.json")
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            
            adapter_config["base_model_name_or_path"] = model_dir
            
            with open(os.path.join(temp_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(adapter_config, f, indent=2, ensure_ascii=False)
            
            load_path = temp_dir
            print(f"权重键已修复，加载路径: {load_path}")
        
        else:
            import json
            adapter_config_path = os.path.join(config["lora_save_dir"], "adapter_config.json")
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            
            adapter_config["base_model_name_or_path"] = model_dir
            
            with open(adapter_config_path, "w", encoding="utf-8") as f:
                json.dump(adapter_config, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"权重修复跳过: {e}")
    
    model = PeftModel.from_pretrained(model, load_path)
    
    print("\n合并 LoRA 权重到基座模型...")
    model = model.merge_and_unload()
    
    print(f"\n保存合并后的模型到: {config['merged_save_dir']}")
    os.makedirs(config["merged_save_dir"], exist_ok=True)
    
    model.save_pretrained(
        config["merged_save_dir"],
        safe_serialization=True,
    )
    tokenizer.save_pretrained(config["merged_save_dir"])
    processor.save_pretrained(config["merged_save_dir"])
    
    print("\n" + "=" * 60)
    print("合并完成!")
    print(f"合并后的模型已保存到: {config['merged_save_dir']}")
    print("\n合并后的模型可直接用于推理，无需加载LoRA适配器")
    print("=" * 60)


if __name__ == "__main__":
    main()
