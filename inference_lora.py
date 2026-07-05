# -*- coding: utf-8 -*-
"""
LoRA增量推理脚本 - 使用已有LoRA adapter进行推理
无需重新训练，直接加载基座模型和LoRA适配器进行推理
"""

import os
import sys
import psutil
from PIL import Image
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    TextStreamer,
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
    data_config = config.get("data", {})
    inference_config = config.get("inference", {})
    
    return {
        "base_dir": base_dir,
        "cache_dir": paths.get("cache_dir", os.path.join(base_dir, "models")),
        "lora_save_dir": paths.get("lora_save_dir", os.path.join(base_dir, "car_insurance_lora_model")),
        "model_id": model_config.get("model_id", "qwen/Qwen2.5-VL-3B-Instruct"),
        "torch_dtype": model_config.get("torch_dtype", "float16"),
        "device_map": model_config.get("device_map", "auto"),
        "test_image_path": data_config.get("test_image_path", os.path.join(base_dir, "data", "images", "1-vehicle-odometer-reading.jpg")),
        "max_new_tokens": inference_config.get("max_new_tokens", 256),
        "temperature": float(inference_config.get("temperature", 0.7)),
        "min_p": float(inference_config.get("min_p", 0.1)),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
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
    print("LoRA增量推理 - 使用已有适配器进行推理")
    print("=" * 60)
    
    config = load_config()
    print(f"运行设备: {config['device']}")
    
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
    print(f"可用内存: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")
    
    try:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("已使用 4bit 量化加载模型")
    except ImportError:
        print("bitsandbytes 未安装，使用 float16 加载...")
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
    
    try:
        model = PeftModel.from_pretrained(model, load_path)
    except Exception as e:
        print(f"加载 LoRA 适配器失败: {e}")
        print("尝试使用 adapter_name_or_path 参数...")
        model = PeftModel.from_pretrained(model, load_path, adapter_name_or_path=load_path)
    
    model.eval()
    
    print("\n加载测试图片...")
    if not os.path.exists(config["test_image_path"]):
        print(f"错误：测试图片不存在: {config['test_image_path']}")
        sys.exit(1)
    
    test_image = Image.open(config["test_image_path"]).convert('RGB')
    
    test_instruction = """你是一名图片识别专家。这里有一张车辆里程表的图片。请从中提取以下关键信息，并按照指定格式输出：

1. **总里程**：从图片中读取总里程数，单位为公里
2. **当前速度**：从图片中读取当前速度，单位为公里/小时
3. **当前时间**：从图片中读取当前时间
4. **当前温度**：从图片中读取当前温度
5. **当前挡位**：从图片中读取当前挡位（如停车挡P、前进挡D、倒车挡R等）

请严格按照以下格式输出，每行一个信息：
- 总里程：XXX公里
- 当前速度：XX公里/小时
- 当前时间：XX:XX
- 当前温度：XX°C
- 当前挡位：XXX"""
    
    test_messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": test_instruction}
        ]}
    ]
    
    test_text = processor.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    test_inputs = processor(
        text=[test_text],
        images=[test_image],
        return_tensors="pt",
    ).to(config["device"])
    
    print("\n" + "=" * 60)
    print("LoRA增量推理结果:")
    print("=" * 60)
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    with torch.no_grad():
        _ = model.generate(
            **test_inputs,
            max_new_tokens=config["max_new_tokens"],
            use_cache=True,
            temperature=config["temperature"],
            min_p=config["min_p"],
            streamer=text_streamer,
        )
    
    print("\n" + "=" * 60)
    print("推理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
