# -*- coding: utf-8 -*-
"""
模型工厂 - 负责模型下载、加载和LoRA配置
"""

import os
import sys
import psutil
import torch

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_JAX'] = '1'

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    TextStreamer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from modelscope import snapshot_download


class ModelFactory:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.processor = None
    
    def download_model(self):
        print(f"\n" + "=" * 60)
        print("使用 ModelScope 下载模型")
        print("=" * 60)
        
        model_dir = self._find_model_dir()
        if model_dir is not None:
            print(f"模型已存在，跳过下载: {model_dir}")
            self.model_dir = model_dir
            return
        
        print(f"正在从 ModelScope 下载模型: {self.config.model_id}")
        print(f"模型将保存到: {self.config.cache_dir}")
        
        try:
            model_dir = snapshot_download(
                self.config.model_id,
                cache_dir=self.config.cache_dir
            )
            print(f"模型已下载到: {model_dir}")
            self.model_dir = model_dir
        except Exception as e:
            print(f"ModelScope 下载失败: {e}")
            print("提示：请确保已安装 modelscope: pip install modelscope")
            sys.exit(1)
    
    def _find_model_dir(self):
        base_dir = self.config.cache_dir
        model_name = self.config.model_id.replace("/", os.sep)
        
        possible_paths = [
            os.path.join(base_dir, model_name),
            os.path.join(base_dir, self.config.model_id.split("/")[1]),
            os.path.join(base_dir, "qwen", "Qwen2___5-VL-3B-Instruct"),
            os.path.join(base_dir, "qwen", "Qwen2.5-VL-3B-Instruct"),
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "config.json")):
                return path
        return None
    
    def load_model(self):
        print(f"\n" + "=" * 60)
        print("使用 HuggingFace 加载模型")
        print("=" * 60)
        
        available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
        used_memory = psutil.virtual_memory().used / 1024 / 1024 / 1024
        print(f"当前可用内存: {available_memory:.2f} GB")
        print(f"当前已用内存: {used_memory:.2f} GB")
        
        if available_memory < 8 and not torch.cuda.is_available():
            print("警告：可用内存不足！")
            print("Qwen2.5-VL-3B 模型在 CPU 上加载需要约 10-12 GB 内存")
            print("建议：1. 关闭其他占用内存的程序")
            print("      2. 安装 GPU 和 CUDA")
            print("      3. 使用 4bit 量化加载（需要安装 bitsandbytes）")
        
        print("正在加载 Qwen2.5-VL-3B 模型...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True
            )
            print("Tokenizer 加载完成")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_dir,
                trust_remote_code=True
            )
            print("Processor 加载完成")
            
            if torch.cuda.is_available():
                print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
                dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
                torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_dir,
                    torch_dtype=torch_dtype,
                    device_map=self.config.device_map,
                    trust_remote_code=True
                )
            else:
                print("未检测到 GPU，使用 CPU（速度较慢）")

                bnb_loaded = False

                if self.config.use_4bit:
                    try:
                        from transformers import BitsAndBytesConfig

                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16
                        )

                        print("使用 4bit 量化加载模型（内存占用约 4-5 GB）")
                        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            self.model_dir,
                            quantization_config=bnb_config,
                            device_map="auto",
                            trust_remote_code=True
                        )
                        print("4bit 量化模型加载成功")
                        bnb_loaded = True
                    except ImportError:
                        print("bitsandbytes 未安装，回退到常规加载方式")
                    except Exception as e:
                        print(f"4bit 量化加载失败，回退到常规加载方式: {e}")

                if not bnb_loaded:
                    print("使用 float32 加载模型（CPU环境下更稳定）")
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_dir,
                        dtype=torch.float32,
                        device_map="cpu",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )

                print(f"模型加载后内存使用: {psutil.virtual_memory().used / 1024 / 1024 / 1024:.2f} GB")
                print("设置 tokenizer 参数...")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "right"
                print("tokenizer 参数设置完成")

                print("模型加载完成！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def inference_original_model(self):
        skip_pre_test = self.config.skip_pre_test
        
        if skip_pre_test:
            print("训练前推理测试已在配置中跳过")
            return None, None
        
        if not os.path.exists(self.config.test_image_path):
            print(f"警告：测试图片不存在: {self.config.test_image_path}")
            print("训练前推理测试将被跳过")
            return None, None
        
        print(f"\n" + "=" * 60)
        print("训练前推理测试（原始模型）")
        print("=" * 60)
        
        print("使用原始下载的 Qwen2.5-VL-3B 模型识别测试图片...")
        
        self.model.eval()
        
        from PIL import Image
        test_image = Image.open(self.config.test_image_path).convert('RGB')
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
        
        test_text = self.processor.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        test_inputs = self.processor(
            text=[test_text],
            images=[test_image],
            return_tensors="pt",
        ).to(self.config.device)
        
        text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        print("原始模型输出:")
        
        with torch.no_grad():
            _ = self.model.generate(
                **test_inputs,
                max_new_tokens=self.config.infer_max_new_tokens,
                use_cache=self.config.infer_use_cache,
                temperature=self.config.infer_temperature,
                min_p=self.config.infer_min_p,
                streamer=text_streamer,
            )
        
        return test_text, test_image
    
    def configure_lora(self):
        print(f"\n" + "=" * 60)
        print("配置 LoRA")
        print("=" * 60)
        
        lora_save_dir = self.config.lora_save_dir
        adapter_path = os.path.join(lora_save_dir, "adapter_model.safetensors")
        
        if self.config.skip_lora_config and os.path.exists(adapter_path):
            print(f"跳过 LoRA 配置，加载已有 adapter: {lora_save_dir}")
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, lora_save_dir)
                print("已有 LoRA adapter 加载完成！")
            except Exception as e:
                print(f"加载已有 adapter 失败，将重新配置 LoRA: {e}")
                self._setup_lora()
        else:
            self._setup_lora()
    
    def _setup_lora(self):
        print("正在配置 LoRA 参数...")
        lora_memory_usage = psutil.virtual_memory().used / 1024 / 1024 / 1024
        print(f"LoRA 配置前内存使用: {lora_memory_usage:.2f} GB")
        
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()
            print("已调用 enable_input_require_grads()")
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            print("已注册 make_inputs_require_grad 钩子")
        
        target_modules = self._discover_target_modules()
        print(f"LoRA 目标模块数量: {len(target_modules)}")
        if len(target_modules) == 0:
            print("警告：未找到任何目标模块！")
        for i, mod in enumerate(target_modules):
            if i < 10:
                print(f"  - {mod}")
        if len(target_modules) > 10:
            print(f"  ... 还有 {len(target_modules) - 10} 个模块")
        
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_CLS": TaskType.SEQ_CLS,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
            "QUESTION_ANS": TaskType.QUESTION_ANS,
            "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
        }
        lora_task_type = task_type_map.get(self.config.lora_task_type, TaskType.CAUSAL_LM)
        print(f"LoRA task_type: {lora_task_type}")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            task_type=lora_task_type,
            target_modules=target_modules,
            use_rslora=self.config.lora_use_rslora,
        )
        print(f"LoRA 配置参数: r={self.config.lora_r}, alpha={self.config.lora_alpha}, dropout={self.config.lora_dropout}")
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        after_lora_memory = psutil.virtual_memory().used / 1024 / 1024 / 1024
        print(f"LoRA 配置后内存使用: {after_lora_memory:.2f} GB")
        print("LoRA 配置完成！")
    
    def _discover_target_modules(self):
        target_modules = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                if 'language_model' in name and any(keyword in name for keyword in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
                    target_modules.append(name)
        
        return list(set(target_modules))
