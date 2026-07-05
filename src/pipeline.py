# -*- coding: utf-8 -*-
"""
训练流水线 - 主控制器
基于 transformers+peft 架构的 LoRA 微调
"""

import os
import sys
import psutil
import torch

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_JAX'] = '1'


class TrainingPipeline:
    def __init__(self):
        self.config = None
        self.model_factory = None
        self.data_factory = None
        self.training_factory = None
        self.inference_factory = None
        self.test_text = None
        self.test_image = None
        self.training_completed = False
    
    def run(self):
        try:
            print("=" * 80)
            print("Qwen2-VL 3B 视觉模型微调 - 图片识别")
            print("=" * 80)
            print(f"开始执行时间: {self._get_current_time()}")
            print()
            
            print("\n" + "=" * 60)
            print("Step 1: 配置初始化")
            print("=" * 60)
            self._setup_config()
            print("Step 1: 配置初始化完成")
            print()
            
            print("\n" + "=" * 60)
            print("Step 2: 模型下载与加载")
            print("=" * 60)
            self._download_and_load_model()
            print("Step 2: 模型下载与加载完成")
            print()
            
            print("\n" + "=" * 60)
            print("Step 3: 训练前推理测试（原始模型）")
            print("=" * 60)
            self._pre_train_inference()
            print("Step 3: 训练前推理测试完成")
            print()
            
            print("\n" + "=" * 60)
            print("Step 4: LoRA 配置")
            print("=" * 60)
            self._configure_lora()
            print("Step 4: LoRA 配置完成")
            print()
            
            if not self.config.skip_data_preparation:
                print("\n" + "=" * 60)
                print("Step 5: 数据准备")
                print("=" * 60)
                self._setup_data()
                print("Step 5: 数据准备完成")
                print()
            
            print("\n" + "=" * 60)
            print("Step 6: 模型训练")
            print("=" * 60)
            self._train()
            if self.training_completed:
                print("Step 6: 模型训练完成")
            else:
                print("Step 6: 模型训练未完成或被跳过")
            print()
            
            if not self.config.skip_inference:
                print("\n" + "=" * 60)
                print("Step 7: 训练后推理测试")
                print("=" * 60)
                self._post_train_inference()
                print("Step 7: 训练后推理测试完成")
                print()
            
            print("\n" + "=" * 60)
            print("Step 8: 模型保存")
            print("=" * 60)
            self._save_model()
            print("Step 8: 模型保存完成")
            print()
            
            print("=" * 80)
            print("所有步骤执行完毕")
            print(f"结束执行时间: {self._get_current_time()}")
            print("=" * 80)
        except Exception as e:
            print(f"\n[严重错误] 流水线执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_current_time(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _print_header(self):
        print("=" * 60)
        print("Qwen2-VL 3B 视觉模型微调 - 图片识别")
        print("=" * 60)
    
    def _setup_config(self):
        from .config_factory import ConfigFactory
        self.config = ConfigFactory()
        self.config.create_directories()
        self.config.print_config()
    
    def _download_and_load_model(self):
        from .model_factory import ModelFactory
        self.model_factory = ModelFactory(self.config)
        self.model_factory.download_model()
        self.model_factory.load_model()
    
    def _pre_train_inference(self):
        self.test_text, self.test_image = self.model_factory.inference_original_model()
    
    def _configure_lora(self):
        self.model_factory.configure_lora()
    
    def _setup_data(self):
        from .data_factory import DataFactory
        self.data_factory = DataFactory(self.config, self.model_factory.processor)
        self.data_factory.load_data()
    
    def _train(self):
        if self.config.skip_training and self.config.skip_data_preparation:
            lora_save_dir = self.config.lora_save_dir
            adapter_path = os.path.join(lora_save_dir, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                print(f"\n跳过训练和数据准备，使用已有模型: {lora_save_dir}")
                return
        
        if self.config.skip_data_preparation:
            print("警告：skip_data_preparation=True 但 skip_training=False，需要数据准备")
            print("将执行数据准备...")
            self._setup_data()
        
        try:
            from .training_factory import TrainingFactory
            
            self.training_factory = TrainingFactory(
                self.config,
                self.model_factory.model,
                self.data_factory.train_dataset,
                self.data_factory.data_collator,
                self.model_factory.tokenizer,
                self.model_factory.processor
            )
            
            self.training_factory.setup_trainer()
            self.training_factory.train()
            
            self.training_completed = self.training_factory.training_successful
            
        except Exception as e:
            print(f"\n训练过程发生错误: {e}")
            import traceback
            traceback.print_exc()
            self.training_completed = False
    
    def _post_train_inference(self):
        if self.test_text is None or self.test_image is None:
            print("\n" + "=" * 60)
            print("训练后推理测试（微调后模型）")
            print("=" * 60)
            print("训练前推理未执行，重新加载测试图片...")
            
            from PIL import Image
            test_image_path = self.config.test_image_path
            
            if not os.path.exists(test_image_path):
                print(f"警告：测试图片不存在: {test_image_path}")
                print("训练后推理测试将被跳过")
                return
            
            self.test_image = Image.open(test_image_path).convert('RGB')
            
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
            
            self.test_text = self.model_factory.processor.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        from .inference_factory import InferenceFactory
        self.inference_factory = InferenceFactory(
            self.config,
            self.model_factory.model,
            self.model_factory.tokenizer,
            self.model_factory.processor
        )
        self.inference_factory.post_train_inference(self.test_text, self.test_image)
    
    def _save_model(self):
        if self.training_factory is None:
            print("\n" + "=" * 60)
            print("训练已跳过，无需保存模型")
            print("=" * 60)
            return
        
        if not self.training_completed:
            print("\n" + "=" * 60)
            print("训练未完成，跳过模型保存")
            print("=" * 60)
            return
        
        self.training_factory.save_model()
