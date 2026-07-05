# -*- coding: utf-8 -*-
"""
配置工厂 - 负责加载和管理所有配置参数
"""

import os
import yaml


class ConfigFactory:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config.yaml")
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = config_path
        self._config = self._load_config()
        
        self._init_paths()
        self._init_model_config()
        self._init_data_config()
        self._init_training_config()
        self._init_lora_config()
        self._init_inference_config()
        self._init_workflow_config()
    
    def _load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}
    
    def _init_paths(self):
        model_config = self._config.get("model", {})
        training_config = self._config.get("training", {})
        
        self.cache_dir = os.path.join(self.base_dir, model_config.get("cache_dir", "models"))
        self.output_dir = os.path.join(self.base_dir, training_config.get("output_dir", "outputs"))
        self.lora_save_dir = os.path.join(self.base_dir, training_config.get("lora_save_dir", "car_insurance_lora_model"))
    
    def _init_model_config(self):
        model_config = self._config.get("model", {})
        self.model_id = model_config.get("model_id", "qwen/Qwen2.5-VL-3B-Instruct")
        self.torch_dtype = model_config.get("torch_dtype", "float16")
        self.device_map = model_config.get("device_map", "auto")
        self.use_4bit = model_config.get("use_4bit", False)
    
    def _init_data_config(self):
        data_config = self._config.get("data", {})
        self.train_excel_path = os.path.join(self.base_dir, data_config.get("train_excel", "data/qwen-vl-train.xlsx"))
        self.test_image_path = os.path.join(self.base_dir, data_config.get("test_image", "data/images/1-vehicle-odometer-reading.jpg"))
        self.max_seq_length = data_config.get("max_seq_length", 1280)
    
    def _init_training_config(self):
        training_config = self._config.get("training", {})
        self.per_device_train_batch_size = training_config.get("per_device_train_batch_size", 1)
        self.gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 4)
        self.warmup_steps = training_config.get("warmup_steps", 5)
        self.max_steps = training_config.get("max_steps", 30)
        self.learning_rate = float(training_config.get("learning_rate", 2e-4))
        self.logging_steps = training_config.get("logging_steps", 1)
        self.optim = training_config.get("optim", "adamw_torch")
        self.weight_decay = float(training_config.get("weight_decay", 0.01))
        self.lr_scheduler_type = training_config.get("lr_scheduler_type", "linear")
        self.seed = training_config.get("seed", 3407)
        self.report_to = training_config.get("report_to", "none")
        self.remove_unused_columns = training_config.get("remove_unused_columns", False)
        self.fp16 = training_config.get("fp16", True)
        self.bf16 = training_config.get("bf16", False)
        self.gradient_checkpointing = training_config.get("gradient_checkpointing", True)
        self.save_strategy = training_config.get("save_strategy", "steps")
        self.save_steps = training_config.get("save_steps", 10)
        self.save_total_limit = training_config.get("save_total_limit", 3)
    
    def _init_lora_config(self):
        lora_config = self._config.get("lora", {})
        self.lora_r = lora_config.get("r", 16)
        self.lora_alpha = lora_config.get("alpha", 16)
        self.lora_dropout = float(lora_config.get("dropout", 0.0))
        self.lora_bias = lora_config.get("bias", "none")
        self.lora_task_type = lora_config.get("task_type", "CAUSAL_LM")
        self.lora_use_rslora = lora_config.get("use_rslora", False)
    
    def _init_inference_config(self):
        inference_config = self._config.get("inference", {})
        self.infer_max_new_tokens = inference_config.get("max_new_tokens", 256)
        self.infer_use_cache = inference_config.get("use_cache", True)
        self.infer_temperature = float(inference_config.get("temperature", 0.7))
        self.infer_min_p = float(inference_config.get("min_p", 0.1))
        self.skip_pre_test = inference_config.get("skip_pre_test", False)
    
    def _init_workflow_config(self):
        workflow_config = self._config.get("workflow", {})
        self.skip_lora_config = workflow_config.get("skip_lora_config", False)
        self.skip_training = workflow_config.get("skip_training", False)
        self.skip_data_preparation = workflow_config.get("skip_data_preparation", False)
        self.skip_inference = workflow_config.get("skip_inference", False)
    
    def create_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.lora_save_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @property
    def device(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def print_config(self):
        print(f"项目根目录: {self.base_dir}")
        print(f"运行设备: {self.device}")
        print(f"输出目录: {self.output_dir}")
        print(f"LoRA保存目录: {self.lora_save_dir}")
        print(f"模型缓存目录: {self.cache_dir}")
        print("配置加载完成！")
