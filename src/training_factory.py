# -*- coding: utf-8 -*-
"""
训练工厂 - 负责模型训练和保存
使用 transformers+peft 架构进行 LoRA 微调
使用手动训练循环（更稳定）
"""

import os
import sys
import psutil
import torch


class TrainingFactory:
    def __init__(self, config, model, train_dataset=None, data_collator=None, tokenizer=None, processor=None):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.processor = processor
        self.training_successful = False
    
    def setup_trainer(self):
        if self.config.skip_training:
            lora_save_dir = self.config.lora_save_dir
            adapter_path = os.path.join(lora_save_dir, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                print(f"\n跳过训练，使用已有模型: {lora_save_dir}")
                return
            else:
                print(f"skip_training=True 但未找到已有模型，将执行训练...")
        
        print(f"\n验证训练环境...")
        self._validate_environment()
        print("训练环境验证完成")
    
    def _validate_environment(self):
        if self.train_dataset is None:
            print("错误：训练数据集为空！")
            sys.exit(1)
        print(f"训练数据集大小: {len(self.train_dataset)}")
        
        if self.tokenizer is None:
            print("警告：Tokenizer 为空")
        
        available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
        print(f"可用内存: {available_memory:.2f} GB")
    
    def train(self):
        if self.config.skip_training:
            lora_save_dir = self.config.lora_save_dir
            adapter_path = os.path.join(lora_save_dir, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                print("跳过训练步骤")
                return
        
        if self.train_dataset is None:
            print("错误：训练数据集未初始化！")
            return
        
        print(f"\n开始训练...")
        print(f"训练前内存使用: {psutil.virtual_memory().used / 1024 / 1024 / 1024:.2f} GB")
        
        try:
            max_steps = self.config.max_steps
            total_steps = max_steps
            warmup_steps = self.config.warmup_steps
            decay_steps = total_steps - warmup_steps
            
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.config.learning_rate, 
                weight_decay=self.config.weight_decay
            )
            
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.01, 
                end_factor=1.0, 
                total_iters=warmup_steps
            )
            decay_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=1.0, 
                end_factor=0.01, 
                total_iters=decay_steps
            )
            
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps]
            )
            
            self.model.train()
            
            total_loss = 0.0
            gradient_accumulation_steps = self.config.gradient_accumulation_steps
            
            for step in range(max_steps):
                batch = self.train_dataset[step % len(self.train_dataset)]
                
                if step % gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                
                model_kwargs = {
                    "input_ids": batch["input_ids"].unsqueeze(0),
                    "attention_mask": batch["attention_mask"].unsqueeze(0),
                    "pixel_values": batch["pixel_values"].unsqueeze(0),
                    "labels": batch["labels"].unsqueeze(0),
                }
                if "image_grid_thw" in batch and batch["image_grid_thw"] is not None:
                    model_kwargs["image_grid_thw"] = batch["image_grid_thw"].unsqueeze(0)
                
                outputs = self.model(**model_kwargs)
                loss = outputs.loss / gradient_accumulation_steps
                
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                
                total_loss += loss.item() * gradient_accumulation_steps
                
                lr = optimizer.param_groups[0]['lr']
                epoch = step + 1.0
                
                log_dict = {
                    'loss': round(loss.item() * gradient_accumulation_steps, 4),
                    'grad_norm': float(grad_norm) if (step + 1) % gradient_accumulation_steps == 0 else 0.0,
                    'learning_rate': lr,
                    'epoch': epoch
                }
                print(log_dict)
            
            avg_loss = total_loss / max_steps
            
            print({
                'train_runtime': 0,
                'train_samples_per_second': 0,
                'train_steps_per_second': 0,
                'train_loss': avg_loss,
                'epoch': float(max_steps)
            })
            
            print(f"\n训练后内存使用: {psutil.virtual_memory().used / 1024 / 1024 / 1024:.2f} GB")
            print("训练完成！")
            self.training_successful = True
            
        except KeyboardInterrupt:
            print("\n训练被用户中断")
            self.training_successful = False
        except Exception as e:
            print(f"\n训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            print(f"错误发生时内存使用: {psutil.virtual_memory().used / 1024 / 1024 / 1024:.2f} GB")
            self.training_successful = False
    
    def save_model(self):
        if not self.training_successful:
            print("\n训练未成功完成，跳过模型保存")
            return
        
        print(f"\n保存 LoRA 适配器...")
        
        try:
            self.model.save_pretrained(self.config.lora_save_dir)
            print(f"模型保存成功: {self.config.lora_save_dir}")
        except Exception as e:
            print(f"模型保存失败: {e}")
            return
        
        if self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(self.config.lora_save_dir)
                print(f"Tokenizer 保存成功")
            except Exception as e:
                print(f"Tokenizer 保存失败: {e}")
        
        if self.processor is not None:
            try:
                self.processor.save_pretrained(self.config.lora_save_dir)
                print(f"Processor 保存成功")
            except Exception as e:
                print(f"Processor 保存失败: {e}")
        
        print(f"\n训练完成! 模型已保存到 {self.config.lora_save_dir} 目录")
