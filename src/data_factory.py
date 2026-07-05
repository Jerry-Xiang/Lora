# -*- coding: utf-8 -*-
"""
数据工厂 - 负责数据加载和处理
"""

import os
import sys
from PIL import Image
import torch
import pandas as pd


class QwenVLDataset(torch.utils.data.Dataset):
    def __init__(self, conversations, processor, max_length=1024):
        self.conversations = conversations
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        image_path = conv["image_path"]
        messages = conv["messages"]
        
        image = Image.open(image_path).convert('RGB')
        
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
        
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
        if "image_sizes" in inputs and inputs["image_sizes"] is not None:
            result["image_sizes"] = inputs["image_sizes"].squeeze(0)
        
        return result


class QwenVLDataCollator:
    def __init__(self, processor, max_length=1024):
        self.processor = processor
        self.max_length = max_length
        self.tokenizer = processor.tokenizer
    
    def __call__(self, examples):
        input_ids = [example["input_ids"] for example in examples]
        attention_mask = [example["attention_mask"] for example in examples]
        pixel_values = [example["pixel_values"] for example in examples]
        labels = [example["labels"] for example in examples]
        
        padded = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        padded_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        padded_labels = padded_labels["input_ids"]
        padded_labels[padded_labels == self.tokenizer.pad_token_id] = -100
        
        pixel_values = torch.stack(pixel_values, dim=0)
        
        result = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "pixel_values": pixel_values,
            "labels": padded_labels,
        }
        
        if "image_grid_thw" in examples[0]:
            image_grid_thw = torch.stack([ex["image_grid_thw"] for ex in examples], dim=0)
            result["image_grid_thw"] = image_grid_thw
        if "image_sizes" in examples[0]:
            image_sizes = torch.stack([ex["image_sizes"] for ex in examples], dim=0)
            result["image_sizes"] = image_sizes
        
        return result


class DataFactory:
    def __init__(self, config, processor):
        self.config = config
        self.processor = processor
        self.train_dataset = None
        self.data_collator = None
    
    def load_data(self):
        print(f"\n" + "=" * 60)
        print("数据准备")
        print("=" * 60)
        
        if not os.path.exists(self.config.train_excel_path):
            print(f"错误：训练数据文件不存在: {self.config.train_excel_path}")
            sys.exit(1)
        print(f"训练数据: {self.config.train_excel_path}")
        
        print("加载训练数据...")
        try:
            train_df = pd.read_excel(self.config.train_excel_path)
            print(f"Excel 文件列名: {list(train_df.columns)}")
            print(f"数据集形状: {train_df.shape}")
        except Exception as e:
            print(f"读取 Excel 文件时出错: {e}")
            sys.exit(1)
        
        conversations = []
        for idx, row in train_df.iterrows():
            image_path = row.get("image", "")
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            
            if image_path and not os.path.isabs(image_path):
                image_path = os.path.join(self.config.base_dir, "data", image_path)
            
            if pd.notna(image_path) and os.path.exists(image_path):
                try:
                    Image.open(image_path).convert('RGB')
                    conversations.append({
                        "image_path": image_path,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image"}
                                ]
                            },
                            {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": response}
                                ]
                            }
                        ]
                    })
                    print(f"成功处理样本 {idx + 1}: {image_path}")
                except Exception as e:
                    print(f"处理图片 {image_path} 时出错: {e}")
        
        if len(conversations) == 0:
            print("没有有效的训练样本，程序退出")
            sys.exit(1)
        print(f"成功加载 {len(conversations)} 个训练样本")
        
        self._create_dataset(conversations)
    
    def _create_dataset(self, conversations):
        print("\n创建数据集和数据整理器...")
        
        self.train_dataset = QwenVLDataset(conversations, self.processor, max_length=self.config.max_seq_length)
        self.data_collator = QwenVLDataCollator(self.processor, max_length=self.config.max_seq_length)
        
        print(f"训练数据集大小: {len(self.train_dataset)}")
        print("数据集创建完成！")
