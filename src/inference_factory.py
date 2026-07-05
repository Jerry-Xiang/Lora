# -*- coding: utf-8 -*-
"""
推理工厂 - 负责微调后模型推理测试
"""

import torch
from transformers import TextStreamer


class InferenceFactory:
    def __init__(self, config, model, tokenizer, processor):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
    
    def post_train_inference(self, test_text, test_image):
        if test_text is None or test_image is None:
            return
        
        print(f"\n" + "=" * 60)
        print("训练后推理测试（微调后模型）")
        print("=" * 60)
        
        print("使用微调后的模型识别测试图片...")
        
        self.model.eval()
        
        test_inputs_eval = self.processor(
            text=[test_text],
            images=[test_image],
            return_tensors="pt",
        ).to(self.config.device)
        
        text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        print("微调后模型输出:")
        with torch.no_grad():
            _ = self.model.generate(
                **test_inputs_eval,
                max_new_tokens=self.config.infer_max_new_tokens,
                use_cache=self.config.infer_use_cache,
                temperature=self.config.infer_temperature,
                min_p=self.config.infer_min_p,
                streamer=text_streamer,
            )
