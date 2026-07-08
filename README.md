# Qwen2-VL 3B LoRA 视觉语言模型微调项目

基于阿里巴巴达摩院 Qwen2.5-VL-3B-Instruct 模型，实现车辆里程表信息提取的 LoRA 微调项目。

## 项目结构

```
Case_lora/
├── app.py                    # 主入口文件
├── config.yaml               # 配置文件
├── requirements.txt          # 依赖包列表
├── ARCHITECTURE.md           # 架构文档
├── test_train.py             # 独立训练测试脚本
├── inference_lora.py         # LoRA增量推理脚本
├── merge_lora.py             # LoRA与基座合并权重脚本
├── data/                     # 数据目录
│   ├── qwen-vl-train.xlsx    # 训练数据
│   └── images/               # 训练图片
├── models/                   # 模型缓存目录
├── outputs/                  # 训练输出目录
├── car_insurance_lora_model/ # LoRA适配器保存目录
├── logs/                     # 日志目录
└── src/                      # 源码目录
    ├── __init__.py
    ├── logger.py             # 日志记录器
    ├── config_factory.py     # 配置工厂
    ├── model_factory.py      # 模型工厂
    ├── data_factory.py       # 数据工厂
    ├── training_factory.py   # 训练工厂
    ├── inference_factory.py  # 推理工厂
    └── pipeline.py           # 训练流水线（主控制器）
```

## 技术栈

| 分类 | 技术 | 版本 |
|------|------|------|
| 语言 | Python | 3.13.12 |
| 框架 | PyTorch | 2.12.1+cpu |
| 模型库 | Transformers | 4.57.6 |
| LoRA | PEFT | 0.19.1 |
| 分布式 | Accelerate | 1.13.0 |
| 量化 | bitsandbytes | 0.49.2 |
| 模型下载 | ModelScope | 1.36.3 |
| 数据处理 | Pandas | 2.3.3 |
| 图片处理 | Pillow | 11.3.0 |
| 内存监控 | psutil | 7.2.2 |
| 配置解析 | PyYAML | 6.0.3 |

## 环境要求

- Python 3.13+
- Windows/Linux 操作系统
- 推荐 16GB+ 内存（CPU训练）
- CUDA 12.x（可选，GPU训练）

## 安装步骤

1. **安装依赖**

```bash
pip install -r requirements.txt
```

2. **配置环境变量**

```bash
# Windows PowerShell
$env:TRANSFORMERS_NO_TF = '1'
$env:TRANSFORMERS_NO_JAX = '1'
```

3. **首次运行**

首次运行会自动从 ModelScope 下载 Qwen2.5-VL-3B-Instruct 模型（约 12GB）。

## 快速开始

### 训练模型

```bash
python app.py
```

训练流程包含以下步骤：
1. 配置初始化
2. 模型下载与加载（float32）
3. 训练前推理测试
4. LoRA 配置（348个目标模块）
5. 数据准备
6. 模型训练（AdamW + SequentialLR）
7. 训练后推理测试
8. 保存模型

### 独立测试

```bash
python test_train.py
```

使用独立脚本验证训练流程，不经过工厂化框架。

### LoRA 推理

```bash
python inference_lora.py
```

使用训练好的 LoRA 适配器进行推理。

### 合并模型

```bash
python merge_lora.py
```

将 LoRA 适配器与基座模型合并为完整模型。

## 配置说明

主要配置项（config.yaml）：

```yaml
model:
  model_id: "qwen/Qwen2.5-VL-3B-Instruct"
  cache_dir: "models"
  use_4bit: false

lora:
  r: 8
  alpha: 8
  dropout: 0.0
  bias: "none"

training:
  max_steps: 60
  learning_rate: 1.0e-4
  warmup_steps: 5
  gradient_accumulation_steps: 1
  weight_decay: 0.01

workflow:
  skip_training: false
  skip_inference: false
```

## LoRA 配置

- **目标模块**：348个（包含 q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj）
- **可训练参数**：约 3700 万（占总参数的 0.98%）
- **使用 GQA**：分组查询注意力，减少内存占用

## 日志说明

训练日志保存在 `logs/` 目录下，命名格式为 `training_log_YYYYMMDD_HHMMSS.txt`。

**重要说明**：请特别关注以下两个日志文件作为最终执行结果参考：
1. `training_log_YYYYMMDD_HHMMSS.txt` - 完整训练过程日志
2. `training_log_YYYYMMDD_HHMMSS.txt` - 最终训练统计信息

日志格式示例：

```
============================================================
Step 6: 模型训练
============================================================
开始训练...
{'loss': 11.8852, 'grad_norm': 2.4085, 'learning_rate': 4.16e-05, 'epoch': 1.0}
{'loss': 15.3215, 'grad_norm': 3.3485, 'learning_rate': 8.12e-05, 'epoch': 2.0}
...
{'train_runtime': 0, 'train_samples_per_second': 0, 'train_steps_per_second': 0, 'train_loss': 8.2479, 'epoch': 60.0}
```

## 常见问题

### 内存不足

1. 关闭其他占用内存的程序
2. 启用 4bit 量化（设置 `use_4bit: true`）
3. 增加 Windows 页面文件（虚拟内存）

### DLL 冲突

Windows 环境下可能出现 TensorFlow 与 PyTorch 的 DLL 冲突：
- 设置环境变量 `TRANSFORMERS_NO_TF=1` 和 `TRANSFORMERS_NO_JAX=1`
- 卸载 TensorFlow（如果不需要）

### NumPy 版本冲突

PyTorch 与 NumPy 2.x 存在兼容性问题：
- 使用 `numpy<2.0` 版本（已在 requirements.txt 中配置）

## 项目架构

项目采用工厂化设计模式，包含以下核心模块：

| 模块 | 文件 | 功能 |
|------|------|------|
| 配置工厂 | config_factory.py | 加载和管理配置参数 |
| 模型工厂 | model_factory.py | 模型下载、加载、LoRA配置 |
| 数据工厂 | data_factory.py | 训练数据加载和预处理 |
| 训练工厂 | training_factory.py | 训练循环和模型保存 |
| 推理工厂 | inference_factory.py | 训练前后推理测试 |
| 流水线 | pipeline.py | 协调各工厂模块执行 |
| 日志记录 | logger.py | 同时输出到终端和日志文件 |

详细架构说明请参考 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 许可证

本项目基于 Apache 2.0 许可证。

```
Copyright 2026 Your Name

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## 项目信息

**项目名称**：Qwen2-VL 3B LoRA 视觉语言模型微调项目  
**作者**：[您的姓名]  
**版本**：1.0.0  
**创建时间**：2026年7月  
**项目类型**：人工智能/深度学习/自然语言处理/计算机视觉  

**项目特点**：
- 基于阿里巴巴达摩院 Qwen2.5-VL-3B 模型
- 实现 LoRA 低秩自适应微调
- 支持车辆里程表信息提取
- 工厂化设计模式，模块化架构
- 完整的训练、推理、合并流程
- 详细的日志记录和错误处理

**应用场景**：
- 车辆信息识别系统
- 图像文本联合识别
- 视觉语言模型微调研究
- 自动化数据标注工具

## 致谢

- Qwen2.5-VL 模型由阿里巴巴达摩院提供
- Hugging Face Transformers 和 PEFT 库
- ModelScope 模型下载平台