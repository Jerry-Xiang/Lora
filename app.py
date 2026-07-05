# -*- coding: utf-8 -*-
"""
Qwen2-VL 3B 视觉模型微调 - 图片识别
主入口文件 - 通过工厂化设计模式逐步调用各个模块
"""

import os
import sys

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_JAX'] = '1'


def main():
    print("=" * 80)
    print("Qwen2-VL 3B 视觉模型微调 - 图片识别")
    print("=" * 80)
    print(f"Python: {sys.executable}")
    print(f"版本: {sys.version}")
    print()
    
    try:
        from src.logger import Logger
        
        logger = Logger()
        sys.stdout = logger
        sys.stderr = logger
        
        try:
            from src.pipeline import TrainingPipeline
            
            pipeline = TrainingPipeline()
            pipeline.run()
            
        except Exception as e:
            sys.stdout = logger.terminal_stdout
            sys.stderr = logger.terminal_stderr
            print(f"\n程序执行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = logger.terminal_stdout
            sys.stderr = logger.terminal_stderr
            logger.close()
            print(f"\n日志已保存到: {logger.log_file}")
            
    except ImportError as e:
        print(f"\n无法导入模块: {e}")
        print("请确保已安装所有依赖：")
        print("pip install transformers peft torch pillow pandas openpyxl modelscope")
    except Exception as e:
        print(f"\n程序启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()