# -*- coding: utf-8 -*-
"""
日志记录器 - 同时输出到终端和日志文件，捕获 stdout 和 stderr
参照 training_log_20260701_104922.txt 格式
"""

import os
import sys
from datetime import datetime


class Logger:
    def __init__(self):
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(logs_dir, f"training_log_{timestamp}.txt")
        self.log_handle = open(self.log_file, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal_stdout.write(message)
        self.log_handle.write(message)
        self.log_handle.flush()
    
    def flush(self):
        self.terminal_stdout.flush()
        self.log_handle.flush()
    
    def close(self):
        sys.stdout = self.terminal_stdout
        sys.stderr = self.terminal_stderr
        if not self.log_handle.closed:
            self.log_handle.close()