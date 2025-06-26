"""
日志工具模块
提供统一的日志记录功能
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import functools
import time

def setup_logger(name: str = "BTD", 
                level: Union[str, int] = logging.INFO,
                log_file: Optional[Union[str, Path]] = None,
                console_output: bool = True,
                file_output: bool = True) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径，如果为None则自动生成
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if console_output:
        # 在Windows上设置控制台编码
        if sys.platform.startswith('win'):
            import os
            os.system('chcp 65001 > nul')

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        # 设置控制台编码
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8')
            except:
                pass

        logger.addHandler(console_handler)
    
    # 文件输出
    if file_output:
        if log_file is None:
            # 自动生成日志文件路径
            from .path_utils import get_project_root, ensure_dir
            log_dir = get_project_root() / 'logs'
            ensure_dir(log_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{name}_{timestamp}.log"
        
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "BTD") -> logging.Logger:
    """
    获取已配置的日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger: 日志记录器
    """
    logger = logging.getLogger(name)
    
    # 如果日志记录器没有处理器，则进行默认配置
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger

def log_function_call(logger: Optional[logging.Logger] = None):
    """
    装饰器：记录函数调用信息
    
    Args:
        logger: 日志记录器，如果为None则使用默认记录器
        
    Returns:
        装饰器函数
    """
    if logger is None:
        logger = get_logger()
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            module_name = func.__module__
            
            # 记录函数开始执行
            start_time = time.time()
            logger.info(f"开始执行函数: {module_name}.{func_name}")
            
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 记录函数执行成功
                end_time = time.time()
                execution_time = end_time - start_time
                logger.info(f"函数执行成功: {module_name}.{func_name}, 耗时: {execution_time:.2f}秒")
                
                return result
                
            except Exception as e:
                # 记录函数执行失败
                end_time = time.time()
                execution_time = end_time - start_time
                logger.error(f"函数执行失败: {module_name}.{func_name}, 耗时: {execution_time:.2f}秒, 错误: {str(e)}")
                raise
        
        return wrapper
    return decorator

class ProgressLogger:
    """进度日志记录器"""
    
    def __init__(self, total: int, logger: Optional[logging.Logger] = None, 
                 log_interval: int = 10):
        """
        初始化进度日志记录器
        
        Args:
            total: 总数量
            logger: 日志记录器
            log_interval: 日志记录间隔（百分比）
        """
        self.total = total
        self.current = 0
        self.logger = logger or get_logger()
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_log_percent = 0
    
    def update(self, count: int = 1):
        """
        更新进度
        
        Args:
            count: 增加的数量
        """
        self.current += count
        percent = (self.current / self.total) * 100
        
        # 检查是否需要记录日志
        if percent - self.last_log_percent >= self.log_interval or self.current == self.total:
            elapsed_time = time.time() - self.start_time
            
            if self.current == self.total:
                self.logger.info(f"进度: 100% ({self.current}/{self.total}), 总耗时: {elapsed_time:.2f}秒")
            else:
                # 估算剩余时间
                if self.current > 0:
                    eta = (elapsed_time / self.current) * (self.total - self.current)
                    self.logger.info(f"进度: {percent:.1f}% ({self.current}/{self.total}), "
                                   f"已耗时: {elapsed_time:.2f}秒, 预计剩余: {eta:.2f}秒")
                else:
                    self.logger.info(f"进度: {percent:.1f}% ({self.current}/{self.total})")
            
            self.last_log_percent = percent

class TimerLogger:
    """计时日志记录器"""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """
        初始化计时日志记录器
        
        Args:
            name: 计时器名称
            logger: 日志记录器
        """
        self.name = name
        self.logger = logger or get_logger()
        self.start_time = None
    
    def __enter__(self):
        """进入上下文管理器"""
        self.start_time = time.time()
        self.logger.info(f"开始计时: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            if exc_type is None:
                self.logger.info(f"计时结束: {self.name}, 耗时: {elapsed_time:.2f}秒")
            else:
                self.logger.error(f"计时结束(异常): {self.name}, 耗时: {elapsed_time:.2f}秒, "
                                f"异常: {exc_type.__name__}: {exc_val}")

def log_system_info(logger: Optional[logging.Logger] = None):
    """
    记录系统信息
    
    Args:
        logger: 日志记录器
    """
    if logger is None:
        logger = get_logger()
    
    import platform

    logger.info("=" * 50)
    logger.info("系统信息")
    logger.info("=" * 50)
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info(f"Python版本: {platform.python_version()}")

    # 尝试获取详细系统信息
    try:
        import psutil
        logger.info(f"CPU核心数: {psutil.cpu_count()}")
        logger.info(f"内存总量: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        logger.info(f"可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    except ImportError:
        logger.warning("psutil 未安装，无法获取详细系统信息")
        logger.info("安装命令: pip install psutil")
    
    # GPU信息
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.2f} GB")
        else:
            logger.info("CUDA不可用")
    except ImportError:
        logger.info("PyTorch未安装")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    # 基本功能演示
    logger = setup_logger("logger_demo")
    logger.info("日志模块加载成功")

    # 记录系统信息
    try:
        log_system_info(logger)
    except ImportError as e:
        logger.warning(f"无法记录完整系统信息: {e}")
