"""
BTD项目自动化脚本模块
提供训练、验证、推理、数据处理等自动化脚本
"""

from .train import *
from .validate import *
from .inference import *
from .data_processing import *

__version__ = "1.0.0"
__author__ = "BTD Team"

__all__ = [
    # 训练相关
    'train_model',
    'resume_training',
    'export_model',
    
    # 验证相关
    'validate_model',
    'evaluate_metrics',
    
    # 推理相关
    'predict_image',
    'predict_batch',
    'predict_video',
    
    # 数据处理相关
    'process_raw_data',
    'convert_annotations',
    'split_dataset',
    'validate_dataset'
]
