"""
BTD项目工具模块
提供通用的工具函数和类
"""

from .path_utils import *
# 注意：paths.py 是 path_utils.py 的别名文件，避免循环导入
from .logger import *
from .data_converter import *
from .file_utils import *
from .performance_utils import *
from .data_yaml_generator import *

__version__ = "1.0.0"
__author__ = "BTD Team"

__all__ = [
    # 路径工具
    'ensure_dir',
    'get_project_root',
    'get_relative_path',
    'list_files_with_extension',
    
    # 日志工具
    'setup_logger',
    'get_logger',
    'log_function_call',
    
    # 数据转换工具
    'convert_coco_to_yolo',
    'convert_pascal_to_yolo',
    'split_dataset',
    'validate_annotations',
    
    # 文件工具
    'read_yaml',
    'write_yaml',
    'read_json',
    'write_json',
    'copy_files',
    'move_files',

    # 性能工具
    'time_it',
    'PerformanceProfiler',
    'benchmark_function',

    # data.yaml生成工具
    'DataYamlGenerator',

    # 数据转换中间层 (已移除data_utils相关功能)
]
