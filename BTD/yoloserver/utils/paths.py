#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :paths.py
# @Time      :2025/6/24 17:00:00
# @Author    :BTD Team
# @Project   :BTD
# @Function  :路径管理模块别名文件，确保与文档要求的命名一致性

"""
路径管理模块别名文件

为了保持与文档要求的一致性，此文件作为path_utils.py的别名。
所有功能都从path_utils.py导入，确保向后兼容性。

使用方式:
    from paths import get_project_root, get_data_paths
    或
    from path_utils import get_project_root, get_data_paths
    
两种导入方式都是等效的。
"""

# 直接导入path_utils模块，避免循环导入
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

# 导入path_utils中的函数
try:
    from yoloserver.utils.path_utils import get_project_root, get_data_paths, ensure_dir
except ImportError:
    # 如果导入失败，直接从当前目录导入
    import importlib.util
    spec = importlib.util.spec_from_file_location("path_utils", Path(__file__).parent / "path_utils.py")
    path_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(path_utils)
    get_project_root = path_utils.get_project_root
    get_data_paths = path_utils.get_data_paths
    ensure_dir = path_utils.ensure_dir

# 获取项目根目录和数据路径
_project_root = get_project_root()
_data_paths = get_data_paths()

# 导出常用路径常量（与文档示例保持一致）
PROJECT_ROOT = _project_root
DATA_ROOT = _data_paths['data_root']
RAW_DATA_DIR = _data_paths['raw_data']
RAW_IMAGES_DIR = _data_paths['raw_images']
RAW_LABELS_DIR = _data_paths['raw_annotations']
YOLO_STAGED_LABELS_DIR = _data_paths['yolo_labels']
TRAIN_IMAGES_DIR = _data_paths['train_images']
TRAIN_LABELS_DIR = _data_paths['train_labels']
VAL_IMAGES_DIR = _data_paths['val_images']
VAL_LABELS_DIR = _data_paths['val_labels']
TEST_IMAGES_DIR = _data_paths['test_images']
TEST_LABELS_DIR = _data_paths['test_labels']
MODELS_DIR = _project_root / 'yoloserver' / 'models'
CONFIGS_DIR = _project_root / 'yoloserver' / 'configs'
LOGS_DIR = _project_root / 'logs'

# 为了与文档中的导入语句兼容
__all__ = [
    # 从path_utils导入的所有函数
    'get_project_root',
    'get_data_paths',
    'get_config_paths',
    'get_model_paths',
    'ensure_dir',
    'create_project_structure',
    'validate_project_structure',
    'get_relative_path',
    'is_project_initialized',
    
    # 路径常量
    'PROJECT_ROOT',
    'DATA_ROOT',
    'RAW_DATA_DIR',
    'RAW_IMAGES_DIR',
    'RAW_LABELS_DIR',
    'YOLO_STAGED_LABELS_DIR',
    'TRAIN_IMAGES_DIR',
    'TRAIN_LABELS_DIR',
    'VAL_IMAGES_DIR',
    'VAL_LABELS_DIR',
    'TEST_IMAGES_DIR',
    'TEST_LABELS_DIR',
    'MODELS_DIR',
    'CONFIGS_DIR',
    'LOGS_DIR'
]

if __name__ == "__main__":
    # 测试路径常量
    print("BTD项目路径信息:")
    print("=" * 50)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据根目录: {DATA_ROOT}")
    print(f"原始数据目录: {RAW_DATA_DIR}")
    print(f"原始图像目录: {RAW_IMAGES_DIR}")
    print(f"原始标签目录: {RAW_LABELS_DIR}")
    print(f"YOLO标签目录: {YOLO_STAGED_LABELS_DIR}")
    print(f"训练图像目录: {TRAIN_IMAGES_DIR}")
    print(f"训练标签目录: {TRAIN_LABELS_DIR}")
    print(f"验证图像目录: {VAL_IMAGES_DIR}")
    print(f"验证标签目录: {VAL_LABELS_DIR}")
    print(f"测试图像目录: {TEST_IMAGES_DIR}")
    print(f"测试标签目录: {TEST_LABELS_DIR}")
    print(f"模型目录: {MODELS_DIR}")
    print(f"配置目录: {CONFIGS_DIR}")
    print(f"日志目录: {LOGS_DIR}")
    print("=" * 50)
    
    # 检查项目结构
    print("检查项目结构...")
    key_dirs = [DATA_ROOT, RAW_DATA_DIR, RAW_IMAGES_DIR, MODELS_DIR, CONFIGS_DIR]
    all_exist = all(path.exists() for path in key_dirs)
    if all_exist:
        print("✅ 主要目录结构存在")
    else:
        print("⚠️ 部分目录不存在，可能需要初始化项目")
    
    print("\npaths.py模块测试完成")
