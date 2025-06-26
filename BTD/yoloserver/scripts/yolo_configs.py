#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @FileName  :yolo_configs.py
# @Time      :2025/6/26 09:00:00
# @Author    :BTD Team
# @Project   :BTD
# @Function  :YOLO模型即用型配置文件，提供训练、验证、推理的默认参数

import sys
import torch
from pathlib import Path
from typing import Dict, Any

# 添加项目路径支持
current_file = Path(__file__).resolve()
if current_file.parent.name == "scripts":
    # 如果在scripts目录中
    project_root = current_file.parent.parent.parent  # BTD/yoloserver/scripts -> project_root
    btd_root = project_root / "BTD"
else:
    # 如果在项目根目录中
    project_root = current_file.parent
    btd_root = project_root / "BTD"

sys.path.insert(0, str(btd_root))

# 导入路径工具
try:
    from yoloserver.utils.path_utils import get_project_root
    _project_root = get_project_root()
    RUNS_DIR = _project_root / "BTD" / "yoloserver" / "runs"
except ImportError:
    # 备用方案
    RUNS_DIR = btd_root / "yoloserver" / "runs"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 默认训练配置 - 针对脑瘤检测优化
# ============================================================================

DEFAULT_TRAIN_CONFIG = {
    # 基本参数
    'data': 'data.yaml',
    'epochs': 100,  # 脑瘤检测建议100-300轮
    'time': None,   # 不限制训练时间
    'batch': 16,    # 根据GPU显存调整
    'imgsz': 640,   # 脑瘤图像建议640或更高
    'device': "0" if torch.cuda.is_available() else "cpu",
    'workers': 8,

    # 训练控制
    'patience': 50,     # 早停耐心值
    'save': True,       # 保存模型
    'save_period': -1,  # 不定期保存
    'cache': False,     # 不缓存数据
    'resume': False,    # 不恢复训练
    'amp': True,        # 混合精度训练

    # 项目设置
    'project': str(RUNS_DIR / 'train'),
    'name': 'brain_tumor_detection',
    'exist_ok': False,

    # 模型配置
    'pretrained': True,
    'optimizer': 'AdamW',  # 适合医学图像
    'seed': 42,
    'deterministic': True,
    'single_cls': False,
    'classes': None,
    'rect': False,
    'cos_lr': True,        # 余弦学习率调度
    'multi_scale': False,

    # 损失权重 - 针对医学图像调优
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'pose': 12.0,
    'kobj': 1.0,

    # 学习率参数
    'lr0': 0.001,      # 初始学习率
    'lrf': 0.01,       # 最终学习率比例
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,

    # 数据增强 - 适合医学图像
    'hsv_h': 0.015,    # 色相增强
    'hsv_s': 0.3,      # 饱和度增强（医学图像较保守）
    'hsv_v': 0.4,      # 明度增强
    'degrees': 0.0,    # 不旋转（医学图像方向重要）
    'translate': 0.1,  # 轻微平移
    'scale': 0.3,      # 缩放范围
    'shear': 0.0,      # 不剪切
    'perspective': 0.0, # 不透视变换
    'flipud': 0.0,     # 不垂直翻转
    'fliplr': 0.5,     # 水平翻转
    'bgr': 0.0,
    'mosaic': 0.8,     # 马赛克增强（医学图像适中）
    'mixup': 0.0,      # 不使用mixup
    'cutmix': 0.0,
    'copy_paste': 0.0,
    'auto_augment': 'randaugment',
    'erasing': 0.2,    # 随机擦除（较保守）

    # 特殊参数
    'close_mosaic': 10,
    'nbs': 64,
    'overlap_mask': True,
    'mask_ratio': 4,
    'dropout': 0.0,
    'val': True,
    'plots': True,
    'profile': False,
    'freeze': None,
    'fraction': 1.0
}

# ============================================================================
# 默认验证配置
# ============================================================================

DEFAULT_VAL_CONFIG = {
    'data': 'data.yaml',
    'imgsz': 640,
    'batch': 16,
    'save_json': False,
    'conf': 0.25,      # 置信度阈值
    'iou': 0.7,        # NMS IoU阈值
    'max_det': 300,    # 最大检测数
    'half': True,      # 半精度推理
    'device': '0' if torch.cuda.is_available() else 'cpu',
    'dnn': False,
    'plots': True,
    'classes': None,
    'rect': True,
    'split': 'val',
    'project': str(RUNS_DIR / 'val'),
    'name': 'brain_tumor_val',
    'verbose': False,
    'save_txt': True,
    'save_conf': True,
    'save_crop': True,
    'workers': 8,
    'augment': False,
    'agnostic_nms': False,
    'single_cls': False
}

# ============================================================================
# 默认推理配置
# ============================================================================

DEFAULT_INFER_CONFIG = {
    # 基本参数
    'source': '0',     # 数据源
    'device': '0' if torch.cuda.is_available() else 'cpu',
    'imgsz': 640,
    'batch': 1,

    # 模型推理
    'conf': 0.25,      # 置信度阈值
    'iou': 0.7,        # NMS IoU阈值
    'max_det': 300,
    'classes': None,
    'agnostic_nms': False,
    'augment': False,
    'half': False,
    'stream_buffer': False,
    'vid_stride': 1,
    'retina_masks': False,

    # 保存与项目
    'project': str(RUNS_DIR / 'predict'),
    'name': 'brain_tumor_predict',
    'save': False,
    'save_frames': False,
    'save_txt': False,
    'save_conf': False,
    'save_crop': False,
    'stream': False,

    # 可视化参数
    'show': False,
    'show_labels': True,
    'show_conf': True,
    'show_boxes': True,
    'line_width': 3,
    'visualize': False,
    'verbose': True,
}

# ============================================================================
# 配置管理和互补功能
# ============================================================================

def get_config_by_name(config_name: str) -> Dict[str, Any]:
    """根据名称获取配置"""
    configs = {
        'train': DEFAULT_TRAIN_CONFIG,
        'val': DEFAULT_VAL_CONFIG,
        'infer': DEFAULT_INFER_CONFIG,
        'inference': DEFAULT_INFER_CONFIG,
        'validation': DEFAULT_VAL_CONFIG,
        'training': DEFAULT_TRAIN_CONFIG
    }
    
    if config_name not in configs:
        raise ValueError(f"未知配置名称: {config_name}. 可用配置: {list(configs.keys())}")
    
    return configs[config_name].copy()


def create_custom_config(config_type: str, **kwargs) -> Dict[str, Any]:
    """创建自定义配置"""
    base_config = get_config_by_name(config_type)
    base_config.update(kwargs)
    return base_config


def validate_config_compatibility(config: Dict[str, Any], config_type: str) -> bool:
    """验证配置兼容性"""
    required_keys = {
        'train': ['data', 'epochs', 'batch', 'imgsz'],
        'val': ['data', 'imgsz', 'batch'],
        'infer': ['source', 'imgsz']
    }
    
    if config_type in required_keys:
        for key in required_keys[config_type]:
            if key not in config:
                return False
    
    return True


def get_all_configs() -> Dict[str, Dict[str, Any]]:
    """获取所有可用配置"""
    return {
        'train': DEFAULT_TRAIN_CONFIG.copy(),
        'val': DEFAULT_VAL_CONFIG.copy(),
        'infer': DEFAULT_INFER_CONFIG.copy()
    }


# ============================================================================
# 与config_manager.py的集成接口
# ============================================================================

def to_config_manager_format(config_type: str) -> Dict[str, Any]:
    """将即用型配置转换为config_manager格式"""
    if config_type == 'train':
        config = DEFAULT_TRAIN_CONFIG.copy()
        return {
            'model': {
                'name': 'yolov8n',
                'type': 'detection',
                'input_size': [config['imgsz'], config['imgsz']],
                'pretrained': config['pretrained']
            },
            'training': {
                'epochs': config['epochs'],
                'batch_size': config['batch'],
                'learning_rate': config['lr0'],
                'momentum': config['momentum'],
                'weight_decay': config['weight_decay'],
                'warmup_epochs': config['warmup_epochs'],
                'optimizer': config['optimizer'],
                'patience': config['patience']
            },
            'augmentation': {
                'hsv_h': config['hsv_h'],
                'hsv_s': config['hsv_s'],
                'hsv_v': config['hsv_v'],
                'fliplr': config['fliplr'],
                'mosaic': config['mosaic'],
                'scale': config['scale'],
                'translate': config['translate']
            }
        }
    elif config_type in ['val', 'validation']:
        config = DEFAULT_VAL_CONFIG.copy()
        return {
            'validation': {
                'batch_size': config['batch'],
                'img_size': config['imgsz'],
                'conf_threshold': config['conf'],
                'iou_threshold': config['iou'],
                'max_detections': config['max_det']
            }
        }
    elif config_type in ['infer', 'inference']:
        config = DEFAULT_INFER_CONFIG.copy()
        return {
            'inference': {
                'confidence_threshold': config['conf'],
                'iou_threshold': config['iou'],
                'max_detections': config['max_det'],
                'input_size': [config['imgsz'], config['imgsz']],
                'batch_size': config['batch']
            }
        }
    else:
        raise ValueError(f"不支持的配置类型: {config_type}")


def main():
    """主函数，演示配置的使用方法"""
    print("=" * 60)
    print("BTD YOLO 即用型配置管理器")
    print("=" * 60)
    
    # 显示所有可用配置
    print("\n📋 可用配置类型:")
    configs = get_all_configs()
    for config_name in configs.keys():
        print(f"  - {config_name}")
    
    # 演示获取配置
    print(f"\n🔧 训练配置示例:")
    train_config = get_config_by_name('train')
    print(f"  epochs: {train_config['epochs']}")
    print(f"  batch: {train_config['batch']}")
    print(f"  device: {train_config['device']}")
    
    # 演示自定义配置
    print(f"\n⚙️ 自定义配置示例:")
    custom_config = create_custom_config('train', epochs=200, batch=32)
    print(f"  自定义epochs: {custom_config['epochs']}")
    print(f"  自定义batch: {custom_config['batch']}")
    
    print(f"\n💡 使用提示:")
    print(f"  1. 直接使用: from yolo_configs import DEFAULT_TRAIN_CONFIG")
    print(f"  2. 自定义配置: create_custom_config('train', epochs=200)")
    print(f"  3. 与config_manager集成: to_config_manager_format('train')")


if __name__ == "__main__":
    main()
