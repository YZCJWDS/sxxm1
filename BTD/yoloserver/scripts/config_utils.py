#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
配置文件读取和参数合并工具

主要功能：
1. 加载配置文件，如果不存在则生成默认配置
2. 合并命令行参数、YAML配置和默认参数
3. 按优先级处理参数：CLI > YAML > 默认值
4. 路径标准化和参数验证
5. 分离YOLO官方参数和项目参数

创建：2025/6/26 09:22:17
作者：xx
"""

import logging
import yaml
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# 自动设置项目路径，支持直接运行
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent  # 回到项目根目录

# 添加项目根目录到Python路径
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"🔧 配置工具路径设置:")
print(f"   当前文件: {current_file}")
print(f"   项目根目录: {project_root}")
print(f"   Python路径已更新")

# 导入配置常量 - 分层导入，确保最大兼容性
try:
    # 第一优先级：使用项目专业配置（同目录导入）
    from yolo_configs import (
        DEFAULT_TRAIN_CONFIG, DEFAULT_VAL_CONFIG, DEFAULT_INFER_CONFIG
    )
    # 尝试相对导入
    try:
        from ..utils.paths import CONFIGS_DIR, RUNS_DIR
    except ImportError:
        # 如果相对导入失败，使用绝对导入
        from BTD.yoloserver.utils.paths import CONFIGS_DIR, RUNS_DIR
    logger.info("✅ 成功导入项目专业配置")
except ImportError:
    try:
        # 第二优先级：使用项目专业配置（绝对导入）
        from BTD.yoloserver.scripts.yolo_configs import (
            DEFAULT_TRAIN_CONFIG, DEFAULT_VAL_CONFIG, DEFAULT_INFER_CONFIG
        )
        from BTD.yoloserver.utils.paths import CONFIGS_DIR, RUNS_DIR
        logger.info("✅ 成功导入项目专业配置")
    except ImportError:
        # 第三优先级：使用备用配置
        logger.info("⚠️ 使用备用配置")
        CONFIGS_DIR = Path("configs")
        RUNS_DIR = Path("runs")

        # 确保目录存在
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

        DEFAULT_TRAIN_CONFIG = {
            'data': 'data.yaml',
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'device': 'cpu',
            'project': str(RUNS_DIR / 'train'),
            'name': 'exp'
        }

        DEFAULT_VAL_CONFIG = {
            'data': 'data.yaml',
            'imgsz': 640,
            'batch': 16,
            'device': 'cpu',
            'project': str(RUNS_DIR / 'val'),
            'name': 'exp'
        }

        DEFAULT_INFER_CONFIG = {
            'source': '0',
            'imgsz': 640,
            'device': 'cpu',
            'project': str(RUNS_DIR / 'predict'),
            'name': 'exp'
        }

        # YOLO参数集合
        VALID_YOLO_TRAIN_ARGS = {
            'data', 'epochs', 'batch', 'imgsz', 'device', 'workers', 'project', 'name',
            'exist_ok', 'pretrained', 'optimizer', 'verbose', 'seed', 'deterministic',
            'single_cls', 'rect', 'cos_lr', 'close_mosaic', 'resume', 'amp', 'fraction',
            'profile', 'freeze', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
            'warmup_momentum', 'warmup_bias_lr', 'box', 'cls', 'dfl', 'pose', 'kobj',
            'label_smoothing', 'nbs', 'overlap_mask', 'mask_ratio', 'dropout', 'val',
            'plots', 'save', 'save_period', 'cache', 'copy_paste', 'auto_augment',
            'erasing', 'crop_fraction', 'bgr', 'mosaic', 'mixup', 'hsv_h', 'hsv_s',
            'hsv_v', 'degrees', 'translate', 'scale', 'shear', 'perspective', 'flipud',
            'fliplr', 'albumentations', 'augment', 'agnostic_nms', 'retina_masks'
        }

        VALID_YOLO_VAL_ARGS = {
            'data', 'imgsz', 'batch', 'save_json', 'save_hybrid', 'conf', 'iou',
            'max_det', 'half', 'device', 'dnn', 'plots', 'rect', 'split', 'project',
            'name', 'verbose', 'save_txt', 'save_conf', 'save_crop', 'show_labels',
            'show_conf', 'visualize', 'augment', 'agnostic_nms', 'classes', 'retina_masks'
        }

        VALID_YOLO_INFER_ARGS = {
            'source', 'imgsz', 'conf', 'iou', 'device', 'show', 'save', 'save_frames',
            'save_txt', 'save_conf', 'save_crop', 'show_labels', 'show_conf', 'vid_stride',
            'stream_buffer', 'line_width', 'visualize', 'augment', 'agnostic_nms',
            'classes', 'retina_masks', 'project', 'name', 'exist_ok', 'half', 'dnn', 'max_det'
        }


def load_config(config_type='train'):
    """
    加载配置文件，如果文件不存在，尝试生成默认配置文件后加载
    
    Args:
        config_type: 配置文件类型 ('train', 'val', 'infer')
        
    Returns:
        dict: 配置文件内容
    """
    config_path = CONFIGS_DIR / f'{config_type}.yaml'
    
    if not config_path.exists():
        logger.warning(f"配置文件 {config_path} 不存在，尝试生成默认配置文件")
        if config_type in ['train', 'val', 'infer']:
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                generate_default_config(config_type)
                logger.info(f"生成默认配置文件成功: {config_path}")
            except Exception as e:
                logger.error(f"生成配置文件失败: {e}")
                raise FileNotFoundError(f"配置文件生成失败: {e}")
        else:
            logger.error(f"配置文件类型错误: {config_type}")
            raise ValueError("仅支持 train/val/infer 类型")

    # 加载配置文件
    try:
        logger.info(f"正在加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"已加载配置文件: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"配置文件解析失败: {e}")
        raise
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise


def generate_default_config(config_type):
    """
    生成默认配置文件
    
    Args:
        config_type: 配置文件类型 ('train', 'val', 'infer')
    """
    config_path = CONFIGS_DIR / f'{config_type}.yaml'
    if config_type == 'train':
        config = DEFAULT_TRAIN_CONFIG
    elif config_type == 'val':
        config = DEFAULT_VAL_CONFIG
    elif config_type == 'infer':
        config = DEFAULT_INFER_CONFIG
    else:
        logger.error(f"不支持的配置文件类型: {config_type}")
        raise ValueError("仅支持 train/val/infer 类型")

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"生成默认 {config_type} 配置文件成功: {config_path}")
    except IOError as e:
        logger.error(f"写入配置文件失败: {e}")
        raise
    except Exception as e:
        logger.error(f"生成配置文件时发生未知错误: {e}")
        raise


def _process_params_value(key: str, value: Any) -> Any:
    """
    处理参数值，进行类型转换
    
    Args:
        key: 参数键名
        value: 参数值
        
    Returns:
        Any: 处理后的参数值
    """
    if value is None:
        return None
    
    # 如果不是字符串，直接返回
    if not isinstance(value, str):
        return value
    
    # 尝试转换为数字
    try:
        if '.' in value:
            return float(value)
        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
    except ValueError:
        pass
    
    # 布尔值转换
    if value.lower() in ('true', 'yes', '1', 'on'):
        return True
    elif value.lower() in ('false', 'no', '0', 'off'):
        return False
    elif value.lower() in ('none', 'null', ''):
        return None
    
    # 特殊处理classes参数
    if key == 'classes' and ',' in value:
        try:
            return [int(x.strip()) for x in value.split(',')]
        except ValueError:
            return value.split(',')
    
    return value


def merge_config(args: argparse.Namespace, yaml_config: Optional[Dict[str, Any]] = None, mode: str = 'train') -> Tuple[argparse.Namespace, argparse.Namespace]:
    """
    合并命令行参数、YAML配置文件参数和默认参数，按优先级CLI > YAML > 默认值

    Args:
        args: 通过argparse解析的参数
        yaml_config: 从YAML配置文件中加载的参数
        mode: 运行模式，支持train, val, infer

    Returns:
        Tuple[argparse.Namespace, argparse.Namespace]: (yolo_args, project_args)
    """

    # 1. 确定运行模式和相关配置，根据传入的mode，选择有效的YOLO参数合集
    if mode == 'train':
        valid_args = VALID_YOLO_TRAIN_ARGS
        default_config = DEFAULT_TRAIN_CONFIG
    elif mode == 'val':
        valid_args = VALID_YOLO_VAL_ARGS
        default_config = DEFAULT_VAL_CONFIG
    elif mode == 'infer':
        valid_args = VALID_YOLO_INFER_ARGS
        default_config = DEFAULT_INFER_CONFIG
    else:
        logger.error(f"{mode}模式不存在，仅仅支持train/val/infer三种模式")
        raise ValueError(f"{mode} 模式不存在，仅仅支持train/val/infer三种模式")

    # 2. 初始化参数存储，project_args用于存储所有最终合并的参数，yolo_args用于存储yolo的参数
    project_args = argparse.Namespace()
    yolo_args = argparse.Namespace()
    merged_params = default_config.copy()

    # 3. 合并YAML参数，按优先级合并，只有当命令行指定了使用YAML文件，才进行合并
    if hasattr(args, 'use_yaml') and args.use_yaml and yaml_config:
        for key, value in yaml_config.items():
            merged_params[key] = _process_params_value(key, value)
        logger.debug(f"合并YAML参数后: {merged_params}")

    # 4. 合并命令行参数，具有最高的优先级，会覆盖YAML参数和默认值
    cmd_args = {k: v for k, v in vars(args).items() if k != 'extra_args' and v is not None}
    for key, value in cmd_args.items():
        # 为参数标记来源
        merged_params[key] = _process_params_value(key, value)
        setattr(project_args, f"{key}_specified", True)

    # 处理动态参数
    if hasattr(args, 'extra_args') and args.extra_args:
        if len(args.extra_args) % 2 != 0:
            logger.error("额外参数格式错误，参数列表必须成对出现，如 --key value")
            raise ValueError("额外参数格式错误")

        for i in range(0, len(args.extra_args), 2):
            key = args.extra_args[i].lstrip("--")
            value = args.extra_args[i + 1]

            processed_value = _process_params_value(key, value)
            merged_params[key] = processed_value
            # 标记额外的参数来源
            setattr(project_args, f"{key}_specified", True)

    # 5. 路径标准化
    if 'data' in merged_params and merged_params['data']:
        data_path = Path(merged_params['data'])
        if not data_path.is_absolute():
            data_path = CONFIGS_DIR / data_path
        merged_params['data'] = str(data_path.resolve())

        # 验证数据集配置文件是否存在
        if not data_path.exists():
            logger.warning(f"数据集配置文件'{data_path}'不存在")
        logger.info(f"标准化数据集路径: '{merged_params['data']}'")

    # 标准化project参数
    if 'project' in merged_params and merged_params['project']:
        project_path = Path(merged_params['project'])
        if not project_path.is_absolute():
            project_path = RUNS_DIR / project_path
        merged_params['project'] = str(project_path)

        try:
            project_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(f"PermissionError: 无权限创建目录 {project_path}")
            raise ValueError(f"PermissionError: 无权限创建目录 {project_path}")
        logger.info(f"标准化project路径: {merged_params['project']}")

    # 6. 分离yolo_args和project_args
    for key, value in merged_params.items():
        setattr(project_args, key, value)
        if key in valid_args:
            setattr(yolo_args, key, value)

        # 为YAML参数设置来源标记
        if yaml_config and key in yaml_config and not hasattr(project_args, f"{key}_specified"):
            setattr(project_args, f"{key}_specified", False)

    # 7. 参数验证，先pass

    # 返回分离后的两组参数
    return yolo_args, project_args


# 演示函数
def demo_config_utils():
    """演示配置工具的使用方法 - 简化版本，避免依赖问题"""

    print("=" * 50)
    print("配置工具演示")
    print("=" * 50)

    try:
        print("\n1. 参数值处理演示:")
        test_values = [
            ('epochs', '100'),
            ('lr0', '0.01'),
            ('save', 'true'),
            ('classes', '0,1,2')
        ]

        for key, value in test_values:
            try:
                processed = _process_params_value(key, value)
                print(f"   {key}: '{value}' -> {processed} ({type(processed).__name__})")
            except Exception as e:
                print(f"   {key}: 处理失败 - {e}")

        print("\n2. 基本功能验证:")
        print(f"   CONFIGS_DIR: {CONFIGS_DIR}")
        print(f"   RUNS_DIR: {RUNS_DIR}")
        print(f"   训练参数数量: {len(DEFAULT_TRAIN_CONFIG)}")
        print(f"   YOLO训练参数数量: {len(VALID_YOLO_TRAIN_ARGS)}")

        print("\n✅ 基本功能验证完成!")
        print("💡 提示: 要测试完整功能，请在项目环境中运行")

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        print("💡 这可能是正常的，如果在独立环境中运行")


if __name__ == '__main__':
    try:
        demo_config_utils()
    except Exception as e:
        print(f"程序运行出错: {e}")
        print("这可能是正常的，配置工具的核心功能仍然可用")
