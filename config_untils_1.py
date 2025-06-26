# -*- coding:utf-8 -*-

# ====================================================
# @File    :   config_untils.py
# @Time    :   2023/07/26 18:02:12
# @Author  :   xx
# @Desc    :   配置文件读取和参数合并工具
# ====================================================

import os
import sys
from pathlib import Path
import logging
import subprocess
import yaml
import argparse
from typing import Dict, Any, Optional, Tuple, Union

# from my_utils.get_commit_id import get_commit_id_and_date

# 获取当前脚本所在目录的上级目录
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print("project_root:", project_root)
# sys.path.append(project_root)

# from config.config_loader import config

logger = logging.getLogger("config_loader")

# 添加BTD项目路径到sys.path以便导入相关模块
current_file = Path(__file__).resolve()
project_root = current_file.parent
btd_root = project_root / "BTD"
if btd_root.exists():
    sys.path.insert(0, str(btd_root))

# 导入项目路径配置
try:
    from yoloserver.utils.paths import CONFIGS_DIR, RUNS_DIR
    from yoloserver.scripts.yolo_configs import (
        DEFAULT_TRAIN_CONFIG, DEFAULT_VAL_CONFIG, DEFAULT_INFER_CONFIG,
        get_config_by_name
    )
except ImportError:
    # 备用路径配置
    CONFIGS_DIR = project_root / "BTD" / "yoloserver" / "configs"
    RUNS_DIR = project_root / "BTD" / "yoloserver" / "runs"

    # 确保目录存在
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # 备用默认配置
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

# YOLO官方参数集合定义
YOLO_TRAIN_ARGS = {
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

YOLO_VAL_ARGS = {
    'data', 'imgsz', 'batch', 'save_json', 'save_hybrid', 'conf', 'iou',
    'max_det', 'half', 'device', 'dnn', 'plots', 'rect', 'split', 'project',
    'name', 'verbose', 'save_txt', 'save_conf', 'save_crop', 'show_labels',
    'show_conf', 'visualize', 'augment', 'agnostic_nms', 'classes', 'retina_masks',
    'boxes', 'format', 'keras', 'optimize', 'int8', 'dynamic', 'simplify', 'opset',
    'workspace', 'nms', 'lr', 'resume', 'workers', 'exist_ok', 'pretrained',
    'optimizer', 'seed', 'deterministic', 'single_cls', 'cos_lr', 'close_mosaic',
    'amp', 'fraction', 'profile', 'freeze'
}

YOLO_INFER_ARGS = {
    'source', 'imgsz', 'conf', 'iou', 'device', 'show', 'save', 'save_frames',
    'save_txt', 'save_conf', 'save_crop', 'show_labels', 'show_conf', 'vid_stride',
    'stream_buffer', 'line_width', 'visualize', 'augment', 'agnostic_nms',
    'classes', 'retina_masks', 'boxes', 'format', 'keras', 'optimize', 'int8',
    'dynamic', 'simplify', 'opset', 'workspace', 'nms', 'lr', 'resume', 'batch',
    'workers', 'project', 'name', 'exist_ok', 'pretrained', 'optimizer', 'verbose',
    'seed', 'deterministic', 'single_cls', 'rect', 'cos_lr', 'close_mosaic',
    'amp', 'fraction', 'profile', 'freeze', 'plots', 'half', 'dnn', 'max_det'
}

CONFIG_DIR = Path(sys.executable).parent / "config" if getattr(sys, 'frozen', False) else Path(
    __file__).parent / "config"

# 配置文件加载逻辑（仅在需要时执行）
def load_config_file():
    """加载配置文件"""
    try:
        # 优先从命令行参数读取配置文件路径，没有则使用默认路径
        # 方便单个脚本调试
        if len(sys.argv) > 1 and sys.argv[1].endswith(".yml"):
            config_path = Path(sys.argv[1])
        else:
            config_path = CONFIG_DIR / "config.yml"

        # 如果配置文件不存在，则尝试生成默认配置文件
        if not config_path.exists():
            logger.warning(f"配置文件 {config_path} 不存在，尝试生成默认配置文件")
            try:
                generate_default_config_if_not_exist()
            except Exception as e:
                logger.error(f"生成默认配置文件失败: {e}")
                return None

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            logger.info(f"成功读取配置文件: {config_path}")
            return config

    except FileNotFoundError:
        logger.error(f"配置文件 {config_path} 不存在，请检查路径")
        return None
    except yaml.YAMLError as e:
        logger.error(f"配置文件 {config_path} 解析错误，请检查语法: {e}")
        return None
    except Exception as e:
        logger.error(f"读取配置文件 {config_path} 时发生未知错误: {e}")
        return None


def print_config_info(config):
    """打印配置信息"""
    if config is None:
        return

    logger.info("=" * 20 + "  CONFIG  " + "=" * 20)
    for first_level_key, first_level_value in config.items():
        if isinstance(first_level_value, dict):
            logger.info(f"{first_level_key}:")
            for second_level_key, second_level_value in first_level_value.items():
                logger.info(f"\t{second_level_key:<20}: {second_level_value}")
        else:
            logger.info(f"{first_level_key:<20}: {first_level_value}")
    logger.info("=" * 50)


def generate_default_config_if_not_exist():
    """
    如果配置文件不存在，则生成一份默认的
    """

    # 检查config目录是否存在，不存在则创建
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir()
    
    config_path = CONFIG_DIR / "config.yml"
    
    config = {}
    config["COMMON_PATH_CFG"] = {
        "IMG_SAVE_PATH_PRE": "D:/img_pre",
        "IMG_SAVE_PATH_INFER": "D:/img_infer",
        "IMG_SAVE_PATH_POST": "D:/img_post",
    }
    
    config["COMMON_CAL_CFG"] = {
        "SAVE_PRE_IMG": True,
        "SAVE_INFER_IMG": True,
        "SAVE_POST_IMG": True,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
        logger.info(f"成功生成默认配置文件: {config_path}")


def merge_configs(mode: str,
                 args: argparse.Namespace,
                 yaml_config: Optional[Dict[str, Any]] = None) -> Tuple[argparse.Namespace, argparse.Namespace]:
    """
    整合来自不同来源的配置参数

    Args:
        mode: 运行模式 ('train', 'val', 'infer')
        args: 命令行参数 (argparse.Namespace)
        yaml_config: YAML配置字典

    Returns:
        Tuple[argparse.Namespace, argparse.Namespace]: (project_args, yolo_args)
        - project_args: 包含所有参数和来源标记的命名空间
        - yolo_args: 仅包含YOLO官方参数的命名空间
    """

    # ============================================================================
    # 1. 确定运行模式和相关配置
    # ============================================================================

    # 根据模式确定有效参数集合和默认配置
    mode_configs = {
        'train': (YOLO_TRAIN_ARGS, DEFAULT_TRAIN_CONFIG),
        'training': (YOLO_TRAIN_ARGS, DEFAULT_TRAIN_CONFIG),
        'val': (YOLO_VAL_ARGS, DEFAULT_VAL_CONFIG),
        'validation': (YOLO_VAL_ARGS, DEFAULT_VAL_CONFIG),
        'infer': (YOLO_INFER_ARGS, DEFAULT_INFER_CONFIG),
        'inference': (YOLO_INFER_ARGS, DEFAULT_INFER_CONFIG),
        'predict': (YOLO_INFER_ARGS, DEFAULT_INFER_CONFIG)
    }

    if mode not in mode_configs:
        raise ValueError(f"不支持的运行模式: {mode}. 支持的模式: {list(mode_configs.keys())}")

    valid_args, default_config = mode_configs[mode]

    logger.info(f"运行模式: {mode}")
    logger.info(f"默认配置参数数量: {len(default_config)}")
    logger.info(f"YOLO官方参数数量: {len(valid_args)}")

    # ============================================================================
    # 2. 初始化参数存储
    # ============================================================================

    # 创建参数存储对象
    project_args = argparse.Namespace()
    yolo_args = argparse.Namespace()

    # 从默认配置开始合并
    merged_params = default_config.copy()

    logger.info("初始化参数存储完成")

    # ============================================================================
    # 3. 合并 YAML 参数（优先级高于默认值）
    # ============================================================================

    if yaml_config and getattr(args, 'use_yaml', True):
        logger.info("开始合并YAML配置参数")
        yaml_param_count = 0

        for key, value in yaml_config.items():
            # 类型转换处理
            processed_value = _convert_yaml_value(key, value)
            merged_params[key] = processed_value
            yaml_param_count += 1

        logger.info(f"YAML参数合并完成，共处理 {yaml_param_count} 个参数")
    else:
        logger.info("跳过YAML配置合并")

    # ============================================================================
    # 4. 合并命令行参数（最高优先级）
    # ============================================================================

    logger.info("开始合并命令行参数")
    cli_param_count = 0

    # 处理预定义参数
    for attr_name in dir(args):
        if not attr_name.startswith('_') and attr_name not in ['extra_args']:
            value = getattr(args, attr_name)
            if value is not None:
                # 类型转换处理
                processed_value = _convert_cli_value(attr_name, value)
                merged_params[attr_name] = processed_value

                # 设置来源标记
                setattr(project_args, f"{attr_name}_specified", True)
                cli_param_count += 1

    # 处理动态参数 (extra_args)
    if hasattr(args, 'extra_args') and args.extra_args:
        extra_args = args.extra_args
        if len(extra_args) % 2 != 0:
            raise ValueError("extra_args 必须是偶数个参数（键值对）")

        for i in range(0, len(extra_args), 2):
            key = extra_args[i].lstrip('-')  # 移除前导的 - 或 --
            value = extra_args[i + 1]

            # 类型转换处理
            processed_value = _convert_cli_value(key, value)
            merged_params[key] = processed_value

            # 设置来源标记
            setattr(project_args, f"{key}_specified", True)
            cli_param_count += 1

    logger.info(f"命令行参数合并完成，共处理 {cli_param_count} 个参数")

    # ============================================================================
    # 5. 路径标准化
    # ============================================================================

    logger.info("开始路径标准化处理")

    # 处理 data 参数路径
    if 'data' in merged_params and merged_params['data']:
        data_path = Path(merged_params['data'])
        if not data_path.is_absolute():
            data_path = CONFIGS_DIR / data_path

        merged_params['data'] = str(data_path)

        # 验证路径存在性
        if not data_path.exists():
            logger.warning(f"数据配置文件不存在: {data_path}")

    # 处理 project 参数路径
    if 'project' in merged_params and merged_params['project']:
        project_path = Path(merged_params['project'])
        if not project_path.is_absolute():
            project_path = RUNS_DIR / project_path

        merged_params['project'] = str(project_path)

        # 尝试创建目录
        try:
            project_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"项目输出目录已准备: {project_path}")
        except Exception as e:
            logger.warning(f"无法创建项目目录 {project_path}: {e}")

    logger.info("路径标准化处理完成")

    # ============================================================================
    # 6. 分离 yolo_args 和 project_args
    # ============================================================================

    logger.info("开始分离参数")
    yolo_param_count = 0
    project_param_count = 0

    for key, value in merged_params.items():
        # 设置到 project_args
        setattr(project_args, key, value)
        project_param_count += 1

        # 如果是YOLO官方参数，也设置到 yolo_args
        if key in valid_args:
            setattr(yolo_args, key, value)
            yolo_param_count += 1

        # 补充来源标记（如果还没有设置）
        specified_attr = f"{key}_specified"
        if not hasattr(project_args, specified_attr):
            # 如果参数来自YAML且没有被CLI覆盖，标记为False
            if yaml_config and key in yaml_config:
                setattr(project_args, specified_attr, False)
            else:
                setattr(project_args, specified_attr, True)

    logger.info(f"参数分离完成 - 项目参数: {project_param_count}, YOLO参数: {yolo_param_count}")

    # ============================================================================
    # 7. 参数验证
    # ============================================================================

    logger.info("开始参数验证")
    _validate_params(mode, merged_params)
    logger.info("参数验证通过")

    # 记录最终参数摘要
    logger.info("=" * 50)
    logger.info("参数合并完成摘要:")
    logger.info(f"  运行模式: {mode}")
    logger.info(f"  项目参数总数: {project_param_count}")
    logger.info(f"  YOLO参数总数: {yolo_param_count}")
    logger.info(f"  数据配置: {merged_params.get('data', 'N/A')}")
    logger.info(f"  输出目录: {merged_params.get('project', 'N/A')}")
    logger.info("=" * 50)

    return project_args, yolo_args


def _convert_yaml_value(key: str, value: Any) -> Any:
    """
    转换YAML配置值到正确的Python类型

    Args:
        key: 参数键名
        value: 原始值

    Returns:
        Any: 转换后的值
    """
    if value is None:
        return None

    # 布尔值转换
    if isinstance(value, str):
        if value.lower() in ['true', 'yes', '1', 'on']:
            return True
        elif value.lower() in ['false', 'no', '0', 'off']:
            return False
        elif value.lower() in ['none', 'null', '']:
            return None

    # classes 参数特殊处理
    if key == 'classes' and isinstance(value, str):
        if ',' in value:
            try:
                return [int(x.strip()) for x in value.split(',')]
            except ValueError:
                return value.split(',')
        else:
            try:
                return [int(value)]
            except ValueError:
                return [value]

    return value


def _convert_cli_value(key: str, value: Any) -> Any:
    """
    转换命令行参数值到正确的Python类型

    Args:
        key: 参数键名
        value: 原始值

    Returns:
        Any: 转换后的值
    """
    if value is None:
        return None

    # 如果已经是正确类型，直接返回
    if not isinstance(value, str):
        return value

    # 尝试类型推断转换
    # 整数转换
    try:
        if '.' not in value and value.isdigit():
            return int(value)
    except (ValueError, AttributeError):
        pass

    # 浮点数转换
    try:
        if '.' in value:
            return float(value)
    except (ValueError, AttributeError):
        pass

    # 布尔值转换
    if value.lower() in ['true', 'yes', '1', 'on']:
        return True
    elif value.lower() in ['false', 'no', '0', 'off']:
        return False

    # None 转换
    if value.lower() in ['none', 'null', '']:
        return None

    # classes 参数特殊处理
    if key == 'classes':
        if ',' in value:
            try:
                return [int(x.strip()) for x in value.split(',')]
            except ValueError:
                return value.split(',')
        else:
            try:
                return [int(value)]
            except ValueError:
                return [value]

    # 默认返回字符串
    return value


def _validate_params(mode: str, params: Dict[str, Any]) -> None:
    """
    验证参数的合法性

    Args:
        mode: 运行模式
        params: 参数字典

    Raises:
        ValueError: 参数验证失败时抛出
    """
    errors = []

    if mode in ['train', 'training']:
        # 训练模式验证
        if 'epochs' in params:
            epochs = params['epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                errors.append(f"epochs 必须是正整数，当前值: {epochs}")

        if 'imgsz' in params:
            imgsz = params['imgsz']
            if not isinstance(imgsz, int) or imgsz <= 0 or imgsz % 8 != 0:
                errors.append(f"imgsz 必须是正整数且为8的倍数，当前值: {imgsz}")

        if 'batch' in params:
            batch = params['batch']
            if batch is not None and (not isinstance(batch, int) or batch <= 0):
                errors.append(f"batch 必须是正整数或None，当前值: {batch}")

        if 'data' in params:
            data_path = Path(params['data'])
            if not data_path.exists():
                errors.append(f"数据配置文件不存在: {data_path}")

    elif mode in ['val', 'validation']:
        # 验证模式验证
        if 'split' not in params:
            params['split'] = 'val'  # 设置默认值

        if 'data' in params:
            data_path = Path(params['data'])
            if not data_path.exists():
                errors.append(f"数据配置文件不存在: {data_path}")

    elif mode in ['infer', 'inference', 'predict']:
        # 推理模式验证
        if 'model' in params:
            model_path = Path(params['model'])
            if not model_path.exists():
                errors.append(f"模型文件不存在: {model_path}")

    # 如果有错误，抛出异常
    if errors:
        error_msg = "参数验证失败:\n" + "\n".join(f"  - {error}" for error in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)


def demo_merge_configs():
    """演示 merge_configs 函数的使用方法"""

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("merge_configs 函数演示")
    print("=" * 60)

    # 模拟命令行参数
    args = argparse.Namespace()
    args.epochs = 200
    args.batch = 32
    args.device = 'cuda:0'
    args.use_yaml = True
    args.extra_args = ['--lr0', '0.001', '--momentum', '0.9']

    # 模拟YAML配置
    yaml_config = {
        'imgsz': 640,
        'workers': 8,
        'save': True,
        'plots': True
    }

    try:
        # 调用 merge_configs
        project_args, yolo_args = merge_configs('train', args, yaml_config)

        print("\n✅ 参数合并成功!")
        print(f"\n📋 项目参数 (project_args):")
        for attr in sorted(dir(project_args)):
            if not attr.startswith('_'):
                value = getattr(project_args, attr)
                print(f"  {attr}: {value}")

        print(f"\n🎯 YOLO参数 (yolo_args):")
        for attr in sorted(dir(yolo_args)):
            if not attr.startswith('_'):
                value = getattr(yolo_args, attr)
                print(f"  {attr}: {value}")

        print(f"\n💡 使用示例:")
        print(f"  # 获取参数值")
        print(f"  epochs = project_args.epochs  # {getattr(project_args, 'epochs', 'N/A')}")
        print(f"  device = project_args.device  # {getattr(project_args, 'device', 'N/A')}")
        print(f"  ")
        print(f"  # 检查参数来源")
        print(f"  epochs_from_cli = project_args.epochs_specified  # {getattr(project_args, 'epochs_specified', 'N/A')}")
        print(f"  ")
        print(f"  # 直接传递给YOLO模型")
        print(f"  model.train(**vars(yolo_args))")

    except Exception as e:
        print(f"\n❌ 参数合并失败: {e}")


if __name__ == "__main__":
    # 如果直接运行此文件，执行演示
    try:
        demo_merge_configs()
    except Exception as e:
        print(f"演示执行失败: {e}")
        import traceback
        traceback.print_exc()