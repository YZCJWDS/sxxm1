#!/usr/bin/env python3
"""
模型训练脚本
提供YOLO模型训练的完整流程
集成了配置管理、系统信息记录、数据集信息记录、训练后处理等完整功能
"""

import argparse
import sys
import logging
import shutil
import json
import time
import platform
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from yoloserver.utils.logger import setup_logger, get_logger, TimerLogger
from yoloserver.utils.path_utils import get_project_root, get_config_paths, get_model_paths
from yoloserver.utils.file_utils import read_yaml, write_yaml
from yoloserver.utils.performance_utils import time_it

logger = get_logger(__name__)

# 尝试导入psutil用于系统信息获取
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil未安装，系统信息记录功能受限。安装命令: pip install psutil")

# ==================== 配置管理功能 ====================

def load_config(config_type='train'):
    """
    加载配置文件，如果文件不存在，尝试生成默认配置文件后加载

    Args:
        config_type: 配置文件类型 ('train', 'val', 'infer')

    Returns:
        dict: 配置文件内容
    """
    try:
        config_paths = get_config_paths()
        config_path = config_paths.get('configs_dir', Path('configs')) / f'{config_type}.yaml'

        if config_path.exists():
            config = read_yaml(config_path)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        else:
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return get_default_config(config_type)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return get_default_config(config_type)

def get_default_config(config_type='train'):
    """获取默认配置"""
    if config_type == 'train':
        return {
            'data': 'data.yaml',
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'device': 'cpu',
            'lr0': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'save': True,
            'plots': True
        }
    elif config_type == 'val':
        return {
            'data': 'data.yaml',
            'imgsz': 640,
            'batch': 16,
            'conf': 0.25,
            'iou': 0.7,
            'save_txt': True,
            'save_conf': True
        }
    else:
        return {}

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
    # 获取默认配置
    default_config = get_default_config(mode)

    # 初始化参数存储
    project_args = argparse.Namespace()
    yolo_args = argparse.Namespace()
    merged_params = default_config.copy()

    # 合并YAML参数
    if hasattr(args, 'use_yaml') and getattr(args, 'use_yaml', False) and yaml_config:
        for key, value in yaml_config.items():
            merged_params[key] = value
        logger.debug(f"合并YAML参数后: {merged_params}")

    # 合并命令行参数，具有最高优先级
    cmd_args = {k: v for k, v in vars(args).items() if v is not None}
    for key, value in cmd_args.items():
        merged_params[key] = value
        setattr(project_args, f"{key}_specified", True)

    # 分离YOLO参数和项目参数
    yolo_param_keys = {
        'data', 'epochs', 'batch', 'imgsz', 'device', 'lr0', 'lrf', 'momentum',
        'weight_decay', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate',
        'scale', 'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup',
        'workers', 'cache', 'rect', 'cos_lr', 'close_mosaic', 'amp', 'fraction',
        'profile', 'freeze', 'project', 'name', 'resume', 'pretrained'
    }

    for key, value in merged_params.items():
        if key in yolo_param_keys:
            setattr(yolo_args, key, value)
        setattr(project_args, key, value)

    return yolo_args, project_args

# ==================== 系统信息记录功能 ====================

def format_bytes(bytes_value):
    """格式化字节数为可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def get_device_info():
    """获取设备信息"""
    device_info = {
        "系统信息": {
            "操作系统": f"{platform.system()} {platform.release()}",
            "Python版本": platform.python_version(),
            "处理器": platform.processor() or "未知"
        }
    }

    if HAS_PSUTIL:
        # CPU信息
        device_info["CPU信息"] = {
            "物理核心数": psutil.cpu_count(logical=False),
            "逻辑核心数": psutil.cpu_count(logical=True),
            "CPU使用率": f"{psutil.cpu_percent(interval=1):.1f}%"
        }

        # 内存信息
        memory = psutil.virtual_memory()
        device_info["内存信息"] = {
            "总内存": format_bytes(memory.total),
            "可用内存": format_bytes(memory.available),
            "已用内存": format_bytes(memory.used),
            "内存使用率": f"{memory.percent:.1f}%"
        }

        # 磁盘信息
        disk = psutil.disk_usage('/')
        device_info["磁盘信息"] = {
            "总空间": format_bytes(disk.total),
            "已用空间": format_bytes(disk.used),
            "剩余空间": format_bytes(disk.free),
            "使用率": f"{disk.percent:.1f}%"
        }

        # GPU信息（尝试获取）
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory
                    gpu_info.append({
                        "GPU名称": gpu_name,
                        "显存": format_bytes(gpu_memory)
                    })
                device_info["GPU信息"] = gpu_info
            else:
                device_info["GPU信息"] = [{"信息": "未检测到CUDA可用GPU"}]
        except ImportError:
            device_info["GPU信息"] = [{"信息": "PyTorch未安装，无法检测GPU"}]

    return device_info

def log_device_info(logger_instance=None):
    """记录设备信息到日志"""
    if logger_instance is None:
        logger_instance = logger

    device_info = get_device_info()

    logger_instance.info("=" * 50)
    logger_instance.info("设备信息概览")
    logger_instance.info("=" * 50)

    for category, info in device_info.items():
        if category == "GPU信息":
            logger_instance.info(f"{category}:")
            for gpu_idx, gpu_detail in enumerate(info):
                if "未检测到CUDA可用GPU" in gpu_detail.get("信息", ""):
                    logger_instance.info(f"  {gpu_detail['信息']}")
                    break
                logger_instance.info(f"  --- GPU {gpu_idx} 详情 ---")
                for key, value in gpu_detail.items():
                    logger_instance.info(f"    {key}: {value}")
        else:
            logger_instance.info(f"{category}:")
            for key, value in info.items():
                logger_instance.info(f"    {key}: {value}")

    logger_instance.info("=" * 50)
    return device_info

# ==================== 数据集信息记录功能 ====================

def get_dataset_info(data_config_name: str, mode: str = "train") -> Tuple[int, list, int, str]:
    """
    获取数据集信息，包括类别数，类别名称和样本数量

    Args:
        data_config_name: 数据集的配置文件名称（如 "data.yaml"）
        mode: 模式，可选值为 "train", "val", "test", "infer"

    Returns:
        tuple: (类别数, 类别名称列表, 样本数, 样本来源描述)
    """
    # 初始化返回值
    nc = 0
    classes_names = []
    samples = 0
    source = "未知"

    # 推理模式下不提供数据集来源信息
    if mode == 'infer':
        return 0, [], 0, "推理模式，不提供数据集来源信息"

    try:
        # 尝试读取数据配置文件
        data_config_path = Path(data_config_name)
        if not data_config_path.exists():
            # 尝试在配置目录中查找
            config_paths = get_config_paths()
            data_config_path = config_paths.get('configs_dir', Path('configs')) / data_config_name

        if data_config_path.exists():
            data_config = read_yaml(data_config_path)
            if data_config:
                # 获取类别信息
                nc = data_config.get('nc', 0)
                classes_names = data_config.get('names', [])

                # 获取数据路径
                data_paths = {
                    'train': data_config.get('train', ''),
                    'val': data_config.get('val', ''),
                    'test': data_config.get('test', '')
                }

                # 统计样本数量
                if mode in data_paths and data_paths[mode]:
                    data_path = Path(data_paths[mode])
                    if data_path.exists():
                        if data_path.is_dir():
                            # 如果是目录，统计图像文件数量
                            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
                            samples = len([f for f in data_path.rglob('*') if f.suffix.lower() in image_extensions])
                            source = f"目录: {data_path}"
                        else:
                            # 如果是文件，读取文件行数
                            try:
                                with open(data_path, 'r', encoding='utf-8') as f:
                                    samples = len(f.readlines())
                                source = f"文件: {data_path}"
                            except Exception:
                                samples = 0
                                source = f"文件读取失败: {data_path}"
                    else:
                        source = f"路径不存在: {data_paths[mode]}"
                else:
                    source = f"配置中未找到{mode}路径"
            else:
                source = f"配置文件读取失败: {data_config_path}"
        else:
            source = f"配置文件不存在: {data_config_name}"

    except Exception as e:
        logger.error(f"获取数据集信息失败: {e}")
        source = f"获取信息时出错: {str(e)}"

    return nc, classes_names, samples, source

def log_dataset_info(data_config_name: str, mode: str = 'train', logger_instance=None) -> dict:
    """
    获取并记录数据集信息到日志

    Args:
        data_config_name: 数据集的配置文件名称
        mode: 模式，可选值为 "train", "val", "test", "infer"
        logger_instance: 日志记录器实例

    Returns:
        dict: 结构化的数据集信息字典
    """
    if logger_instance is None:
        logger_instance = logger

    nc, classes_names, samples, source = get_dataset_info(data_config_name, mode)

    logger_instance.info("=" * 50)
    logger_instance.info(f"数据集信息 ({mode.capitalize()} 模式)")
    logger_instance.info("-" * 50)
    logger_instance.info(f"{'Config File':<20}: {data_config_name}")
    logger_instance.info(f"{'Class Count':<20}: {nc}")
    logger_instance.info(f"{'Class Names':<20}: {', '.join(classes_names) if classes_names else '未知'}")
    logger_instance.info(f"{'Sample Count':<20}: {samples}")
    logger_instance.info(f"{'Data Source':<20}: {source}")
    logger_instance.info("-" * 50)

    return {
        "config_file": data_config_name,
        "mode": mode,
        "class_count": nc,
        "class_names": classes_names,
        "sample_count": samples,
        "data_source": source
    }

# ==================== 参数记录功能 ====================

def log_parameters(project_args, logger_instance=None):
    """
    记录参数来源和详细的训练参数信息

    Args:
        project_args: 项目参数命名空间
        logger_instance: 日志记录器实例
    """
    if logger_instance is None:
        logger_instance = logger

    logger_instance.info("=" * 50)
    logger_instance.info("训练参数详情")
    logger_instance.info("-" * 50)

    # 记录参数来源
    specified_params = []
    for attr_name in dir(project_args):
        if attr_name.endswith('_specified') and getattr(project_args, attr_name, False):
            param_name = attr_name.replace('_specified', '')
            specified_params.append(param_name)

    if specified_params:
        logger_instance.info(f"命令行指定参数: {', '.join(specified_params)}")
    else:
        logger_instance.info("命令行指定参数: 无")

    # 记录所有参数
    logger_instance.info("\n所有训练参数:")
    for attr_name in sorted(dir(project_args)):
        if not attr_name.startswith('_') and not attr_name.endswith('_specified'):
            value = getattr(project_args, attr_name)
            logger_instance.info(f"  {attr_name:<20}: {value}")

    logger_instance.info("-" * 50)

# ==================== 训练结果记录功能 ====================

def log_results(results, logger_instance=None, model_trainer=None) -> dict:
    """
    记录YOLO模型训练结果信息

    Args:
        results: Ultralytics的训练结果对象
        logger_instance: 日志记录器实例
        model_trainer: Ultralytics的Trainer对象

    Returns:
        dict: 包含模型评估结果的结构化字典
    """
    if logger_instance is None:
        logger_instance = logger

    logger_instance.info("=" * 50)
    logger_instance.info("训练结果概览")
    logger_instance.info("-" * 50)

    result_info = {}

    try:
        # 获取保存目录
        save_dir = None
        if hasattr(results, 'save_dir') and results.save_dir:
            save_dir = str(results.save_dir)
        elif model_trainer and hasattr(model_trainer, 'save_dir'):
            save_dir = str(model_trainer.save_dir)

        if save_dir:
            result_info['save_dir'] = save_dir
            logger_instance.info(f"{'保存目录':<20}: {save_dir}")

        # 尝试获取训练指标
        if hasattr(results, 'box') and results.box:
            metrics = results.box
            if hasattr(metrics, 'map50'):
                result_info['map50'] = float(metrics.map50)
                logger_instance.info(f"{'mAP@0.5':<20}: {metrics.map50:.4f}")
            if hasattr(metrics, 'map'):
                result_info['map50_95'] = float(metrics.map)
                logger_instance.info(f"{'mAP@0.5:0.95':<20}: {metrics.map:.4f}")
            if hasattr(metrics, 'mp'):
                result_info['precision'] = float(metrics.mp)
                logger_instance.info(f"{'Precision':<20}: {metrics.mp:.4f}")
            if hasattr(metrics, 'mr'):
                result_info['recall'] = float(metrics.mr)
                logger_instance.info(f"{'Recall':<20}: {metrics.mr:.4f}")

        # 记录训练时间信息
        if hasattr(results, 'speed') and results.speed:
            speed_info = results.speed
            for key, value in speed_info.items():
                result_info[f'speed_{key}'] = value
                logger_instance.info(f"{'Speed ' + key:<20}: {value:.2f}ms")

        # 记录最佳权重路径
        if save_dir:
            best_weights = Path(save_dir) / "weights" / "best.pt"
            last_weights = Path(save_dir) / "weights" / "last.pt"

            if best_weights.exists():
                result_info['best_weights'] = str(best_weights)
                logger_instance.info(f"{'最佳权重':<20}: {best_weights}")

            if last_weights.exists():
                result_info['last_weights'] = str(last_weights)
                logger_instance.info(f"{'最终权重':<20}: {last_weights}")

        result_info['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger_instance.info(f"{'完成时间':<20}: {result_info['timestamp']}")

    except Exception as e:
        logger_instance.error(f"记录训练结果时出错: {e}")
        result_info['error'] = str(e)

    logger_instance.info("-" * 50)
    return result_info

# ==================== 日志文件管理功能 ====================

def rename_log_file(logger_instance, save_dir: str, model_name: str):
    """
    重命名日志文件，添加模型名称和保存目录信息

    Args:
        logger_instance: 日志记录器实例
        save_dir: 训练结果保存目录
        model_name: 模型名称
    """
    try:
        # 获取当前日志文件路径
        for handler in logger_instance.handlers:
            if isinstance(handler, logging.FileHandler):
                current_log_path = Path(handler.baseFilename)

                # 生成新的日志文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_log_name = f"train_{model_name}_{timestamp}.log"
                new_log_path = current_log_path.parent / new_log_name

                # 关闭当前文件处理器
                handler.close()
                logger_instance.removeHandler(handler)

                # 重命名文件
                if current_log_path.exists():
                    shutil.move(str(current_log_path), str(new_log_path))
                    logger_instance.info(f"日志文件已重命名: {new_log_path}")

                # 创建新的文件处理器
                new_handler = logging.FileHandler(new_log_path, encoding='utf-8')
                new_handler.setLevel(handler.level)
                new_handler.setFormatter(handler.formatter)
                logger_instance.addHandler(new_handler)

                break

    except Exception as e:
        logger.error(f"重命名日志文件失败: {e}")

# ==================== 模型检查点管理功能 ====================

def copy_checkpoint_models(save_dir: Path, model_name: str, checkpoints_dir: Path, logger_instance=None):
    """
    复制训练检查点模型到指定目录

    Args:
        save_dir: 训练结果保存目录
        model_name: 模型名称
        checkpoints_dir: 检查点保存目录
        logger_instance: 日志记录器实例
    """
    if logger_instance is None:
        logger_instance = logger

    try:
        # 确保检查点目录存在
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        weights_dir = save_dir / "weights"
        if not weights_dir.exists():
            logger_instance.warning(f"权重目录不存在: {weights_dir}")
            return

        # 复制最佳模型
        best_model = weights_dir / "best.pt"
        if best_model.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_best = checkpoints_dir / f"{model_name}_best_{timestamp}.pt"
            shutil.copy2(best_model, dest_best)
            logger_instance.info(f"最佳模型已复制: {dest_best}")

        # 复制最终模型
        last_model = weights_dir / "last.pt"
        if last_model.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_last = checkpoints_dir / f"{model_name}_last_{timestamp}.pt"
            shutil.copy2(last_model, dest_last)
            logger_instance.info(f"最终模型已复制: {dest_last}")

    except Exception as e:
        logger_instance.error(f"复制检查点模型失败: {e}")

def train_model(config_path: Optional[str] = None,
               data_config_path: Optional[str] = None,
               model_name: str = "yolo11n",  # 更新为YOLO v11
               epochs: int = 100,
               batch: int = 16,  # 修正参数名
               imgsz: int = 640,  # 修正参数名
               device: str = "",
               project: str = None,  # 将使用项目默认路径
               name: str = "exp",
               resume: bool = False,
               pretrained: bool = True,
               use_yaml: bool = True,  # 添加YAML配置支持
               **kwargs) -> bool:
    """
    训练YOLO模型

    Args:
        config_path: 模型配置文件路径
        data_config_path: 数据配置文件路径
        model_name: 模型名称
        epochs: 训练轮数
        batch: 批次大小
        imgsz: 输入图像尺寸
        device: 训练设备
        project: 项目目录
        name: 实验名称
        resume: 是否恢复训练
        pretrained: 是否使用预训练权重
        use_yaml: 是否使用YAML配置文件
        **kwargs: 其他训练参数

    Returns:
        bool: 训练是否成功
    """
    try:
        logger.info("YOLO 模型训练脚本启动".center(80, "="))

        # 1. 加载配置文件
        yaml_config = {}
        if use_yaml:
            yaml_config = load_config(config_type='train')

        # 2. 创建参数命名空间用于合并配置
        args = argparse.Namespace()
        args.data = data_config_path
        args.batch = batch
        args.epochs = epochs
        args.imgsz = imgsz
        args.device = device
        args.weights = f"{model_name}.pt"
        args.use_yaml = use_yaml

        # 3. 合并参数
        yolo_args, project_args = merge_config(args, yaml_config, mode='train')

        # 4. 记录设备信息
        log_device_info(logger)

        # 5. 获取数据信息
        if data_config_path:
            data_file = data_config_path
        else:
            # 使用配置目录中的data.yaml
            config_paths = get_config_paths()
            data_file = str(config_paths['data_yaml'])

        logger.info(f"使用数据配置文件: {data_file}")
        log_dataset_info(data_file, mode='train', logger_instance=logger)

        # 6. 记录参数来源
        log_parameters(project_args, logger_instance=logger)

        # 7. 准备训练配置
        train_config = {}

        # 设置项目路径 - 使用项目结构
        if project is None:
            model_paths = get_model_paths()
            project = str(model_paths['models_dir'].parent / "runs" / "train")

        train_config.update({
            'epochs': epochs,
            'batch': batch,
            'imgsz': imgsz,
            'device': device,
            'project': project,
            'name': name,
            'resume': resume,
            'pretrained': pretrained
        })

        # 过滤kwargs，只保留有效的YOLO参数，避免参数名冲突
        valid_yolo_params = {
            'lr0', 'lrf', 'momentum', 'weight_decay', 'hsv_h', 'hsv_s', 'hsv_v',
            'degrees', 'translate', 'scale', 'shear', 'perspective', 'flipud', 'fliplr',
            'mosaic', 'mixup', 'workers', 'cache', 'rect', 'cos_lr', 'close_mosaic',
            'amp', 'fraction', 'profile', 'freeze'
        }

        # 只添加有效的YOLO参数
        for key, value in kwargs.items():
            if key in valid_yolo_params:
                train_config[key] = value

        # 8. 检查是否安装了ultralytics
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("请安装ultralytics: pip install ultralytics")
            return False

        # 9. 初始化模型
        logger.info(f"初始化模型，加载模型: {project_args.weights}")

        # 优先从项目预训练目录查找
        model_paths = get_model_paths()
        pretrained_model = model_paths['pretrained'] / project_args.weights

        if pretrained_model.exists():
            model_path = str(pretrained_model)
            logger.info(f"使用项目预训练模型: {model_path}")
        else:
            # 如果项目目录没有，使用默认路径（会自动下载）
            model_path = project_args.weights
            logger.info(f"使用预训练模型（自动下载）: {model_path}")
            logger.info(f"建议将预训练模型放入: {model_paths['pretrained']}")

        if not Path(model_path).exists() and not model_path.endswith('.yaml'):
            logger.warning(f"模型文件不存在: {model_path}")

        model = YOLO(model_path)

        # 10. 使用time_it装饰器执行训练
        @time_it(iterations=1, name="模型训练", logger_instance=logger)
        def run_training(model_instance, train_args):
            return model_instance.train(**train_args)

        # 开始训练
        logger.info("开始训练...")
        logger.info(f"训练参数: {train_config}")

        results = run_training(model, {
            'data': data_file,
            **train_config
        })

        # 11. 记录结果信息
        log_results(results, logger_instance=logger, model_trainer=model.trainer)

        # 12. 重命名日志文件
        model_name_for_log = project_args.weights.replace(".pt", "")
        rename_log_file(logger, str(model.trainer.save_dir), model_name_for_log)

        # 13. 复制检查点模型
        checkpoints_dir = model_paths.get('checkpoints', Path('checkpoints'))
        copy_checkpoint_models(Path(model.trainer.save_dir),
                             project_args.weights,
                             checkpoints_dir,
                             logger_instance=logger)

        logger.info("YOLO 模型训练脚本结束")
        return True

    except Exception as e:
        logger.error(f"训练失败: {e}")
        return False

def resume_training(checkpoint_path: str, **kwargs) -> bool:
    """
    恢复训练
    
    Args:
        checkpoint_path: 检查点文件路径
        **kwargs: 其他训练参数
        
    Returns:
        bool: 恢复训练是否成功
    """
    try:
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.error(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        logger.info(f"从检查点恢复训练: {checkpoint_path}")
        
        from ultralytics import YOLO
        model = YOLO(str(checkpoint_path))
        
        # 恢复训练
        results = model.train(resume=True, **kwargs)
        
        logger.info("恢复训练完成")
        return True
        
    except Exception as e:
        logger.error(f"恢复训练失败: {e}")
        return False

def export_model(model_path: str, 
                format: str = "onnx",
                img_size: int = 640,
                half: bool = False,
                dynamic: bool = False,
                simplify: bool = True,
                **kwargs) -> bool:
    """
    导出模型
    
    Args:
        model_path: 模型文件路径
        format: 导出格式 (onnx, torchscript, tflite, etc.)
        img_size: 输入图像尺寸
        half: 是否使用半精度
        dynamic: 是否使用动态输入尺寸
        simplify: 是否简化ONNX模型
        **kwargs: 其他导出参数
        
    Returns:
        bool: 导出是否成功
    """
    try:
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        logger.info(f"导出模型: {model_path} -> {format}")
        
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        
        # 导出模型
        export_path = model.export(
            format=format,
            imgsz=img_size,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
            **kwargs
        )
        
        logger.info(f"模型导出成功: {export_path}")
        return True
        
    except Exception as e:
        logger.error(f"模型导出失败: {e}")
        return False

def main():
    """主函数，处理命令行参数"""
    # 检查是否有命令行参数，如果没有则使用默认配置直接运行
    if len(sys.argv) == 1:
        print("🚀 检测到直接运行模式，使用默认配置开始训练...")
        print("💡 如需自定义参数，请使用命令行模式：python train.py --help")

        # 直接运行训练，使用默认参数
        success = train_model()

        if success:
            print("✅ 训练成功完成")
            sys.exit(0)
        else:
            print("❌ 训练失败")
            sys.exit(1)

    # 有命令行参数时，正常解析参数
    parser = argparse.ArgumentParser(description="YOLO模型训练脚本")

    # 基本参数
    parser.add_argument("--config", type=str, help="模型配置文件路径")
    parser.add_argument("--data", type=str, help="数据配置文件路径")
    parser.add_argument("--model", type=str, default="yolo11n", help="模型名称")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")  # 修正参数名
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")  # 修正参数名
    parser.add_argument("--device", type=str, default="", help="训练设备")
    parser.add_argument("--project", type=str, default="runs/train", help="项目目录")
    parser.add_argument("--name", type=str, default="exp", help="实验名称")

    # 训练选项
    parser.add_argument("--resume", action="store_true", help="恢复训练")
    parser.add_argument("--no-pretrained", action="store_true", help="不使用预训练权重")
    parser.add_argument("--use-yaml", action="store_true", default=True, help="使用YAML配置文件")
    
    # 学习率参数
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD动量")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="权重衰减")
    
    # 数据增强参数
    parser.add_argument("--hsv-h", type=float, default=0.015, help="色调增强")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="饱和度增强")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="明度增强")
    parser.add_argument("--degrees", type=float, default=0.0, help="旋转角度")
    parser.add_argument("--translate", type=float, default=0.1, help="平移")
    parser.add_argument("--scale", type=float, default=0.5, help="缩放")
    parser.add_argument("--shear", type=float, default=0.0, help="剪切")
    parser.add_argument("--perspective", type=float, default=0.0, help="透视变换")
    parser.add_argument("--flipud", type=float, default=0.0, help="上下翻转概率")
    parser.add_argument("--fliplr", type=float, default=0.5, help="左右翻转概率")
    parser.add_argument("--mosaic", type=float, default=1.0, help="马赛克增强概率")
    parser.add_argument("--mixup", type=float, default=0.0, help="mixup增强概率")
    
    # 其他参数
    parser.add_argument("--workers", type=int, default=8, help="数据加载工作进程数")
    parser.add_argument("--cache", action="store_true", help="缓存图像")
    parser.add_argument("--rect", action="store_true", help="矩形训练")
    parser.add_argument("--cos-lr", action="store_true", help="余弦学习率调度")
    parser.add_argument("--close-mosaic", type=int, default=10, help="关闭马赛克增强的轮数")
    parser.add_argument("--amp", action="store_true", default=True, help="自动混合精度训练")
    parser.add_argument("--fraction", type=float, default=1.0, help="数据集使用比例")
    parser.add_argument("--profile", action="store_true", help="性能分析")
    parser.add_argument("--freeze", type=int, help="冻结层数")
    
    args = parser.parse_args()
    
    # 设置统一日志系统
    from pathlib import Path
    from datetime import datetime

    # 创建统一的日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 使用统一的日志文件名（按日期）
    today = datetime.now().strftime("%Y%m%d")
    unified_log_file = log_dir / f"yolo_training_{today}.log"

    # 配置统一日志 - 指定日志文件路径
    setup_logger("train", log_file=unified_log_file, console_output=True, file_output=True)
    
    # 准备训练参数
    train_kwargs = {
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'shear': args.shear,
        'perspective': args.perspective,
        'flipud': args.flipud,
        'fliplr': args.fliplr,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        'workers': args.workers,
        'cache': args.cache,
        'rect': args.rect,
        'cos_lr': args.cos_lr,
        'close_mosaic': args.close_mosaic,
        'amp': args.amp,
        'fraction': args.fraction,
        'profile': args.profile,
    }
    
    if args.freeze is not None:
        train_kwargs['freeze'] = args.freeze
    
    # 开始训练
    success = train_model(
        config_path=args.config,
        data_config_path=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,  # 修正参数名
        imgsz=args.imgsz,  # 修正参数名
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        pretrained=not args.no_pretrained,
        use_yaml=args.use_yaml,  # 添加YAML配置支持
        **train_kwargs
    )
    
    if success:
        logger.info("训练成功完成")
        sys.exit(0)
    else:
        logger.error("训练失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
