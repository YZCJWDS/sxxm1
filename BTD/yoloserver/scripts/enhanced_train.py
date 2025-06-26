#!/usr/bin/env python3
"""
增强版YOLO模型训练脚本
支持多种训练模式、数据增强、模型优化等功能
"""

import argparse
import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from yoloserver.utils.logger import setup_logger, get_logger, TimerLogger
from yoloserver.utils.path_utils import get_project_root, get_config_paths, get_model_paths
from yoloserver.utils.file_utils import read_yaml, write_yaml, write_json

logger = get_logger(__name__)

class YOLOTrainer:
    """YOLO模型训练器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.device = self._setup_device()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_path and Path(self.config_path).exists():
            return read_yaml(self.config_path)
        else:
            # 使用默认配置
            config_paths = get_config_paths()
            return read_yaml(config_paths['model_config'])
    
    def _setup_device(self) -> str:
        """设置训练设备"""
        if torch.cuda.is_available():
            device = "0"  # 使用第一个GPU
            logger.info(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("使用CPU训练")
        return device
    
    def _prepare_data(self, data_config_path: str) -> bool:
        """准备训练数据"""
        try:
            data_config = read_yaml(data_config_path)
            
            # 检查数据路径
            data_root = Path(data_config.get('path', './data'))
            train_path = data_root / data_config.get('train', 'train/images')
            val_path = data_root / data_config.get('val', 'val/images')
            
            if not train_path.exists():
                logger.error(f"训练数据路径不存在: {train_path}")
                return False
                
            if not val_path.exists():
                logger.error(f"验证数据路径不存在: {val_path}")
                return False
            
            # 统计数据
            train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
            val_images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))
            
            logger.info(f"训练图像数量: {len(train_images)}")
            logger.info(f"验证图像数量: {len(val_images)}")
            
            return True
            
        except Exception as e:
            logger.error(f"数据准备失败: {e}")
            return False
    
    def _setup_model(self, model_name: str, pretrained: bool = True) -> bool:
        """设置模型"""
        try:
            from ultralytics import YOLO
            
            if pretrained:
                # 使用预训练模型
                model_path = f"{model_name}.pt"
                logger.info(f"加载预训练模型: {model_path}")
            else:
                # 从配置文件创建模型
                model_path = f"{model_name}.yaml"
                logger.info(f"从配置创建模型: {model_path}")
            
            self.model = YOLO(model_path)
            return True
            
        except Exception as e:
            logger.error(f"模型设置失败: {e}")
            return False
    
    def _setup_training_params(self, **kwargs) -> Dict[str, Any]:
        """设置训练参数"""
        # 默认训练参数
        default_params = {
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save': True,
            'save_period': -1,
            'cache': False,
            'device': self.device,
            'workers': 8,
            'project': 'runs/train',
            'name': 'exp',
            'exist_ok': False,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
        }
        
        # 从配置文件更新参数
        if 'train' in self.config:
            default_params.update(self.config['train'])
        
        # 从命令行参数更新
        default_params.update(kwargs)
        
        return default_params
    
    def _setup_data_augmentation(self) -> Dict[str, Any]:
        """设置数据增强参数"""
        aug_params = {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
        }
        
        # 从配置文件更新
        if 'augmentation' in self.config:
            aug_params.update(self.config['augmentation'])
        
        return aug_params
    
    def train(self, 
              data_config_path: str,
              model_name: str = "yolov8n",
              **kwargs) -> bool:
        """
        开始训练
        
        Args:
            data_config_path: 数据配置文件路径
            model_name: 模型名称
            **kwargs: 其他训练参数
            
        Returns:
            bool: 训练是否成功
        """
        try:
            with TimerLogger("模型训练", logger):
                logger.info("=" * 60)
                logger.info("开始YOLO模型训练")
                logger.info("=" * 60)
                
                # 1. 准备数据
                logger.info("1. 准备训练数据...")
                if not self._prepare_data(data_config_path):
                    return False
                
                # 2. 设置模型
                logger.info("2. 设置模型...")
                pretrained = kwargs.get('pretrained', True)
                if not self._setup_model(model_name, pretrained):
                    return False
                
                # 3. 设置训练参数
                logger.info("3. 设置训练参数...")
                train_params = self._setup_training_params(**kwargs)
                
                # 4. 设置数据增强
                logger.info("4. 设置数据增强...")
                aug_params = self._setup_data_augmentation()
                train_params.update(aug_params)
                
                # 5. 开始训练
                logger.info("5. 开始训练...")
                logger.info(f"训练参数: {train_params}")
                
                # 保存训练配置
                self._save_training_config(train_params, data_config_path)
                
                # 执行训练
                results = self.model.train(
                    data=data_config_path,
                    **train_params
                )
                
                # 6. 保存训练结果
                logger.info("6. 保存训练结果...")
                self._save_training_results(results, train_params)
                
                logger.info("=" * 60)
                logger.info("训练完成！")
                logger.info(f"最佳模型保存在: {results.save_dir}")
                logger.info("=" * 60)
                
                return True
                
        except Exception as e:
            logger.error(f"训练失败: {e}")
            return False
    
    def _save_training_config(self, train_params: Dict[str, Any], data_config_path: str):
        """保存训练配置"""
        try:
            config_info = {
                'training_time': datetime.now().isoformat(),
                'model_config': self.config,
                'training_params': train_params,
                'data_config_path': data_config_path,
                'device_info': {
                    'device': self.device,
                    'cuda_available': torch.cuda.is_available(),
                    'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                }
            }
            
            # 保存到项目根目录
            config_file = get_project_root() / 'last_training_config.yaml'
            write_yaml(config_info, config_file)
            logger.info(f"训练配置已保存: {config_file}")
            
        except Exception as e:
            logger.warning(f"保存训练配置失败: {e}")
    
    def _save_training_results(self, results, train_params: Dict[str, Any]):
        """保存训练结果摘要"""
        try:
            if hasattr(results, 'save_dir'):
                results_info = {
                    'training_completed': datetime.now().isoformat(),
                    'save_directory': str(results.save_dir),
                    'training_params': train_params,
                    'best_model_path': str(results.save_dir / 'weights' / 'best.pt'),
                    'last_model_path': str(results.save_dir / 'weights' / 'last.pt'),
                }
                
                # 保存结果信息
                results_file = results.save_dir / 'training_summary.json'
                write_json(results_info, results_file)
                logger.info(f"训练结果摘要已保存: {results_file}")
                
        except Exception as e:
            logger.warning(f"保存训练结果失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="增强版YOLO模型训练脚本")
    
    # 基本参数
    parser.add_argument("--config", type=str, help="模型配置文件路径")
    parser.add_argument("--data", type=str, required=True, help="数据配置文件路径")
    parser.add_argument("--model", type=str, default="yolov8n", help="模型名称")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--img-size", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--device", type=str, default="", help="训练设备")
    parser.add_argument("--project", type=str, default="runs/train", help="项目目录")
    parser.add_argument("--name", type=str, default="exp", help="实验名称")
    
    # 训练选项
    parser.add_argument("--resume", action="store_true", help="恢复训练")
    parser.add_argument("--no-pretrained", action="store_true", help="不使用预训练权重")
    parser.add_argument("--cache", action="store_true", help="缓存图像")
    parser.add_argument("--rect", action="store_true", help="矩形训练")
    parser.add_argument("--cos-lr", action="store_true", help="余弦学习率调度")
    parser.add_argument("--amp", action="store_true", default=True, help="自动混合精度训练")
    
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
    parser.add_argument("--close-mosaic", type=int, default=10, help="关闭马赛克增强的轮数")
    parser.add_argument("--fraction", type=float, default=1.0, help="数据集使用比例")
    parser.add_argument("--profile", action="store_true", help="性能分析")
    parser.add_argument("--freeze", type=int, help="冻结层数")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger("enhanced_train", console_output=True, file_output=True)
    
    # 创建训练器
    trainer = YOLOTrainer(config_path=args.config)
    
    # 准备训练参数
    train_kwargs = {
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.img_size,
        'device': args.device if args.device else trainer.device,
        'project': args.project,
        'name': args.name,
        'resume': args.resume,
        'pretrained': not args.no_pretrained,
        'cache': args.cache,
        'rect': args.rect,
        'cos_lr': args.cos_lr,
        'amp': args.amp,
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
        'close_mosaic': args.close_mosaic,
        'fraction': args.fraction,
        'profile': args.profile,
    }
    
    if args.freeze is not None:
        train_kwargs['freeze'] = args.freeze
    
    # 开始训练
    success = trainer.train(
        data_config_path=args.data,
        model_name=args.model,
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
