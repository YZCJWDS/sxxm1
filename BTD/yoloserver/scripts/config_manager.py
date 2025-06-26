#!/usr/bin/env python3
"""
配置管理脚本
提供配置文件创建、验证、更新等功能
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from yoloserver.utils.logger import setup_logger, get_logger
from yoloserver.utils.path_utils import get_config_paths, ensure_dir
from yoloserver.utils.file_utils import read_yaml, write_yaml, write_json

logger = get_logger(__name__)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        """初始化配置管理器"""
        self.config_paths = get_config_paths()
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict]:
        """加载配置模板"""
        templates = {
            'model_config': {
                'model': {
                    'name': 'yolov8n',
                    'type': 'detection',
                    'input_size': [640, 640],
                    'num_classes': 80,
                    'class_names': [],
                    'pretrained': True,
                    'weights_path': None
                },
                'training': {
                    'epochs': 100,
                    'batch_size': 16,
                    'learning_rate': 0.01,
                    'momentum': 0.937,
                    'weight_decay': 0.0005,
                    'warmup_epochs': 3,
                    'optimizer': 'SGD',
                    'scheduler': 'cosine',
                    'early_stopping': {
                        'enabled': True,
                        'patience': 50,
                        'min_delta': 0.001
                    }
                },
                'augmentation': {
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
                    'copy_paste': 0.0
                },
                'validation': {
                    'val_split': 0.2,
                    'save_period': 10,
                    'save_best': True,
                    'metrics': ['mAP50', 'mAP50-95', 'precision', 'recall']
                }
            },
            
            'dataset_config': {
                'path': './data',
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': 80,
                'names': {
                    0: 'person',
                    1: 'bicycle',
                    2: 'car',
                    3: 'motorcycle',
                    4: 'airplane',
                    5: 'bus',
                    6: 'train',
                    7: 'truck'
                },
                'download': None,
                'format': 'yolo',
                'annotation_format': 'txt',
                'image_formats': ['jpg', 'jpeg', 'png', 'bmp'],
                'preprocessing': {
                    'resize': True,
                    'normalize': True,
                    'target_size': [640, 640]
                }
            },
            
            'inference_config': {
                'model_path': 'models/best.pt',
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_detections': 1000,
                'device': 'auto',
                'half_precision': False,
                'input_size': [640, 640],
                'batch_size': 1,
                'save_results': True,
                'output_format': 'json',
                'visualization': {
                    'show_labels': True,
                    'show_confidence': True,
                    'line_thickness': 2,
                    'font_size': 12
                }
            },
            
            'server_config': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False,
                'workers': 4,
                'max_request_size': '16MB',
                'timeout': 30,
                'cors': {
                    'enabled': True,
                    'origins': ['*'],
                    'methods': ['GET', 'POST'],
                    'headers': ['*']
                },
                'rate_limiting': {
                    'enabled': True,
                    'requests_per_minute': 60,
                    'burst_size': 10
                },
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'file': 'logs/server.log',
                    'max_size': '10MB',
                    'backup_count': 5
                }
            }
        }
        
        return templates
    
    def create_config(self, config_type: str, output_path: Optional[str] = None, 
                     custom_values: Optional[Dict] = None) -> str:
        """
        创建配置文件
        
        Args:
            config_type: 配置类型
            output_path: 输出路径
            custom_values: 自定义值
            
        Returns:
            str: 配置文件路径
        """
        try:
            if config_type not in self.templates:
                logger.error(f"未知配置类型: {config_type}")
                logger.info(f"可用类型: {list(self.templates.keys())}")
                return ""
            
            # 获取模板
            config_template = self.templates[config_type].copy()
            
            # 应用自定义值
            if custom_values:
                config_template = self._merge_configs(config_template, custom_values)
            
            # 确定输出路径
            if output_path is None:
                if config_type in self.config_paths:
                    output_path = self.config_paths[config_type]
                else:
                    output_path = Path(f"{config_type}.yaml")
            else:
                output_path = Path(output_path)
            
            # 确保目录存在
            ensure_dir(output_path.parent)
            
            # 写入配置文件
            write_yaml(config_template, output_path)
            
            logger.info(f"配置文件已创建: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"创建配置文件失败: {e}")
            return ""
    
    def _merge_configs(self, base_config: Dict, custom_config: Dict) -> Dict:
        """合并配置"""
        result = base_config.copy()
        
        for key, value in custom_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(self, config_path: str, config_type: Optional[str] = None) -> bool:
        """
        验证配置文件
        
        Args:
            config_path: 配置文件路径
            config_type: 配置类型
            
        Returns:
            bool: 验证是否通过
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                logger.error(f"配置文件不存在: {config_path}")
                return False
            
            # 读取配置
            config = read_yaml(config_path)
            
            # 自动检测配置类型
            if config_type is None:
                config_type = self._detect_config_type(config)
                if config_type is None:
                    logger.warning("无法自动检测配置类型")
                    return True  # 跳过类型验证
            
            # 验证配置结构
            if config_type in self.templates:
                validation_result = self._validate_config_structure(config, self.templates[config_type])
                if validation_result['valid']:
                    logger.info(f"✓ 配置文件验证通过: {config_path}")
                    return True
                else:
                    logger.error(f"配置文件验证失败: {config_path}")
                    for error in validation_result['errors']:
                        logger.error(f"  - {error}")
                    return False
            else:
                logger.warning(f"未知配置类型: {config_type}")
                return True
                
        except Exception as e:
            logger.error(f"配置文件验证失败: {e}")
            return False
    
    def _detect_config_type(self, config: Dict) -> Optional[str]:
        """自动检测配置类型"""
        # 检查关键字段来判断配置类型
        if 'model' in config and 'training' in config:
            return 'model_config'
        elif 'path' in config and 'names' in config and 'nc' in config:
            return 'dataset_config'
        elif 'confidence_threshold' in config and 'iou_threshold' in config:
            return 'inference_config'
        elif 'host' in config and 'port' in config:
            return 'server_config'
        else:
            return None
    
    def _validate_config_structure(self, config: Dict, template: Dict) -> Dict:
        """验证配置结构"""
        errors = []
        
        def validate_recursive(cfg: Dict, tmpl: Dict, path: str = ""):
            for key, value in tmpl.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in cfg:
                    errors.append(f"缺少必需字段: {current_path}")
                    continue
                
                if isinstance(value, dict) and isinstance(cfg[key], dict):
                    validate_recursive(cfg[key], value, current_path)
                elif isinstance(value, list) and not isinstance(cfg[key], list):
                    errors.append(f"字段类型错误: {current_path} 应为列表")
                elif isinstance(value, (int, float)) and not isinstance(cfg[key], (int, float)):
                    errors.append(f"字段类型错误: {current_path} 应为数字")
                elif isinstance(value, str) and not isinstance(cfg[key], str):
                    errors.append(f"字段类型错误: {current_path} 应为字符串")
        
        validate_recursive(config, template)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def update_config(self, config_path: str, updates: Dict, backup: bool = True) -> bool:
        """
        更新配置文件
        
        Args:
            config_path: 配置文件路径
            updates: 更新内容
            backup: 是否备份原文件
            
        Returns:
            bool: 更新是否成功
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                logger.error(f"配置文件不存在: {config_path}")
                return False
            
            # 备份原文件
            if backup:
                backup_path = config_path.with_suffix(f"{config_path.suffix}.backup")
                config_path.rename(backup_path)
                logger.info(f"原配置已备份: {backup_path}")
                
                # 读取原配置
                original_config = read_yaml(backup_path)
            else:
                original_config = read_yaml(config_path)
            
            # 合并更新
            updated_config = self._merge_configs(original_config, updates)
            
            # 写入更新后的配置
            write_yaml(updated_config, config_path)
            
            logger.info(f"配置文件已更新: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"更新配置文件失败: {e}")
            return False
    
    def list_configs(self) -> None:
        """列出所有配置文件"""
        logger.info("项目配置文件:")
        logger.info("-" * 50)
        
        for config_name, config_path in self.config_paths.items():
            config_path = Path(config_path)
            status = "✓" if config_path.exists() else "✗"
            size = f"({config_path.stat().st_size} bytes)" if config_path.exists() else "(不存在)"
            
            logger.info(f"{status} {config_name}: {config_path} {size}")
    
    def show_template(self, config_type: str) -> None:
        """显示配置模板"""
        if config_type not in self.templates:
            logger.error(f"未知配置类型: {config_type}")
            logger.info(f"可用类型: {list(self.templates.keys())}")
            return
        
        logger.info(f"{config_type} 配置模板:")
        logger.info("-" * 50)
        
        template = self.templates[config_type]
        self._print_dict(template)
    
    def _print_dict(self, d: Dict, indent: int = 0) -> None:
        """递归打印字典"""
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")
    
    def export_all_configs(self, output_dir: str) -> bool:
        """导出所有配置到指定目录"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            exported_count = 0
            
            for config_type in self.templates.keys():
                output_file = output_dir / f"{config_type}.yaml"
                if self.create_config(config_type, str(output_file)):
                    exported_count += 1
            
            logger.info(f"已导出 {exported_count} 个配置文件到: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="配置管理脚本")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 创建配置
    create_parser = subparsers.add_parser('create', help='创建配置文件')
    create_parser.add_argument('type', type=str, help='配置类型')
    create_parser.add_argument('--output', type=str, help='输出文件路径')
    create_parser.add_argument('--values', type=str, help='自定义值 (JSON格式)')
    
    # 验证配置
    validate_parser = subparsers.add_parser('validate', help='验证配置文件')
    validate_parser.add_argument('config_file', type=str, help='配置文件路径')
    validate_parser.add_argument('--type', type=str, help='配置类型')
    
    # 更新配置
    update_parser = subparsers.add_parser('update', help='更新配置文件')
    update_parser.add_argument('config_file', type=str, help='配置文件路径')
    update_parser.add_argument('updates', type=str, help='更新内容 (JSON格式)')
    update_parser.add_argument('--no-backup', action='store_true', help='不备份原文件')
    
    # 列出配置
    list_parser = subparsers.add_parser('list', help='列出所有配置文件')
    
    # 显示模板
    template_parser = subparsers.add_parser('template', help='显示配置模板')
    template_parser.add_argument('type', type=str, help='配置类型')
    
    # 导出配置
    export_parser = subparsers.add_parser('export', help='导出所有配置')
    export_parser.add_argument('output_dir', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 设置日志
    setup_logger("config_manager", console_output=True, file_output=True)
    
    # 创建配置管理器
    manager = ConfigManager()
    
    success = False
    
    if args.command == 'create':
        custom_values = None
        if args.values:
            try:
                custom_values = json.loads(args.values)
            except json.JSONDecodeError:
                logger.error("自定义值格式错误，应为有效的JSON")
                sys.exit(1)
        
        config_file = manager.create_config(args.type, args.output, custom_values)
        success = bool(config_file)
    
    elif args.command == 'validate':
        success = manager.validate_config(args.config_file, args.type)
    
    elif args.command == 'update':
        try:
            updates = json.loads(args.updates)
            success = manager.update_config(args.config_file, updates, not args.no_backup)
        except json.JSONDecodeError:
            logger.error("更新内容格式错误，应为有效的JSON")
            sys.exit(1)
    
    elif args.command == 'list':
        manager.list_configs()
        success = True
    
    elif args.command == 'template':
        manager.show_template(args.type)
        success = True
    
    elif args.command == 'export':
        success = manager.export_all_configs(args.output_dir)
    
    if success:
        logger.info("操作成功完成")
        sys.exit(0)
    else:
        logger.error("操作失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
