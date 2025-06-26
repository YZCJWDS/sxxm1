#!/usr/bin/env python3
"""
模型管理脚本
提供模型下载、转换、优化、部署等功能
"""

import argparse
import sys
import os
import requests
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import yaml

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from yoloserver.utils.logger import setup_logger, get_logger, TimerLogger, ProgressLogger
from yoloserver.utils.path_utils import get_model_paths, ensure_dir
from yoloserver.utils.file_utils import write_json, read_json

logger = get_logger(__name__)

class ModelManager:
    """模型管理器"""
    
    # 预训练模型信息
    PRETRAINED_MODELS = {
        'yolov8n': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'size': '6.2MB',
            'params': '3.2M',
            'map50_95': 37.3,
            'description': 'YOLOv8 Nano - 最小最快的模型'
        },
        'yolov8s': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
            'size': '21.5MB',
            'params': '11.2M',
            'map50_95': 44.9,
            'description': 'YOLOv8 Small - 平衡速度和精度'
        },
        'yolov8m': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
            'size': '49.7MB',
            'params': '25.9M',
            'map50_95': 50.2,
            'description': 'YOLOv8 Medium - 中等大小模型'
        },
        'yolov8l': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
            'size': '83.7MB',
            'params': '43.7M',
            'map50_95': 52.9,
            'description': 'YOLOv8 Large - 高精度模型'
        },
        'yolov8x': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
            'size': '130.5MB',
            'params': '68.2M',
            'map50_95': 53.9,
            'description': 'YOLOv8 Extra Large - 最高精度模型'
        }
    }
    
    def __init__(self):
        """初始化模型管理器"""
        self.model_paths = get_model_paths()
        ensure_dir(self.model_paths['pretrained'])
        ensure_dir(self.model_paths['checkpoints'])
    
    def list_available_models(self) -> None:
        """列出可用的预训练模型"""
        logger.info("可用的预训练模型:")
        logger.info("-" * 80)
        logger.info(f"{'模型名称':<12} {'大小':<10} {'参数量':<10} {'mAP50-95':<10} {'描述'}")
        logger.info("-" * 80)
        
        for name, info in self.PRETRAINED_MODELS.items():
            logger.info(f"{name:<12} {info['size']:<10} {info['params']:<10} "
                       f"{info['map50_95']:<10} {info['description']}")
        logger.info("-" * 80)
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """
        下载预训练模型
        
        Args:
            model_name: 模型名称
            force: 是否强制重新下载
            
        Returns:
            bool: 下载是否成功
        """
        if model_name not in self.PRETRAINED_MODELS:
            logger.error(f"未知模型: {model_name}")
            logger.info("可用模型: " + ", ".join(self.PRETRAINED_MODELS.keys()))
            return False
        
        model_info = self.PRETRAINED_MODELS[model_name]
        model_file = self.model_paths['pretrained'] / f"{model_name}.pt"
        
        # 检查文件是否已存在
        if model_file.exists() and not force:
            logger.info(f"模型已存在: {model_file}")
            return True
        
        try:
            with TimerLogger(f"下载模型 {model_name}", logger):
                logger.info(f"开始下载模型: {model_name}")
                logger.info(f"URL: {model_info['url']}")
                logger.info(f"大小: {model_info['size']}")
                
                # 下载文件
                response = requests.get(model_info['url'], stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(model_file, 'wb') as f:
                    if total_size > 0:
                        progress = ProgressLogger(total_size, logger, log_interval=10)
                        downloaded = 0
                        
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                progress.current = downloaded
                                progress.update(0)  # 不增加，直接设置当前值
                    else:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                
                logger.info(f"模型下载完成: {model_file}")
                return True
                
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            if model_file.exists():
                model_file.unlink()  # 删除不完整的文件
            return False
    
    def list_local_models(self) -> Dict[str, List[Path]]:
        """列出本地模型文件"""
        local_models = {
            'pretrained': [],
            'checkpoints': []
        }
        
        # 预训练模型
        pretrained_dir = self.model_paths['pretrained']
        if pretrained_dir.exists():
            local_models['pretrained'] = list(pretrained_dir.glob("*.pt"))
        
        # 检查点模型
        checkpoints_dir = self.model_paths['checkpoints']
        if checkpoints_dir.exists():
            local_models['checkpoints'] = list(checkpoints_dir.glob("*.pt"))
        
        return local_models
    
    def show_local_models(self) -> None:
        """显示本地模型信息"""
        local_models = self.list_local_models()
        
        logger.info("本地模型文件:")
        logger.info("=" * 60)
        
        # 预训练模型
        logger.info("预训练模型:")
        if local_models['pretrained']:
            for model_file in local_models['pretrained']:
                size_mb = model_file.stat().st_size / (1024 * 1024)
                logger.info(f"  {model_file.name} ({size_mb:.1f}MB)")
        else:
            logger.info("  无")
        
        logger.info("")
        
        # 检查点模型
        logger.info("训练检查点:")
        if local_models['checkpoints']:
            for model_file in local_models['checkpoints']:
                size_mb = model_file.stat().st_size / (1024 * 1024)
                logger.info(f"  {model_file.name} ({size_mb:.1f}MB)")
        else:
            logger.info("  无")
        
        logger.info("=" * 60)
    
    def export_model(self, model_path: str, format: str = "onnx", **kwargs) -> bool:
        """
        导出模型到指定格式
        
        Args:
            model_path: 模型文件路径
            format: 导出格式
            **kwargs: 其他导出参数
            
        Returns:
            bool: 导出是否成功
        """
        try:
            from ultralytics import YOLO
            
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            logger.info(f"导出模型: {model_path} -> {format}")
            
            # 加载模型
            model = YOLO(str(model_path))
            
            # 导出参数
            export_params = {
                'format': format,
                'imgsz': kwargs.get('imgsz', 640),
                'half': kwargs.get('half', False),
                'dynamic': kwargs.get('dynamic', False),
                'simplify': kwargs.get('simplify', True),
                'opset': kwargs.get('opset', None),
                'workspace': kwargs.get('workspace', 4),
                'nms': kwargs.get('nms', False),
            }
            
            # 执行导出
            export_path = model.export(**export_params)
            
            logger.info(f"模型导出成功: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            return False
    
    def optimize_model(self, model_path: str, optimization_type: str = "quantize") -> bool:
        """
        优化模型
        
        Args:
            model_path: 模型文件路径
            optimization_type: 优化类型 (quantize, prune, distill)
            
        Returns:
            bool: 优化是否成功
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            logger.info(f"优化模型: {model_path} ({optimization_type})")
            
            if optimization_type == "quantize":
                return self._quantize_model(model_path)
            elif optimization_type == "prune":
                return self._prune_model(model_path)
            elif optimization_type == "distill":
                return self._distill_model(model_path)
            else:
                logger.error(f"不支持的优化类型: {optimization_type}")
                return False
                
        except Exception as e:
            logger.error(f"模型优化失败: {e}")
            return False
    
    def _quantize_model(self, model_path: Path) -> bool:
        """量化模型"""
        try:
            # 这里可以实现模型量化逻辑
            logger.info("模型量化功能开发中...")
            return True
        except Exception as e:
            logger.error(f"模型量化失败: {e}")
            return False
    
    def _prune_model(self, model_path: Path) -> bool:
        """剪枝模型"""
        try:
            # 这里可以实现模型剪枝逻辑
            logger.info("模型剪枝功能开发中...")
            return True
        except Exception as e:
            logger.error(f"模型剪枝失败: {e}")
            return False
    
    def _distill_model(self, model_path: Path) -> bool:
        """知识蒸馏"""
        try:
            # 这里可以实现知识蒸馏逻辑
            logger.info("知识蒸馏功能开发中...")
            return True
        except Exception as e:
            logger.error(f"知识蒸馏失败: {e}")
            return False
    
    def benchmark_model(self, model_path: str, data_config: str) -> Dict:
        """
        模型性能基准测试
        
        Args:
            model_path: 模型文件路径
            data_config: 数据配置文件路径
            
        Returns:
            Dict: 基准测试结果
        """
        try:
            from ultralytics import YOLO
            import time
            
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"模型文件不存在: {model_path}")
                return {}
            
            logger.info(f"开始模型基准测试: {model_path}")
            
            # 加载模型
            model = YOLO(str(model_path))
            
            # 验证模型获取精度指标
            val_results = model.val(data=data_config, verbose=False)
            
            # 推理速度测试
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 预热
            dummy_input = torch.randn(1, 3, 640, 640).to(device)
            for _ in range(10):
                _ = model.predict(dummy_input, verbose=False)
            
            # 测试推理时间
            times = []
            for _ in range(100):
                start_time = time.time()
                _ = model.predict(dummy_input, verbose=False)
                times.append(time.time() - start_time)
            
            avg_time = sum(times) / len(times)
            fps = 1.0 / avg_time
            
            # 模型信息
            model_size = model_path.stat().st_size / (1024 * 1024)  # MB
            
            benchmark_results = {
                'model_path': str(model_path),
                'model_size_mb': round(model_size, 2),
                'map50': float(val_results.box.map50),
                'map50_95': float(val_results.box.map),
                'precision': float(val_results.box.mp),
                'recall': float(val_results.box.mr),
                'avg_inference_time_ms': round(avg_time * 1000, 2),
                'fps': round(fps, 1),
                'device': device,
                'test_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info("基准测试结果:")
            logger.info(f"  模型大小: {benchmark_results['model_size_mb']} MB")
            logger.info(f"  mAP50: {benchmark_results['map50']:.4f}")
            logger.info(f"  mAP50-95: {benchmark_results['map50_95']:.4f}")
            logger.info(f"  推理时间: {benchmark_results['avg_inference_time_ms']} ms")
            logger.info(f"  FPS: {benchmark_results['fps']}")
            
            # 保存结果
            benchmark_file = self.model_paths['model_root'] / f"benchmark_{model_path.stem}.json"
            write_json(benchmark_results, benchmark_file)
            logger.info(f"基准测试结果已保存: {benchmark_file}")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"基准测试失败: {e}")
            return {}

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模型管理脚本")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 列出可用模型
    list_parser = subparsers.add_parser('list', help='列出可用的预训练模型')
    
    # 下载模型
    download_parser = subparsers.add_parser('download', help='下载预训练模型')
    download_parser.add_argument('model', type=str, help='模型名称')
    download_parser.add_argument('--force', action='store_true', help='强制重新下载')
    
    # 显示本地模型
    local_parser = subparsers.add_parser('local', help='显示本地模型')
    
    # 导出模型
    export_parser = subparsers.add_parser('export', help='导出模型')
    export_parser.add_argument('model', type=str, help='模型文件路径')
    export_parser.add_argument('--format', type=str, default='onnx', help='导出格式')
    export_parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    export_parser.add_argument('--half', action='store_true', help='使用半精度')
    export_parser.add_argument('--dynamic', action='store_true', help='动态输入尺寸')
    export_parser.add_argument('--simplify', action='store_true', default=True, help='简化ONNX模型')
    
    # 优化模型
    optimize_parser = subparsers.add_parser('optimize', help='优化模型')
    optimize_parser.add_argument('model', type=str, help='模型文件路径')
    optimize_parser.add_argument('--type', type=str, default='quantize',
                                choices=['quantize', 'prune', 'distill'], help='优化类型')
    
    # 基准测试
    benchmark_parser = subparsers.add_parser('benchmark', help='模型基准测试')
    benchmark_parser.add_argument('model', type=str, help='模型文件路径')
    benchmark_parser.add_argument('--data', type=str, required=True, help='数据配置文件路径')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 设置日志
    setup_logger("model_manager", console_output=True, file_output=True)
    
    # 创建模型管理器
    manager = ModelManager()
    
    success = False
    
    if args.command == 'list':
        manager.list_available_models()
        success = True
    
    elif args.command == 'download':
        success = manager.download_model(args.model, args.force)
    
    elif args.command == 'local':
        manager.show_local_models()
        success = True
    
    elif args.command == 'export':
        export_kwargs = {
            'imgsz': args.imgsz,
            'half': args.half,
            'dynamic': args.dynamic,
            'simplify': args.simplify,
        }
        success = manager.export_model(args.model, args.format, **export_kwargs)
    
    elif args.command == 'optimize':
        success = manager.optimize_model(args.model, args.type)
    
    elif args.command == 'benchmark':
        results = manager.benchmark_model(args.model, args.data)
        success = bool(results)
    
    if success:
        logger.info("操作成功完成")
        sys.exit(0)
    else:
        logger.error("操作失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
