#!/usr/bin/env python3
"""
模型验证脚本
提供YOLO模型验证和评估功能
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from yoloserver.utils.logger import setup_logger, get_logger, TimerLogger
from yoloserver.utils.path_utils import get_project_root, get_config_paths
from yoloserver.utils.file_utils import read_yaml, write_json

logger = get_logger(__name__)

def validate_model(model_path: str,
                  data_config_path: Optional[str] = None,
                  img_size: int = 640,
                  batch_size: int = 32,
                  conf_thres: float = 0.001,
                  iou_thres: float = 0.6,
                  max_det: int = 300,
                  device: str = "",
                  save_json: bool = False,
                  save_hybrid: bool = False,
                  save_conf: bool = False,
                  save_txt: bool = False,
                  plots: bool = True,
                  project: str = "runs/val",
                  name: str = "exp",
                  **kwargs) -> Optional[Dict[str, Any]]:
    """
    验证YOLO模型
    
    Args:
        model_path: 模型文件路径
        data_config_path: 数据配置文件路径
        img_size: 验证图像尺寸
        batch_size: 批次大小
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        max_det: 最大检测数量
        device: 验证设备
        save_json: 保存COCO格式结果
        save_hybrid: 保存混合标签
        save_conf: 保存置信度
        save_txt: 保存txt格式结果
        plots: 保存验证图片
        project: 项目目录
        name: 实验名称
        **kwargs: 其他验证参数
        
    Returns:
        Optional[Dict]: 验证结果，失败返回None
    """
    try:
        with TimerLogger("模型验证", logger):
            logger.info("开始YOLO模型验证")
            
            # 检查模型文件
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"模型文件不存在: {model_path}")
                return None
            
            # 获取数据配置
            if data_config_path is None:
                config_paths = get_config_paths()
                data_config_path = config_paths['dataset_config']
            
            data_config = read_yaml(data_config_path)
            if not data_config:
                logger.error("数据配置文件读取失败")
                return None
            
            # 检查是否安装了ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                logger.error("请安装ultralytics: pip install ultralytics")
                return None
            
            # 加载模型
            logger.info(f"加载模型: {model_path}")
            model = YOLO(str(model_path))
            
            # 验证参数
            val_config = {
                'data': str(data_config_path),
                'imgsz': img_size,
                'batch': batch_size,
                'conf': conf_thres,
                'iou': iou_thres,
                'max_det': max_det,
                'device': device,
                'save_json': save_json,
                'save_hybrid': save_hybrid,
                'save_conf': save_conf,
                'save_txt': save_txt,
                'plots': plots,
                'project': project,
                'name': name
            }
            val_config.update(kwargs)
            
            # 开始验证
            logger.info("开始验证...")
            logger.info(f"验证参数: {val_config}")
            
            results = model.val(**val_config)
            
            # 提取验证结果
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'fitness': float(results.fitness),
            }
            
            # 按类别的指标
            if hasattr(results.box, 'ap_class_index') and len(results.box.ap_class_index) > 0:
                class_metrics = {}
                for i, class_idx in enumerate(results.box.ap_class_index):
                    class_name = data_config.get('names', {}).get(int(class_idx), f"class_{class_idx}")
                    class_metrics[class_name] = {
                        'AP50': float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
                        'AP50-95': float(results.box.ap[i]) if i < len(results.box.ap) else 0.0,
                    }
                metrics['class_metrics'] = class_metrics
            
            logger.info("验证完成")
            logger.info(f"mAP50: {metrics['mAP50']:.4f}")
            logger.info(f"mAP50-95: {metrics['mAP50-95']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            
            # 保存验证结果
            results_file = Path(project) / name / "validation_results.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            write_json(metrics, results_file)
            logger.info(f"验证结果保存到: {results_file}")
            
            return metrics
            
    except Exception as e:
        logger.error(f"验证失败: {e}")
        return None

def evaluate_metrics(results: Dict[str, Any], 
                    thresholds: Optional[Dict[str, float]] = None) -> Dict[str, str]:
    """
    评估模型性能指标
    
    Args:
        results: 验证结果
        thresholds: 评估阈值
        
    Returns:
        Dict: 评估结果
    """
    if thresholds is None:
        thresholds = {
            'mAP50': 0.5,
            'mAP50-95': 0.3,
            'precision': 0.7,
            'recall': 0.6
        }
    
    evaluation = {}
    
    for metric, threshold in thresholds.items():
        if metric in results:
            value = results[metric]
            if value >= threshold:
                evaluation[metric] = "优秀"
            elif value >= threshold * 0.8:
                evaluation[metric] = "良好"
            elif value >= threshold * 0.6:
                evaluation[metric] = "一般"
            else:
                evaluation[metric] = "需要改进"
        else:
            evaluation[metric] = "未知"
    
    return evaluation

def compare_models(model_paths: list, 
                  data_config_path: Optional[str] = None,
                  **val_kwargs) -> Dict[str, Any]:
    """
    比较多个模型的性能
    
    Args:
        model_paths: 模型文件路径列表
        data_config_path: 数据配置文件路径
        **val_kwargs: 验证参数
        
    Returns:
        Dict: 比较结果
    """
    try:
        logger.info(f"开始比较{len(model_paths)}个模型")
        
        comparison_results = {}
        
        for i, model_path in enumerate(model_paths):
            model_name = Path(model_path).stem
            logger.info(f"验证模型 {i+1}/{len(model_paths)}: {model_name}")
            
            # 为每个模型创建独立的验证目录
            val_kwargs['name'] = f"compare_{model_name}"
            
            results = validate_model(
                model_path=model_path,
                data_config_path=data_config_path,
                **val_kwargs
            )
            
            if results:
                comparison_results[model_name] = results
            else:
                logger.warning(f"模型验证失败: {model_name}")
        
        # 生成比较报告
        if comparison_results:
            logger.info("模型比较结果:")
            logger.info("-" * 80)
            logger.info(f"{'模型名称':<20} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
            logger.info("-" * 80)
            
            for model_name, results in comparison_results.items():
                logger.info(f"{model_name:<20} "
                          f"{results['mAP50']:<10.4f} "
                          f"{results['mAP50-95']:<10.4f} "
                          f"{results['precision']:<10.4f} "
                          f"{results['recall']:<10.4f}")
            
            # 找出最佳模型
            best_model = max(comparison_results.items(), 
                           key=lambda x: x[1]['mAP50-95'])
            logger.info("-" * 80)
            logger.info(f"最佳模型: {best_model[0]} (mAP50-95: {best_model[1]['mAP50-95']:.4f})")
        
        return comparison_results
        
    except Exception as e:
        logger.error(f"模型比较失败: {e}")
        return {}

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="YOLO模型验证脚本")
    
    # 基本参数
    parser.add_argument("--model", type=str, required=True, help="模型文件路径")
    parser.add_argument("--data", type=str, help="数据配置文件路径")
    parser.add_argument("--img-size", type=int, default=640, help="验证图像尺寸")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--device", type=str, default="", help="验证设备")
    parser.add_argument("--project", type=str, default="runs/val", help="项目目录")
    parser.add_argument("--name", type=str, default="exp", help="实验名称")
    
    # 验证参数
    parser.add_argument("--conf-thres", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU阈值")
    parser.add_argument("--max-det", type=int, default=300, help="最大检测数量")
    
    # 保存选项
    parser.add_argument("--save-json", action="store_true", help="保存COCO格式结果")
    parser.add_argument("--save-hybrid", action="store_true", help="保存混合标签")
    parser.add_argument("--save-conf", action="store_true", help="保存置信度")
    parser.add_argument("--save-txt", action="store_true", help="保存txt格式结果")
    parser.add_argument("--no-plots", action="store_true", help="不保存验证图片")
    
    # 比较模式
    parser.add_argument("--compare", nargs="+", help="比较多个模型")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger("validate", console_output=True, file_output=True)
    
    if args.compare:
        # 比较模式
        results = compare_models(
            model_paths=args.compare,
            data_config_path=args.data,
            img_size=args.img_size,
            batch_size=args.batch_size,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            max_det=args.max_det,
            device=args.device,
            save_json=args.save_json,
            save_hybrid=args.save_hybrid,
            save_conf=args.save_conf,
            save_txt=args.save_txt,
            plots=not args.no_plots,
            project=args.project
        )
        
        if results:
            logger.info("模型比较完成")
            sys.exit(0)
        else:
            logger.error("模型比较失败")
            sys.exit(1)
    else:
        # 单模型验证
        results = validate_model(
            model_path=args.model,
            data_config_path=args.data,
            img_size=args.img_size,
            batch_size=args.batch_size,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            max_det=args.max_det,
            device=args.device,
            save_json=args.save_json,
            save_hybrid=args.save_hybrid,
            save_conf=args.save_conf,
            save_txt=args.save_txt,
            plots=not args.no_plots,
            project=args.project,
            name=args.name
        )
        
        if results:
            # 评估指标
            evaluation = evaluate_metrics(results)
            logger.info("性能评估:")
            for metric, rating in evaluation.items():
                logger.info(f"  {metric}: {rating}")
            
            logger.info("验证成功完成")
            sys.exit(0)
        else:
            logger.error("验证失败")
            sys.exit(1)

if __name__ == "__main__":
    main()
