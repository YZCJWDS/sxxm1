#!/usr/bin/env python3
"""
模型推理脚本
提供YOLO模型推理功能，支持图像、视频和批量推理
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import cv2
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from yoloserver.utils.logger import setup_logger, get_logger, TimerLogger, ProgressLogger
from yoloserver.utils.path_utils import get_project_root, list_files_with_extension
from yoloserver.utils.file_utils import read_yaml

logger = get_logger(__name__)

def predict_image(model_path: str,
                 image_path: str,
                 output_dir: Optional[str] = None,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 max_det: int = 1000,
                 device: str = "",
                 save_txt: bool = False,
                 save_conf: bool = False,
                 save_crop: bool = False,
                 show_labels: bool = True,
                 show_conf: bool = True,
                 line_width: Optional[int] = None,
                 **kwargs) -> Optional[Dict[str, Any]]:
    """
    对单张图像进行推理
    
    Args:
        model_path: 模型文件路径
        image_path: 图像文件路径
        output_dir: 输出目录
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        max_det: 最大检测数量
        device: 推理设备
        save_txt: 保存txt格式结果
        save_conf: 保存置信度
        save_crop: 保存裁剪的检测框
        show_labels: 显示标签
        show_conf: 显示置信度
        line_width: 边界框线宽
        **kwargs: 其他推理参数
        
    Returns:
        Optional[Dict]: 推理结果，失败返回None
    """
    try:
        logger.info(f"开始推理图像: {image_path}")
        
        # 检查文件
        model_path = Path(model_path)
        image_path = Path(image_path)
        
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return None
        
        if not image_path.exists():
            logger.error(f"图像文件不存在: {image_path}")
            return None
        
        # 检查是否安装了ultralytics
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("请安装ultralytics: pip install ultralytics")
            return None
        
        # 加载模型
        model = YOLO(str(model_path))
        
        # 推理参数
        predict_config = {
            'source': str(image_path),
            'conf': conf_thres,
            'iou': iou_thres,
            'max_det': max_det,
            'device': device,
            'save_txt': save_txt,
            'save_conf': save_conf,
            'save_crop': save_crop,
            'show_labels': show_labels,
            'show_conf': show_conf,
        }
        
        if output_dir:
            predict_config['project'] = str(Path(output_dir).parent)
            predict_config['name'] = Path(output_dir).name
        
        if line_width:
            predict_config['line_width'] = line_width
        
        predict_config.update(kwargs)
        
        # 执行推理
        results = model.predict(**predict_config)
        
        # 提取结果信息
        if results and len(results) > 0:
            result = results[0]
            
            detection_info = {
                'image_path': str(image_path),
                'image_shape': result.orig_shape,
                'detection_count': len(result.boxes) if result.boxes is not None else 0,
                'detections': []
            }
            
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]),
                        'class_name': model.names[int(box.cls[0])]
                    }
                    detection_info['detections'].append(detection)
            
            logger.info(f"检测到 {detection_info['detection_count']} 个目标")
            return detection_info
        
        return None
        
    except Exception as e:
        logger.error(f"图像推理失败: {e}")
        return None

def predict_batch(model_path: str,
                 input_dir: str,
                 output_dir: Optional[str] = None,
                 image_extensions: List[str] = None,
                 **kwargs) -> List[Dict[str, Any]]:
    """
    批量推理图像
    
    Args:
        model_path: 模型文件路径
        input_dir: 输入目录
        output_dir: 输出目录
        image_extensions: 图像文件扩展名
        **kwargs: 其他推理参数
        
    Returns:
        List[Dict]: 推理结果列表
    """
    try:
        if image_extensions is None:
            image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        
        input_dir = Path(input_dir)
        
        # 获取所有图像文件
        image_files = list_files_with_extension(input_dir, image_extensions, recursive=True)
        
        if not image_files:
            logger.warning(f"在 {input_dir} 中未找到图像文件")
            return []
        
        logger.info(f"找到 {len(image_files)} 个图像文件")
        
        results = []
        progress = ProgressLogger(len(image_files), logger)
        
        for image_file in image_files:
            result = predict_image(
                model_path=model_path,
                image_path=str(image_file),
                output_dir=output_dir,
                **kwargs
            )
            
            if result:
                results.append(result)
            
            progress.update()
        
        logger.info(f"批量推理完成，成功处理 {len(results)}/{len(image_files)} 个图像")
        return results
        
    except Exception as e:
        logger.error(f"批量推理失败: {e}")
        return []

def predict_video(model_path: str,
                 video_path: str,
                 output_path: Optional[str] = None,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 max_det: int = 1000,
                 device: str = "",
                 show_labels: bool = True,
                 show_conf: bool = True,
                 line_width: Optional[int] = None,
                 save_frames: bool = False,
                 **kwargs) -> bool:
    """
    对视频进行推理
    
    Args:
        model_path: 模型文件路径
        video_path: 视频文件路径
        output_path: 输出视频路径
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        max_det: 最大检测数量
        device: 推理设备
        show_labels: 显示标签
        show_conf: 显示置信度
        line_width: 边界框线宽
        save_frames: 保存检测帧
        **kwargs: 其他推理参数
        
    Returns:
        bool: 推理是否成功
    """
    try:
        logger.info(f"开始推理视频: {video_path}")
        
        # 检查文件
        model_path = Path(model_path)
        video_path = Path(video_path)
        
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            return False
        
        # 检查是否安装了ultralytics
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("请安装ultralytics: pip install ultralytics")
            return False
        
        # 加载模型
        model = YOLO(str(model_path))
        
        # 推理参数
        predict_config = {
            'source': str(video_path),
            'conf': conf_thres,
            'iou': iou_thres,
            'max_det': max_det,
            'device': device,
            'show_labels': show_labels,
            'show_conf': show_conf,
            'save': True,  # 保存结果视频
        }
        
        if output_path:
            predict_config['project'] = str(Path(output_path).parent)
            predict_config['name'] = Path(output_path).stem
        
        if line_width:
            predict_config['line_width'] = line_width
        
        if save_frames:
            predict_config['save_frames'] = True
        
        predict_config.update(kwargs)
        
        # 执行推理
        with TimerLogger("视频推理", logger):
            results = model.predict(**predict_config)
        
        logger.info("视频推理完成")
        return True
        
    except Exception as e:
        logger.error(f"视频推理失败: {e}")
        return False

def predict_webcam(model_path: str,
                  camera_id: int = 0,
                  conf_thres: float = 0.25,
                  iou_thres: float = 0.45,
                  max_det: int = 1000,
                  device: str = "",
                  show_labels: bool = True,
                  show_conf: bool = True,
                  line_width: Optional[int] = None,
                  **kwargs) -> bool:
    """
    实时摄像头推理
    
    Args:
        model_path: 模型文件路径
        camera_id: 摄像头ID
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        max_det: 最大检测数量
        device: 推理设备
        show_labels: 显示标签
        show_conf: 显示置信度
        line_width: 边界框线宽
        **kwargs: 其他推理参数
        
    Returns:
        bool: 推理是否成功
    """
    try:
        logger.info(f"开始实时推理，摄像头ID: {camera_id}")
        
        # 检查模型文件
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        # 检查是否安装了ultralytics
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("请安装ultralytics: pip install ultralytics")
            return False
        
        # 加载模型
        model = YOLO(str(model_path))
        
        # 推理参数
        predict_config = {
            'source': camera_id,
            'conf': conf_thres,
            'iou': iou_thres,
            'max_det': max_det,
            'device': device,
            'show_labels': show_labels,
            'show_conf': show_conf,
            'show': True,  # 显示实时画面
        }
        
        if line_width:
            predict_config['line_width'] = line_width
        
        predict_config.update(kwargs)
        
        # 执行实时推理
        logger.info("开始实时推理，按 'q' 键退出")
        results = model.predict(**predict_config)
        
        logger.info("实时推理结束")
        return True
        
    except Exception as e:
        logger.error(f"实时推理失败: {e}")
        return False

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="YOLO模型推理脚本")
    
    # 基本参数
    parser.add_argument("--model", type=str, required=True, help="模型文件路径")
    parser.add_argument("--source", type=str, help="输入源（图像/视频/目录路径，或摄像头ID）")
    parser.add_argument("--output", type=str, help="输出路径")
    parser.add_argument("--device", type=str, default="", help="推理设备")
    
    # 推理参数
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU阈值")
    parser.add_argument("--max-det", type=int, default=1000, help="最大检测数量")
    parser.add_argument("--line-width", type=int, help="边界框线宽")
    
    # 显示选项
    parser.add_argument("--no-labels", action="store_true", help="不显示标签")
    parser.add_argument("--no-conf", action="store_true", help="不显示置信度")
    
    # 保存选项
    parser.add_argument("--save-txt", action="store_true", help="保存txt格式结果")
    parser.add_argument("--save-conf", action="store_true", help="保存置信度")
    parser.add_argument("--save-crop", action="store_true", help="保存裁剪的检测框")
    parser.add_argument("--save-frames", action="store_true", help="保存视频帧")
    
    # 模式选择
    parser.add_argument("--webcam", action="store_true", help="摄像头模式")
    parser.add_argument("--batch", action="store_true", help="批量处理模式")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger("inference", console_output=True, file_output=True)
    
    # 推理参数
    inference_kwargs = {
        'conf_thres': args.conf_thres,
        'iou_thres': args.iou_thres,
        'max_det': args.max_det,
        'device': args.device,
        'show_labels': not args.no_labels,
        'show_conf': not args.no_conf,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'save_crop': args.save_crop,
    }
    
    if args.line_width:
        inference_kwargs['line_width'] = args.line_width
    
    success = False
    
    if args.webcam:
        # 摄像头模式
        camera_id = int(args.source) if args.source and args.source.isdigit() else 0
        success = predict_webcam(
            model_path=args.model,
            camera_id=camera_id,
            **inference_kwargs
        )
    elif args.batch:
        # 批量处理模式
        if not args.source:
            logger.error("批量模式需要指定输入目录")
            sys.exit(1)
        
        results = predict_batch(
            model_path=args.model,
            input_dir=args.source,
            output_dir=args.output,
            **inference_kwargs
        )
        success = len(results) > 0
    else:
        # 单文件模式
        if not args.source:
            logger.error("需要指定输入源")
            sys.exit(1)
        
        source_path = Path(args.source)
        
        if source_path.is_file():
            # 检查是否为视频文件
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            if source_path.suffix.lower() in video_extensions:
                # 视频推理
                success = predict_video(
                    model_path=args.model,
                    video_path=args.source,
                    output_path=args.output,
                    save_frames=args.save_frames,
                    **inference_kwargs
                )
            else:
                # 图像推理
                result = predict_image(
                    model_path=args.model,
                    image_path=args.source,
                    output_dir=args.output,
                    **inference_kwargs
                )
                success = result is not None
        else:
            logger.error(f"输入源不存在: {args.source}")
            sys.exit(1)
    
    if success:
        logger.info("推理成功完成")
        sys.exit(0)
    else:
        logger.error("推理失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
