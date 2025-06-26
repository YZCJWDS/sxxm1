#!/usr/bin/env python3
"""
数据集分析脚本
提供数据集统计、可视化、质量检查等功能
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import json

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from yoloserver.utils.logger import setup_logger, get_logger, TimerLogger, ProgressLogger
from yoloserver.utils.path_utils import get_data_paths, list_files_with_extension
from yoloserver.utils.file_utils import read_yaml, write_json
from yoloserver.utils.data_converter import validate_annotations

logger = get_logger(__name__)

class DatasetAnalyzer:
    """数据集分析器"""
    
    def __init__(self, dataset_config_path: Optional[str] = None):
        """
        初始化数据集分析器
        
        Args:
            dataset_config_path: 数据集配置文件路径
        """
        self.config = self._load_config(dataset_config_path)
        self.class_names = self._get_class_names()
        self.stats = {}
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载数据集配置"""
        if config_path and Path(config_path).exists():
            return read_yaml(config_path)
        else:
            # 使用默认配置
            from yoloserver.utils.path_utils import get_config_paths
            config_paths = get_config_paths()
            return read_yaml(config_paths['dataset_config'])
    
    def _get_class_names(self) -> List[str]:
        """获取类别名称"""
        names = self.config.get('names', {})
        if isinstance(names, dict):
            # 字典格式: {0: 'person', 1: 'car'}
            return [names.get(i, f'class_{i}') for i in range(len(names))]
        elif isinstance(names, list):
            # 列表格式: ['person', 'car']
            return names
        else:
            return []
    
    def analyze_dataset(self, split: str = 'all') -> Dict:
        """
        分析数据集
        
        Args:
            split: 数据集分割 ('train', 'val', 'test', 'all')
            
        Returns:
            Dict: 分析结果
        """
        try:
            with TimerLogger(f"数据集分析 ({split})", logger):
                logger.info(f"开始分析数据集: {split}")
                
                # 获取数据路径
                data_root = Path(self.config.get('path', './data'))
                
                if split == 'all':
                    splits_to_analyze = ['train', 'val', 'test']
                else:
                    splits_to_analyze = [split]
                
                analysis_results = {}
                
                for current_split in splits_to_analyze:
                    logger.info(f"分析 {current_split} 数据集...")
                    
                    # 获取图像和标签路径
                    split_config = self.config.get(current_split, f'{current_split}/images')
                    if isinstance(split_config, str):
                        image_dir = data_root / split_config
                        label_dir = data_root / split_config.replace('images', 'labels')
                    else:
                        image_dir = data_root / f'{current_split}/images'
                        label_dir = data_root / f'{current_split}/labels'
                    
                    if not image_dir.exists():
                        logger.warning(f"{current_split} 图像目录不存在: {image_dir}")
                        continue
                    
                    if not label_dir.exists():
                        logger.warning(f"{current_split} 标签目录不存在: {label_dir}")
                        continue
                    
                    # 分析当前分割
                    split_stats = self._analyze_split(image_dir, label_dir, current_split)
                    analysis_results[current_split] = split_stats
                
                # 生成总体统计
                if len(analysis_results) > 1:
                    analysis_results['summary'] = self._generate_summary(analysis_results)
                
                self.stats = analysis_results
                logger.info("数据集分析完成")
                
                return analysis_results
                
        except Exception as e:
            logger.error(f"数据集分析失败: {e}")
            return {}
    
    def _analyze_split(self, image_dir: Path, label_dir: Path, split_name: str) -> Dict:
        """分析单个数据集分割"""
        stats = {
            'split_name': split_name,
            'image_count': 0,
            'label_count': 0,
            'annotation_count': 0,
            'class_distribution': Counter(),
            'image_sizes': [],
            'bbox_sizes': [],
            'bbox_aspect_ratios': [],
            'images_per_class': defaultdict(set),
            'annotations_per_image': [],
            'empty_images': 0,
            'errors': []
        }
        
        # 获取图像文件
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*.{ext}"))
            image_files.extend(image_dir.glob(f"*.{ext.upper()}"))
        
        stats['image_count'] = len(image_files)
        
        if stats['image_count'] == 0:
            logger.warning(f"在 {image_dir} 中未找到图像文件")
            return stats
        
        # 分析图像和标注
        progress = ProgressLogger(len(image_files), logger)
        
        for image_file in image_files:
            try:
                # 分析图像
                image_stats = self._analyze_image(image_file)
                stats['image_sizes'].append(image_stats['size'])
                
                # 分析对应的标注文件
                label_file = label_dir / f"{image_file.stem}.txt"
                if label_file.exists():
                    stats['label_count'] += 1
                    annotation_stats = self._analyze_annotations(label_file, image_stats['size'])
                    
                    if annotation_stats['count'] == 0:
                        stats['empty_images'] += 1
                    else:
                        stats['annotation_count'] += annotation_stats['count']
                        stats['annotations_per_image'].append(annotation_stats['count'])
                        
                        # 更新类别统计
                        for class_id in annotation_stats['classes']:
                            stats['class_distribution'][class_id] += 1
                            stats['images_per_class'][class_id].add(image_file.stem)
                        
                        # 更新边界框统计
                        stats['bbox_sizes'].extend(annotation_stats['bbox_sizes'])
                        stats['bbox_aspect_ratios'].extend(annotation_stats['aspect_ratios'])
                else:
                    stats['errors'].append(f"缺少标注文件: {image_file.name}")
                
                progress.update()
                
            except Exception as e:
                stats['errors'].append(f"处理文件失败 {image_file.name}: {str(e)}")
                progress.update()
        
        # 转换集合为计数
        stats['images_per_class'] = {k: len(v) for k, v in stats['images_per_class'].items()}
        
        return stats
    
    def _analyze_image(self, image_path: Path) -> Dict:
        """分析单个图像"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("无法读取图像")
            
            height, width = image.shape[:2]
            
            return {
                'size': (width, height),
                'channels': image.shape[2] if len(image.shape) == 3 else 1,
                'file_size': image_path.stat().st_size
            }
            
        except Exception as e:
            logger.warning(f"图像分析失败 {image_path}: {e}")
            return {'size': (0, 0), 'channels': 0, 'file_size': 0}
    
    def _analyze_annotations(self, label_path: Path, image_size: Tuple[int, int]) -> Dict:
        """分析单个标注文件"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            annotations = []
            classes = []
            bbox_sizes = []
            aspect_ratios = []
            
            img_width, img_height = image_size
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    
                    # 计算实际像素尺寸
                    bbox_width = w * img_width
                    bbox_height = h * img_height
                    bbox_area = bbox_width * bbox_height
                    
                    classes.append(class_id)
                    bbox_sizes.append(bbox_area)
                    
                    if bbox_height > 0:
                        aspect_ratios.append(bbox_width / bbox_height)
                    
                    annotations.append({
                        'class_id': class_id,
                        'bbox': [x, y, w, h],
                        'area': bbox_area
                    })
                    
                except ValueError:
                    continue
            
            return {
                'count': len(annotations),
                'classes': classes,
                'bbox_sizes': bbox_sizes,
                'aspect_ratios': aspect_ratios,
                'annotations': annotations
            }
            
        except Exception as e:
            logger.warning(f"标注分析失败 {label_path}: {e}")
            return {'count': 0, 'classes': [], 'bbox_sizes': [], 'aspect_ratios': [], 'annotations': []}
    
    def _generate_summary(self, analysis_results: Dict) -> Dict:
        """生成总体统计摘要"""
        summary = {
            'total_images': 0,
            'total_annotations': 0,
            'total_classes': len(self.class_names),
            'class_distribution': Counter(),
            'split_distribution': {}
        }
        
        for split_name, split_stats in analysis_results.items():
            if split_name == 'summary':
                continue
            
            summary['total_images'] += split_stats['image_count']
            summary['total_annotations'] += split_stats['annotation_count']
            summary['split_distribution'][split_name] = split_stats['image_count']
            
            # 合并类别分布
            for class_id, count in split_stats['class_distribution'].items():
                summary['class_distribution'][class_id] += count
        
        return summary
    
    def generate_report(self, output_dir: Optional[str] = None) -> str:
        """
        生成分析报告
        
        Args:
            output_dir: 输出目录
            
        Returns:
            str: 报告文件路径
        """
        if not self.stats:
            logger.error("请先运行数据集分析")
            return ""
        
        try:
            if output_dir is None:
                output_dir = Path("dataset_analysis")
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成JSON报告
            report_file = output_dir / "dataset_analysis_report.json"
            write_json(self.stats, report_file)
            
            # 生成可视化图表
            self._generate_visualizations(output_dir)
            
            # 生成文本报告
            text_report = self._generate_text_report()
            text_report_file = output_dir / "dataset_analysis_report.txt"
            with open(text_report_file, 'w', encoding='utf-8') as f:
                f.write(text_report)
            
            logger.info(f"分析报告已生成: {output_dir}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            return ""
    
    def _generate_visualizations(self, output_dir: Path):
        """生成可视化图表"""
        try:
            plt.style.use('default')
            
            # 1. 类别分布图
            if 'summary' in self.stats and self.stats['summary']['class_distribution']:
                self._plot_class_distribution(output_dir)
            
            # 2. 数据集分割分布图
            if 'summary' in self.stats and self.stats['summary']['split_distribution']:
                self._plot_split_distribution(output_dir)
            
            # 3. 边界框尺寸分布图
            self._plot_bbox_distributions(output_dir)
            
            # 4. 图像尺寸分布图
            self._plot_image_size_distribution(output_dir)
            
            logger.info("可视化图表生成完成")
            
        except Exception as e:
            logger.error(f"生成可视化图表失败: {e}")
    
    def _plot_class_distribution(self, output_dir: Path):
        """绘制类别分布图"""
        class_dist = self.stats['summary']['class_distribution']
        
        # 准备数据
        class_ids = list(class_dist.keys())
        counts = list(class_dist.values())
        class_labels = [self.class_names[i] if i < len(self.class_names) else f'class_{i}' 
                       for i in class_ids]
        
        # 绘制柱状图
        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_labels, counts)
        plt.title('类别分布')
        plt.xlabel('类别')
        plt.ylabel('实例数量')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_split_distribution(self, output_dir: Path):
        """绘制数据集分割分布图"""
        split_dist = self.stats['summary']['split_distribution']
        
        # 绘制饼图
        plt.figure(figsize=(8, 8))
        plt.pie(split_dist.values(), labels=split_dist.keys(), autopct='%1.1f%%', startangle=90)
        plt.title('数据集分割分布')
        plt.axis('equal')
        plt.savefig(output_dir / 'split_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bbox_distributions(self, output_dir: Path):
        """绘制边界框分布图"""
        # 收集所有边界框数据
        all_bbox_sizes = []
        all_aspect_ratios = []
        
        for split_name, split_stats in self.stats.items():
            if split_name == 'summary':
                continue
            all_bbox_sizes.extend(split_stats.get('bbox_sizes', []))
            all_aspect_ratios.extend(split_stats.get('bbox_aspect_ratios', []))
        
        if all_bbox_sizes:
            # 边界框面积分布
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(all_bbox_sizes, bins=50, alpha=0.7)
            plt.title('边界框面积分布')
            plt.xlabel('面积 (像素²)')
            plt.ylabel('频次')
            plt.yscale('log')
            
            plt.subplot(1, 2, 2)
            plt.hist(all_aspect_ratios, bins=50, alpha=0.7)
            plt.title('边界框宽高比分布')
            plt.xlabel('宽高比')
            plt.ylabel('频次')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'bbox_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_image_size_distribution(self, output_dir: Path):
        """绘制图像尺寸分布图"""
        # 收集所有图像尺寸数据
        all_widths = []
        all_heights = []
        
        for split_name, split_stats in self.stats.items():
            if split_name == 'summary':
                continue
            for width, height in split_stats.get('image_sizes', []):
                all_widths.append(width)
                all_heights.append(height)
        
        if all_widths:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(all_widths, bins=30, alpha=0.7, label='宽度')
            plt.hist(all_heights, bins=30, alpha=0.7, label='高度')
            plt.title('图像尺寸分布')
            plt.xlabel('像素')
            plt.ylabel('频次')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.scatter(all_widths, all_heights, alpha=0.5, s=1)
            plt.title('图像尺寸散点图')
            plt.xlabel('宽度')
            plt.ylabel('高度')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'image_size_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_text_report(self) -> str:
        """生成文本报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("数据集分析报告")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        if 'summary' in self.stats:
            summary = self.stats['summary']
            report_lines.append("总体统计:")
            report_lines.append(f"  总图像数量: {summary['total_images']}")
            report_lines.append(f"  总标注数量: {summary['total_annotations']}")
            report_lines.append(f"  类别数量: {summary['total_classes']}")
            report_lines.append("")
            
            # 数据集分割分布
            if summary['split_distribution']:
                report_lines.append("数据集分割分布:")
                for split, count in summary['split_distribution'].items():
                    percentage = count / summary['total_images'] * 100
                    report_lines.append(f"  {split}: {count} ({percentage:.1f}%)")
                report_lines.append("")
            
            # 类别分布
            if summary['class_distribution']:
                report_lines.append("类别分布:")
                for class_id, count in summary['class_distribution'].most_common():
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                    percentage = count / summary['total_annotations'] * 100
                    report_lines.append(f"  {class_name}: {count} ({percentage:.1f}%)")
                report_lines.append("")
        
        # 各分割详细统计
        for split_name, split_stats in self.stats.items():
            if split_name == 'summary':
                continue
            
            report_lines.append(f"{split_name.upper()} 数据集:")
            report_lines.append(f"  图像数量: {split_stats['image_count']}")
            report_lines.append(f"  标注数量: {split_stats['annotation_count']}")
            report_lines.append(f"  空图像数量: {split_stats['empty_images']}")
            
            if split_stats['annotations_per_image']:
                avg_annotations = np.mean(split_stats['annotations_per_image'])
                report_lines.append(f"  平均每图标注数: {avg_annotations:.2f}")
            
            if split_stats['errors']:
                report_lines.append(f"  错误数量: {len(split_stats['errors'])}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据集分析脚本")
    
    parser.add_argument("--config", type=str, help="数据集配置文件路径")
    parser.add_argument("--split", type=str, default="all", 
                       choices=['train', 'val', 'test', 'all'], help="要分析的数据集分割")
    parser.add_argument("--output", type=str, default="dataset_analysis", help="输出目录")
    parser.add_argument("--no-visualizations", action="store_true", help="不生成可视化图表")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger("dataset_analyzer", console_output=True, file_output=True)
    
    # 创建分析器
    analyzer = DatasetAnalyzer(args.config)
    
    # 执行分析
    results = analyzer.analyze_dataset(args.split)
    
    if results:
        # 生成报告
        if not args.no_visualizations:
            report_file = analyzer.generate_report(args.output)
            if report_file:
                logger.info(f"分析完成，报告已保存: {report_file}")
            else:
                logger.error("报告生成失败")
        else:
            # 只输出统计信息
            logger.info("数据集分析结果:")
            if 'summary' in results:
                summary = results['summary']
                logger.info(f"总图像数量: {summary['total_images']}")
                logger.info(f"总标注数量: {summary['total_annotations']}")
                logger.info(f"类别数量: {summary['total_classes']}")
        
        sys.exit(0)
    else:
        logger.error("数据集分析失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
