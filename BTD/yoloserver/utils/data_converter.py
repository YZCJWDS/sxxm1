"""
数据转换工具模块
提供各种标注格式之间的转换功能
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Union, Optional
import shutil
import random
from PIL import Image

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .logger import get_logger
except ImportError:
    # 直接运行时使用绝对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from yoloserver.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> None:
    """确保目录存在，如果不存在则创建"""
    Path(path).mkdir(parents=True, exist_ok=True)

class AnnotationConverter:
    """标注格式转换器基类"""
    
    def __init__(self):
        self.class_names = []
        self.class_to_id = {}
        self.id_to_class = {}
    
    def set_classes(self, class_names: List[str]):
        """设置类别名称"""
        self.class_names = class_names
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
        self.id_to_class = {idx: name for idx, name in enumerate(class_names)}

class COCOToYOLOConverter(AnnotationConverter):
    """COCO格式到YOLO格式转换器"""
    
    def convert_annotation_file(self, coco_json_path: Union[str, Path], 
                              output_dir: Union[str, Path]) -> bool:
        """
        转换COCO JSON文件到YOLO格式
        
        Args:
            coco_json_path: COCO JSON文件路径
            output_dir: 输出目录
            
        Returns:
            bool: 转换是否成功
        """
        try:
            with open(coco_json_path, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            # 提取类别信息
            categories = coco_data.get('categories', [])
            class_names = [cat['name'] for cat in categories]
            self.set_classes(class_names)
            
            # 创建输出目录
            output_dir = Path(output_dir)
            ensure_dir(output_dir)
            
            # 创建图像ID到文件名的映射
            images = coco_data.get('images', [])
            image_id_to_info = {img['id']: img for img in images}
            
            # 按图像分组标注
            annotations = coco_data.get('annotations', [])
            image_annotations = {}
            for ann in annotations:
                image_id = ann['image_id']
                if image_id not in image_annotations:
                    image_annotations[image_id] = []
                image_annotations[image_id].append(ann)
            
            # 转换每个图像的标注
            for image_id, anns in image_annotations.items():
                if image_id not in image_id_to_info:
                    continue
                
                image_info = image_id_to_info[image_id]
                image_width = image_info['width']
                image_height = image_info['height']
                image_filename = Path(image_info['file_name']).stem
                
                # 转换标注
                yolo_lines = []
                for ann in anns:
                    bbox = ann['bbox']  # [x, y, width, height]
                    category_id = ann['category_id']
                    
                    # 转换为YOLO格式
                    yolo_bbox = self._coco_bbox_to_yolo(bbox, image_width, image_height)
                    yolo_class_id = self._coco_category_to_yolo_class(category_id, categories)
                    
                    yolo_line = f"{yolo_class_id} {' '.join(map(str, yolo_bbox))}"
                    yolo_lines.append(yolo_line)
                
                # 保存YOLO标注文件
                output_file = output_dir / f"{image_filename}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))
            
            logger.info(f"成功转换COCO标注到YOLO格式，输出目录: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"COCO到YOLO转换失败: {e}")
            return False
    
    def _coco_bbox_to_yolo(self, coco_bbox: List[float], 
                          image_width: int, image_height: int) -> List[float]:
        """将COCO边界框转换为YOLO格式"""
        x, y, width, height = coco_bbox
        
        # 计算中心点坐标
        center_x = x + width / 2
        center_y = y + height / 2
        
        # 归一化
        center_x /= image_width
        center_y /= image_height
        width /= image_width
        height /= image_height
        
        return [center_x, center_y, width, height]
    
    def _coco_category_to_yolo_class(self, category_id: int, categories: List[Dict]) -> int:
        """将COCO类别ID转换为YOLO类别ID"""
        for idx, cat in enumerate(categories):
            if cat['id'] == category_id:
                return idx
        return 0

class PascalVOCToYOLOConverter(AnnotationConverter):
    """Pascal VOC格式到YOLO格式转换器"""
    
    def convert_annotation_file(self, xml_path: Union[str, Path], 
                              output_path: Union[str, Path],
                              image_width: int, image_height: int) -> bool:
        """
        转换单个Pascal VOC XML文件到YOLO格式
        
        Args:
            xml_path: XML文件路径
            output_path: 输出文件路径
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            bool: 转换是否成功
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            yolo_lines = []
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in self.class_to_id:
                    logger.warning(f"未知类别: {class_name}")
                    continue
                
                class_id = self.class_to_id[class_name]
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # 转换为YOLO格式
                center_x = (xmin + xmax) / 2 / image_width
                center_y = (ymin + ymax) / 2 / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height
                
                yolo_line = f"{class_id} {center_x} {center_y} {width} {height}"
                yolo_lines.append(yolo_line)
            
            # 保存YOLO标注文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
            
            return True
            
        except Exception as e:
            logger.error(f"Pascal VOC到YOLO转换失败: {e}")
            return False

def convert_coco_to_yolo(coco_json_path: Union[str, Path], 
                        output_dir: Union[str, Path]) -> bool:
    """
    转换COCO格式标注到YOLO格式
    
    Args:
        coco_json_path: COCO JSON文件路径
        output_dir: 输出目录
        
    Returns:
        bool: 转换是否成功
    """
    converter = COCOToYOLOConverter()
    return converter.convert_annotation_file(coco_json_path, output_dir)

def convert_pascal_to_yolo(xml_dir: Union[str, Path], 
                          output_dir: Union[str, Path],
                          class_names: List[str],
                          image_dir: Optional[Union[str, Path]] = None) -> bool:
    """
    批量转换Pascal VOC格式标注到YOLO格式
    
    Args:
        xml_dir: XML文件目录
        output_dir: 输出目录
        class_names: 类别名称列表
        image_dir: 图像目录，用于获取图像尺寸
        
    Returns:
        bool: 转换是否成功
    """
    try:
        xml_dir = Path(xml_dir)
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        converter = PascalVOCToYOLOConverter()
        converter.set_classes(class_names)
        
        xml_files = list(xml_dir.glob("*.xml"))
        
        for xml_file in xml_files:
            # 获取对应的图像文件
            image_file = None
            if image_dir:
                image_dir = Path(image_dir)
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    potential_image = image_dir / f"{xml_file.stem}{ext}"
                    if potential_image.exists():
                        image_file = potential_image
                        break
            
            # 获取图像尺寸
            if image_file and image_file.exists():
                with Image.open(image_file) as img:
                    image_width, image_height = img.size
            else:
                # 如果找不到图像文件，尝试从XML中读取尺寸
                tree = ET.parse(xml_file)
                root = tree.getroot()
                size = root.find('size')
                if size is not None:
                    image_width = int(size.find('width').text)
                    image_height = int(size.find('height').text)
                else:
                    logger.warning(f"无法获取图像尺寸: {xml_file}")
                    continue
            
            # 转换标注
            output_file = output_dir / f"{xml_file.stem}.txt"
            converter.convert_annotation_file(xml_file, output_file, image_width, image_height)
        
        logger.info(f"成功转换{len(xml_files)}个Pascal VOC标注文件到YOLO格式")
        return True

    except Exception as e:
        logger.error(f"Pascal VOC到YOLO批量转换失败: {e}")
        return False

def organize_valid_pairs(image_dir: Union[str, Path],
                        label_dir: Union[str, Path],
                        output_dir: Union[str, Path]) -> bool:
    """
    整理有效的图像-标签对到processed目录

    Args:
        image_dir: 原始图像目录
        label_dir: 标签目录
        output_dir: 输出目录(processed)

    Returns:
        bool: 整理是否成功
    """
    try:
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        output_dir = Path(output_dir)

        # 创建输出目录
        output_images_dir = output_dir / 'images'
        output_labels_dir = output_dir / 'labels'
        ensure_dir(output_images_dir)
        ensure_dir(output_labels_dir)

        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))

        # 去重
        image_files = list(set(image_files))

        if not image_files:
            logger.error(f"在{image_dir}中未找到图像文件")
            return False

        # 过滤有对应标签文件的图像
        valid_pairs = []
        for image_file in image_files:
            label_file = label_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                valid_pairs.append((image_file, label_file))

        if not valid_pairs:
            logger.error("未找到有效的图像-标签对")
            return False

        logger.info(f"找到{len(valid_pairs)}个有效的图像-标签对")

        # 复制有效的图像和标签文件
        for image_file, label_file in valid_pairs:
            # 复制图像文件
            dst_image = output_images_dir / image_file.name
            shutil.copy2(image_file, dst_image)

            # 复制标签文件
            dst_label = output_labels_dir / label_file.name
            shutil.copy2(label_file, dst_label)

        logger.info(f"有效数据整理完成，输出目录: {output_dir}")
        logger.info(f"有效图像: {len(valid_pairs)}张")
        return True

    except Exception as e:
        logger.error(f"有效数据整理失败: {e}")
        return False


def split_dataset(image_dir: Union[str, Path],
                 label_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 seed: int = 42,
                 stratify: bool = True) -> bool:
    """
    分割数据集为训练集、验证集和测试集

    Args:
        image_dir: 图像目录
        label_dir: 标签目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        stratify: 是否使用分层分割（按类别比例分割），默认True

    Returns:
        bool: 分割是否成功
    """
    try:
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        output_dir = Path(output_dir)

        # 检查比例总和
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            logger.error("训练集、验证集和测试集比例总和必须为1.0")
            return False

        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))

        # 去重（防止Windows系统大小写不敏感导致的重复）
        image_files = list(set(image_files))

        if not image_files:
            logger.error(f"在{image_dir}中未找到图像文件")
            return False

        # 过滤有对应标签文件的图像
        valid_pairs = []
        for image_file in image_files:
            label_file = label_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                valid_pairs.append((image_file, label_file))

        if not valid_pairs:
            logger.error("未找到有效的图像-标签对")
            return False

        logger.info(f"找到{len(valid_pairs)}个有效的图像-标签对")

        # 清理之前的输出目录
        if output_dir.exists():
            logger.info("清理之前的分割数据...")
            for split_name in ['train', 'val', 'test']:
                split_dir = output_dir / split_name
                if split_dir.exists():
                    shutil.rmtree(split_dir)
                    logger.info(f"已清理: {split_dir}")

        # 根据stratify参数选择分割方式
        if stratify:
            logger.info("使用分层分割（按类别比例分割）")
            return _stratified_split(valid_pairs, train_ratio, val_ratio, test_ratio, seed, output_dir)
        else:
            logger.info("使用随机分割")
            return _random_split(valid_pairs, train_ratio, val_ratio, test_ratio, seed, output_dir)

    except Exception as e:
        logger.error(f"数据集分割失败: {e}")
        return False


def _stratified_split(valid_pairs: List[tuple],
                     train_ratio: float,
                     val_ratio: float,
                     test_ratio: float,
                     seed: int,
                     output_dir: Path) -> bool:
    """
    分层分割：按类别比例分割数据集

    Args:
        valid_pairs: 有效的图像-标签对列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        output_dir: 输出目录

    Returns:
        bool: 分割是否成功
    """
    from collections import defaultdict

    try:
        # 按类别分组
        class_groups = defaultdict(list)
        class_names = {0: 'objects', 1: 'glioma_tumor', 2: 'meningioma_tumor', 3: 'pituitary_tumor'}

        for image_file, label_file in valid_pairs:
            # 读取标注文件获取主要类别（取第一个标注的类别）
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        class_id = int(first_line.split()[0])
                        class_groups[class_id].append((image_file, label_file))
                    else:
                        # 如果标注文件为空，归类到objects类别
                        class_groups[0].append((image_file, label_file))
            except (ValueError, IndexError) as e:
                logger.warning(f"解析标注文件失败 {label_file}: {e}，归类到objects类别")
                class_groups[0].append((image_file, label_file))

        # 显示类别分布
        logger.info("原始数据类别分布:")
        total_samples = 0
        for class_id, pairs in class_groups.items():
            class_name = class_names.get(class_id, f'class_{class_id}')
            logger.info(f"  {class_name} (ID:{class_id}): {len(pairs)} 个样本")
            total_samples += len(pairs)

        # 对每个类别分别分割
        train_pairs, val_pairs, test_pairs = [], [], []

        for class_id, pairs in class_groups.items():
            if not pairs:
                continue

            # 设置随机种子并打乱
            random.seed(seed + class_id)  # 每个类别使用不同的种子
            random.shuffle(pairs)

            total = len(pairs)
            train_count = max(1, int(total * train_ratio))  # 至少保证1个样本
            val_count = max(0, int(total * val_ratio))
            test_count = total - train_count - val_count

            # 分割当前类别的数据
            class_train = pairs[:train_count]
            class_val = pairs[train_count:train_count + val_count]
            class_test = pairs[train_count + val_count:]

            train_pairs.extend(class_train)
            val_pairs.extend(class_val)
            test_pairs.extend(class_test)

            class_name = class_names.get(class_id, f'class_{class_id}')
            logger.info(f"  {class_name}: 训练{len(class_train)}, 验证{len(class_val)}, 测试{len(class_test)}")

        # 最后随机打乱各个数据集（保持类别分布的同时增加随机性）
        random.seed(seed)
        random.shuffle(train_pairs)
        random.shuffle(val_pairs)
        random.shuffle(test_pairs)

        # 验证分割结果
        total_split = len(train_pairs) + len(val_pairs) + len(test_pairs)
        logger.info(f"分层分割完成: 总样本{total_samples}, 分割后{total_split}")
        logger.info(f"最终分布: 训练{len(train_pairs)}, 验证{len(val_pairs)}, 测试{len(test_pairs)}")

        if total_split != total_samples:
            logger.error(f"分割错误: 总样本数不匹配! 原始:{total_samples}, 分割后:{total_split}")
            return False

        # 复制文件到对应目录
        return _copy_files_to_splits(train_pairs, val_pairs, test_pairs, output_dir)

    except Exception as e:
        logger.error(f"分层分割失败: {e}")
        return False


def _random_split(valid_pairs: List[tuple],
                 train_ratio: float,
                 val_ratio: float,
                 test_ratio: float,
                 seed: int,
                 output_dir: Path) -> bool:
    """
    随机分割：随机分割数据集

    Args:
        valid_pairs: 有效的图像-标签对列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        output_dir: 输出目录

    Returns:
        bool: 分割是否成功
    """
    try:
        # 随机打乱
        random.seed(seed)
        random.shuffle(valid_pairs)

        # 计算分割点
        total_count = len(valid_pairs)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        # test_count = total_count - train_count - val_count  # 自动计算

        # 分割数据
        train_pairs = valid_pairs[:train_count]
        val_pairs = valid_pairs[train_count:train_count + val_count]
        test_pairs = valid_pairs[train_count + val_count:]

        # 验证分割结果
        total_split = len(train_pairs) + len(val_pairs) + len(test_pairs)
        logger.info(f"随机分割完成: 总样本{total_count}, 分割后{total_split}")
        logger.info(f"最终分布: 训练{len(train_pairs)}, 验证{len(val_pairs)}, 测试{len(test_pairs)}")

        if total_split != total_count:
            logger.error(f"分割错误: 总样本数不匹配! 原始:{total_count}, 分割后:{total_split}")
            return False

        # 复制文件到对应目录
        return _copy_files_to_splits(train_pairs, val_pairs, test_pairs, output_dir)

    except Exception as e:
        logger.error(f"随机分割失败: {e}")
        return False


def _copy_files_to_splits(train_pairs: List[tuple],
                         val_pairs: List[tuple],
                         test_pairs: List[tuple],
                         output_dir: Path) -> bool:
    """
    复制文件到对应的分割目录

    Args:
        train_pairs: 训练集图像-标签对
        val_pairs: 验证集图像-标签对
        test_pairs: 测试集图像-标签对
        output_dir: 输出目录

    Returns:
        bool: 复制是否成功
    """
    try:
        # 创建输出目录
        splits = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }

        for split_name, pairs in splits.items():
            if not pairs:
                logger.info(f"{split_name}集为空，跳过")
                continue

            split_image_dir = output_dir / split_name / 'images'
            split_label_dir = output_dir / split_name / 'labels'
            ensure_dir(split_image_dir)
            ensure_dir(split_label_dir)

            # 复制文件
            for image_file, label_file in pairs:
                # 复制图像文件
                dst_image = split_image_dir / image_file.name
                shutil.copy2(image_file, dst_image)

                # 复制标签文件
                dst_label = split_label_dir / label_file.name
                shutil.copy2(label_file, dst_label)

            logger.info(f"{split_name}集: {len(pairs)}个样本已复制到 {split_image_dir.parent}")

        logger.info(f"所有文件复制完成，输出目录: {output_dir}")
        return True

    except Exception as e:
        logger.error(f"文件复制失败: {e}")
        return False

def convert_coco_ultralytics_style(labels_dir: Union[str, Path],
                                   save_dir: Union[str, Path],
                                   use_segments: bool = False,
                                   copy_images: bool = True,
                                   specific_file: Optional[str] = None,
                                   file_pattern: str = "*.json") -> bool:
    """
    Ultralytics风格的COCO转换函数
    模拟ultralytics.data.converter.convert_coco的功能

    Args:
        labels_dir: COCO JSON标注文件所在目录
        save_dir: 输出目录
        use_segments: 是否转换分割标注（暂不支持）
        copy_images: 是否复制图像文件
        specific_file: 指定转换的文件名（可选）
        file_pattern: 文件匹配模式（默认"*.json"）

    Returns:
        bool: 转换是否成功
    """
    try:
        labels_dir = Path(labels_dir)
        save_dir = Path(save_dir)

        logger.info(f"开始COCO到YOLO转换...")
        logger.info(f"输入目录: {labels_dir}")
        logger.info(f"输出目录: {save_dir}")
        logger.info(f"分割模式: {use_segments}")

        if use_segments:
            logger.warning("分割模式暂不支持，将转换为边界框格式")

        # 查找COCO JSON文件 - 改进的文件选择逻辑
        if specific_file:
            # 转换指定文件
            specific_path = labels_dir / specific_file
            if not specific_path.exists():
                logger.error(f"指定的文件不存在: {specific_path}")
                return False
            json_files = [specific_path]
            logger.info(f"转换指定文件: {specific_file}")
        else:
            # 使用模式匹配查找文件
            json_files = list(labels_dir.glob(file_pattern))
            if not json_files:
                logger.error(f"在{labels_dir}中未找到匹配'{file_pattern}'的文件")
                return False
            logger.info(f"找到{len(json_files)}个匹配文件: {[f.name for f in json_files]}")

            # 如果找到多个文件，警告用户
            if len(json_files) > 1:
                logger.warning("⚠️  找到多个JSON文件，将全部转换。类别信息将被最后处理的文件覆盖！")
                logger.warning("💡 建议使用 specific_file 参数指定单个文件，或使用 file_pattern 过滤")
                for i, f in enumerate(json_files):
                    logger.info(f"  {i+1}. {f.name}")

                # 给用户5秒时间考虑
                import time
                logger.warning("⏰ 5秒后开始转换，按Ctrl+C取消...")
                try:
                    time.sleep(5)
                except KeyboardInterrupt:
                    logger.info("用户取消转换")
                    return False

        # 创建输出目录结构
        labels_output_dir = save_dir / "labels"
        images_output_dir = save_dir / "images"
        ensure_dir(labels_output_dir)
        ensure_dir(images_output_dir)

        success_count = 0
        total_images = 0
        all_class_names = []

        # 改进的类别处理逻辑
        if len(json_files) > 1:
            logger.info("🔍 多文件模式：先检查类别一致性...")

            # 检查所有文件的类别是否一致
            for i, json_file in enumerate(json_files):
                with open(json_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                categories = coco_data.get('categories', [])
                class_names = [cat['name'] for cat in categories]

                if i == 0:
                    all_class_names = class_names
                    logger.info(f"基准类别 ({json_file.name}): {class_names}")
                else:
                    if class_names != all_class_names:
                        logger.error(f"❌ 类别不一致！")
                        logger.error(f"基准文件类别: {all_class_names}")
                        logger.error(f"当前文件 ({json_file.name}) 类别: {class_names}")
                        logger.error("💡 建议：使用 specific_file 参数单独转换，或确保所有文件类别一致")
                        return False
                    logger.info(f"✅ 类别一致 ({json_file.name})")

            logger.info("✅ 所有文件类别一致，继续转换...")

        for json_file in json_files:
            logger.info(f"📄 处理文件: {json_file.name}")

            # 加载COCO数据
            with open(json_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)

            # 提取类别信息
            categories = coco_data.get('categories', [])
            class_names = [cat['name'] for cat in categories]

            # 只在第一个文件或单文件时保存类别信息
            if json_files.index(json_file) == 0:
                classes_file = save_dir / "classes.txt"
                with open(classes_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(class_names))
                logger.info(f"💾 保存类别信息到: {classes_file}")
                logger.info(f"🏷️  类别列表: {class_names}")
            else:
                logger.info(f"⏭️  跳过类别保存 (已存在)")

            # 创建转换器
            converter = COCOToYOLOConverter()
            converter.set_classes(class_names)

            # 创建图像ID到文件名的映射
            images = coco_data.get('images', [])
            image_id_to_info = {img['id']: img for img in images}
            total_images += len(images)

            # 按图像分组标注
            annotations = coco_data.get('annotations', [])
            image_annotations = {}
            for ann in annotations:
                image_id = ann['image_id']
                if image_id not in image_annotations:
                    image_annotations[image_id] = []
                image_annotations[image_id].append(ann)

            # 转换每个图像的标注
            for image_id, anns in image_annotations.items():
                if image_id not in image_id_to_info:
                    continue

                image_info = image_id_to_info[image_id]
                image_width = image_info['width']
                image_height = image_info['height']
                image_filename = Path(image_info['file_name']).stem

                # 转换标注
                yolo_lines = []
                for ann in anns:
                    if use_segments and 'segmentation' in ann:
                        # 分割标注处理（暂时跳过）
                        logger.warning(f"跳过分割标注: {image_filename}")
                        continue

                    bbox = ann['bbox']  # [x, y, width, height]
                    category_id = ann['category_id']

                    # 转换为YOLO格式
                    yolo_bbox = converter._coco_bbox_to_yolo(bbox, image_width, image_height)
                    yolo_class_id = converter._coco_category_to_yolo_class(category_id, categories)

                    yolo_line = f"{yolo_class_id} {' '.join(map(str, yolo_bbox))}"
                    yolo_lines.append(yolo_line)

                # 保存YOLO标注文件
                output_file = labels_output_dir / f"{image_filename}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))

                # 复制图像文件（如果需要）
                if copy_images:
                    # 智能路径查找原始图像文件
                    image_filename = image_info['file_name']
                    search_paths = [
                        labels_dir.parent / "images" / image_filename,  # raw/images/
                        labels_dir / image_filename,                    # original_annotations/
                        labels_dir.parent / image_filename,             # raw/
                        labels_dir.parent.parent / "images" / image_filename,  # data/images/
                        Path.cwd() / "images" / image_filename,         # 当前目录/images/
                        Path.cwd() / image_filename,                    # 当前目录
                    ]

                    original_image_path = None
                    for path in search_paths:
                        if path.exists():
                            original_image_path = path
                            break

                    if original_image_path:
                        dst_image_path = images_output_dir / image_filename
                        shutil.copy2(original_image_path, dst_image_path)
                        logger.debug(f"复制图片: {original_image_path.name} -> {dst_image_path.name}")
                    else:
                        logger.warning(f"未找到图片文件: {image_filename}")
                        logger.debug(f"已搜索路径: {len(search_paths)}个位置")

                success_count += 1

        logger.info(f"转换完成!")
        logger.info(f"成功转换: {success_count}/{total_images} 个图像")
        logger.info(f"输出目录: {save_dir}")
        logger.info(f"标签目录: {labels_output_dir}")
        logger.info(f"图像目录: {images_output_dir}")

        return True

    except Exception as e:
        logger.error(f"COCO转换失败: {e}")
        return False

def convert_single_coco_file(coco_file_path: Union[str, Path],
                            save_dir: Union[str, Path],
                            use_segments: bool = False,
                            copy_images: bool = True) -> bool:
    """
    转换单个COCO文件到YOLO格式 - 更简单直接的接口

    Args:
        coco_file_path: COCO JSON文件的完整路径
        save_dir: 输出目录
        use_segments: 是否转换分割标注（暂不支持）
        copy_images: 是否复制图像文件

    Returns:
        bool: 转换是否成功
    """
    coco_file_path = Path(coco_file_path)

    if not coco_file_path.exists():
        logger.error(f"COCO文件不存在: {coco_file_path}")
        return False

    if not coco_file_path.suffix.lower() == '.json':
        logger.error(f"文件不是JSON格式: {coco_file_path}")
        return False

    # 使用文件所在目录作为labels_dir，指定具体文件名
    labels_dir = coco_file_path.parent
    specific_file = coco_file_path.name

    logger.info(f"🎯 转换单个文件: {coco_file_path}")

    return convert_coco_ultralytics_style(
        labels_dir=labels_dir,
        save_dir=save_dir,
        use_segments=use_segments,
        copy_images=copy_images,
        specific_file=specific_file
    )

def validate_annotations(label_dir: Union[str, Path],
                        image_dir: Optional[Union[str, Path]] = None,
                        class_count: Optional[int] = None) -> Dict:
    """
    验证YOLO格式标注文件

    Args:
        label_dir: 标签目录
        image_dir: 图像目录（可选）
        class_count: 类别数量（可选）

    Returns:
        Dict: 验证结果统计
    """
    try:
        label_dir = Path(label_dir)

        stats = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'empty_files': 0,
            'errors': [],
            'class_distribution': {},
            'bbox_stats': {
                'total_boxes': 0,
                'invalid_boxes': 0
            }
        }

        label_files = list(label_dir.glob("*.txt"))
        stats['total_files'] = len(label_files)

        for label_file in label_files:
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if not lines or all(not line.strip() for line in lines):
                    stats['empty_files'] += 1
                    continue

                file_valid = True
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        stats['errors'].append(f"{label_file.name}:{line_num} - 格式错误，应为5个值")
                        file_valid = False
                        continue

                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])

                        # 检查类别ID
                        if class_count is not None and (class_id < 0 or class_id >= class_count):
                            stats['errors'].append(f"{label_file.name}:{line_num} - 类别ID超出范围: {class_id}")
                            file_valid = False

                        # 检查坐标范围
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            stats['errors'].append(f"{label_file.name}:{line_num} - 坐标超出范围 [0,1]")
                            stats['bbox_stats']['invalid_boxes'] += 1
                            file_valid = False

                        # 统计类别分布
                        if class_id not in stats['class_distribution']:
                            stats['class_distribution'][class_id] = 0
                        stats['class_distribution'][class_id] += 1
                        stats['bbox_stats']['total_boxes'] += 1

                    except ValueError:
                        stats['errors'].append(f"{label_file.name}:{line_num} - 数值格式错误")
                        file_valid = False

                if file_valid:
                    stats['valid_files'] += 1
                else:
                    stats['invalid_files'] += 1

            except Exception as e:
                stats['errors'].append(f"{label_file.name} - 读取文件失败: {e}")
                stats['invalid_files'] += 1

        # 检查图像-标签对应关系
        if image_dir:
            image_dir = Path(image_dir)
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

            # 找到所有图像文件
            image_files = set()
            for ext in image_extensions:
                image_files.update(f.stem for f in image_dir.glob(f"*{ext}"))
                image_files.update(f.stem for f in image_dir.glob(f"*{ext.upper()}"))

            # 找到所有标签文件
            label_files = set(f.stem for f in label_dir.glob("*.txt"))

            # 检查不匹配的文件
            images_without_labels = image_files - label_files
            labels_without_images = label_files - image_files

            if images_without_labels:
                stats['errors'].append(f"缺少标签的图像: {len(images_without_labels)}个")

            if labels_without_images:
                stats['errors'].append(f"缺少图像的标签: {len(labels_without_images)}个")

        logger.info(f"标注验证完成: {stats['valid_files']}/{stats['total_files']} 文件有效")
        return stats

    except Exception as e:
        logger.error(f"标注验证失败: {e}")
        return {'error': str(e)}

# 直接运行的示例
if __name__ == "__main__":
    """
    直接运行COCO转换的示例
    可以直接运行此文件进行COCO到YOLO的转换

    使用方法:
    python data_converter.py              # 交互式运行
    python data_converter.py --auto       # 自动运行，不需要确认
    python data_converter.py -a           # 自动运行的简写
    """
    import sys
    import argparse
    from pathlib import Path

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="COCO到YOLO转换工具")
    parser.add_argument('--auto', '-a', action='store_true', help='自动运行，不需要确认')
    parser.add_argument('--input', '-i', type=str, help='输入目录路径')
    parser.add_argument('--output', '-o', type=str, help='输出目录路径')
    args = parser.parse_args()

    # 设置默认路径
    project_root = Path(__file__).parent.parent.parent
    default_input = Path(args.input) if args.input else project_root / "yoloserver" / "data" / "raw" / "original_annotations"
    default_output = Path(args.output) if args.output else project_root / "yoloserver" / "data" / "raw" / "yolo_converted"

    print("=" * 60)
    print("🚀 BTD COCO转换工具")
    print("=" * 60)

    # 检查默认输入目录
    if default_input.exists():
        json_files = list(default_input.glob("*.json"))
        if json_files:
            print(f"✅ 找到输入目录: {default_input}")
            print(f"📁 发现 {len(json_files)} 个JSON文件:")
            for f in json_files:
                print(f"   - {f.name}")

            print(f"📤 输出目录: {default_output}")

            # 根据参数决定是否需要确认
            if args.auto:
                print("\n🔄 自动模式，开始转换...")
                should_convert = True
            else:
                # 检查是否在交互式环境中
                try:
                    # 尝试获取用户输入
                    response = input("\n🤔 是否开始转换？(y/N): ").strip().lower()
                    should_convert = (response == 'y')
                except (EOFError, KeyboardInterrupt):
                    # 如果无法获取输入（如在VS Code中运行），自动开始转换
                    print("\n🔄 检测到非交互式环境，自动开始转换...")
                    should_convert = True

            if should_convert:
                print("\n🔄 开始转换...")
                success = convert_coco_ultralytics_style(
                    labels_dir=str(default_input),
                    save_dir=str(default_output),
                    use_segments=False,
                    copy_images=True
                )

                if success:
                    print("✅ 转换完成！")
                    print(f"📁 输出目录: {default_output}")
                else:
                    print("❌ 转换失败！")
                    sys.exit(1)
            else:
                print("取消转换")
        else:
            print(f"❌ 输入目录中没有JSON文件: {default_input}")
            print("💡 请将COCO JSON文件放入该目录")
    else:
        print(f"❌ 输入目录不存在: {default_input}")
        print("💡 请先运行项目初始化: python main.py init")

    print("\n💡 其他使用方法:")
    print("1. python yoloserver/scripts/data_processing.py process --input <目录> --format auto")
    print("2. python main.py data convert --input <输入> --output <输出> --input-format coco")
