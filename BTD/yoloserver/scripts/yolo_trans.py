# -*- coding: utf-8 -*-
# -*- coding:utf-8 -*-
# @FileName  :yolo_trans.py
# @Time      :2025/6/24 16:00:00
# @Author    :BTD Team
# @Project   :BTD
# @Function  :YOLO数据转换顶层脚本，提供完整的数据转换、分割、data.yaml生成的一站式解决方案

import sys
import shutil
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from yoloserver.utils.logger import setup_logger, get_logger
from yoloserver.utils.performance_utils import time_it, PerformanceProfiler
from yoloserver.utils.path_utils import get_data_paths, get_config_paths, ensure_dir
from yoloserver.utils.file_utils import write_yaml
from yoloserver.utils.data_converter import (
    convert_coco_ultralytics_style,
    convert_pascal_to_yolo,
    split_dataset,
    validate_annotations
)

logger = get_logger(__name__)


class YOLODataTransformer:
    """YOLO数据转换器"""
    
    def __init__(self, clean_previous: bool = True):
        """
        初始化YOLO数据转换器
        
        Args:
            clean_previous: 是否清理之前的数据
        """
        self.data_paths = get_data_paths()
        self.config_paths = get_config_paths()
        self.clean_previous = clean_previous
        self.profiler = PerformanceProfiler(logger)
        self.class_names = []
        
        # 设置日志
        setup_logger("yolo_trans", console_output=True, file_output=True)
        
    def clean_previous_data(self) -> None:
        """清理之前的划分目录和data.yaml文件"""
        logger.info("开始清理之前的数据...")

        # 清理processed和train/val/test目录
        dirs_to_clean = [
            self.data_paths['processed_data'],
            self.data_paths['train_images'],
            self.data_paths['train_labels'],
            self.data_paths['val_images'],
            self.data_paths['val_labels'],
            self.data_paths['test_images'],
            self.data_paths['test_labels']
        ]

        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.info(f"已清理目录: {dir_path}")

        # 清理混乱的转换输出目录
        messy_dirs = [
            self.data_paths['raw_data'] / 'coco_converted',
            self.data_paths['raw_data'] / 'yolo_converted_cli'
        ]

        for dir_path in messy_dirs:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.info(f"已清理混乱的转换目录: {dir_path}")

        # 清理data.yaml文件
        data_yaml_path = self.config_paths['configs_dir'] / 'data.yaml'
        if data_yaml_path.exists():
            data_yaml_path.unlink()
            logger.info(f"已清理配置文件: {data_yaml_path}")

        logger.info("数据清理完成")
    
    @time_it(iterations=1, name="数据转换", logger_instance=logger)
    def convert_annotations(self, input_dir: str, annotation_format: str = "auto", 
                          class_names: Optional[List[str]] = None) -> bool:
        """
        转换标注格式
        
        Args:
            input_dir: 输入目录
            annotation_format: 标注格式 (auto, coco, pascal, yolo)
            class_names: 类别名称列表（Pascal VOC需要）
            
        Returns:
            bool: 转换是否成功
        """
        logger.info(f"开始转换标注格式: {annotation_format}")
        self.profiler.start_timer("标注转换")
        
        try:
            input_dir = Path(input_dir)
            output_dir = self.data_paths['yolo_labels']
            ensure_dir(output_dir)
            
            # 自动检测标注格式
            if annotation_format == "auto":
                if list(input_dir.glob("*.json")):
                    annotation_format = "coco"
                    logger.info("自动检测到COCO格式标注")
                elif list(input_dir.glob("*.xml")):
                    annotation_format = "pascal"
                    logger.info("自动检测到Pascal VOC格式标注")
                elif list(input_dir.glob("*.txt")):
                    annotation_format = "yolo"
                    logger.info("自动检测到YOLO格式标注")
                else:
                    logger.error("无法自动检测标注格式")
                    return False
            
            # 执行转换
            if annotation_format == "coco":
                success = convert_coco_ultralytics_style(
                    labels_dir=str(input_dir),
                    save_dir=str(output_dir.parent / "coco_converted"),
                    use_segments=False,
                    copy_images=False  # 不复制图像，避免重复
                )
                
                if success:
                    # 提取类别信息
                    classes_file = output_dir.parent / "coco_converted" / "classes.txt"
                    if classes_file.exists():
                        with open(classes_file, 'r', encoding='utf-8') as f:
                            self.class_names = [line.strip() for line in f.readlines() if line.strip()]
                    
                    # 移动标签文件到指定位置
                    source_labels = output_dir.parent / "coco_converted" / "labels"
                    if source_labels.exists():
                        for txt_file in source_labels.glob("*.txt"):
                            shutil.move(str(txt_file), str(output_dir / txt_file.name))
                        logger.info(f"已移动标签文件到: {output_dir}")
                
            elif annotation_format == "pascal":
                if not class_names:
                    logger.error("Pascal VOC转换需要提供类别名称列表")
                    return False
                
                self.class_names = class_names
                success = convert_pascal_to_yolo(
                    xml_dir=input_dir,
                    output_dir=output_dir,
                    class_names=class_names,
                    image_dir=self.data_paths['raw_images']
                )
                
            elif annotation_format == "yolo":
                # 已经是YOLO格式，直接复制
                txt_files = list(input_dir.glob("*.txt"))
                for txt_file in txt_files:
                    shutil.copy2(txt_file, output_dir / txt_file.name)
                logger.info(f"复制了{len(txt_files)}个YOLO标注文件")
                
                # 需要手动指定类别名称
                if class_names:
                    self.class_names = class_names
                else:
                    logger.warning("YOLO格式转换建议提供类别名称列表")
                    self.class_names = [f"class_{i}" for i in range(10)]  # 默认10个类别
                
                success = True
            else:
                logger.error(f"不支持的标注格式: {annotation_format}")
                return False
            
            self.profiler.end_timer("标注转换")
            
            if success:
                logger.info(f"标注转换完成，类别数量: {len(self.class_names)}")
                logger.info(f"类别列表: {self.class_names}")
            
            return success
            
        except Exception as e:
            logger.error(f"标注转换失败: {e}")
            return False
    
    @time_it(iterations=1, name="整理有效数据", logger_instance=logger)
    def organize_valid_data(self) -> bool:
        """整理有效数据到processed目录"""
        logger.info("开始整理有效数据...")
        self.profiler.start_timer("整理有效数据")

        try:
            from yoloserver.utils.data_converter import organize_valid_pairs
            success = organize_valid_pairs(
                image_dir=self.data_paths['raw_images'],
                label_dir=self.data_paths['yolo_labels'],
                output_dir=self.data_paths['processed_data']
            )

            self.profiler.end_timer("整理有效数据")

            if success:
                logger.info("有效数据整理完成")
            else:
                logger.error("有效数据整理失败")

            return success

        except Exception as e:
            logger.error(f"有效数据整理失败: {e}")
            return False

    @time_it(iterations=1, name="数据集分割", logger_instance=logger)
    def split_dataset_data(self, train_ratio: float = 0.7, val_ratio: float = 0.2,
                          test_ratio: float = 0.1, seed: int = 42) -> bool:
        """
        分割数据集

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子

        Returns:
            bool: 分割是否成功
        """
        logger.info("开始分割数据集...")
        self.profiler.start_timer("数据集分割")

        try:
            success = split_dataset(
                image_dir=self.data_paths['processed_images'],
                label_dir=self.data_paths['processed_labels'],
                output_dir=self.data_paths['data_root'],
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed
            )
            
            self.profiler.end_timer("数据集分割")
            
            if success:
                logger.info("数据集分割完成")
                
                # 统计分割结果
                train_count = len(list(self.data_paths['train_images'].glob("*")))
                val_count = len(list(self.data_paths['val_images'].glob("*")))
                test_count = len(list(self.data_paths['test_images'].glob("*")))
                
                logger.info(f"训练集: {train_count} 个样本")
                logger.info(f"验证集: {val_count} 个样本")
                logger.info(f"测试集: {test_count} 个样本")
            
            return success
            
        except Exception as e:
            logger.error(f"数据集分割失败: {e}")
            return False
    
    @time_it(iterations=1, name="生成data.yaml", logger_instance=logger)
    def generate_data_yaml(self) -> bool:
        """
        生成data.yaml配置文件
        
        Returns:
            bool: 生成是否成功
        """
        logger.info("开始生成data.yaml配置文件...")
        self.profiler.start_timer("生成data.yaml")
        
        try:
            # 构建data.yaml内容 - 使用绝对路径确保训练脚本能正确找到数据
            # 获取数据目录的绝对路径
            data_dir_abs = self.data_paths['data_root'].resolve()

            data_config = {
                'path': str(data_dir_abs),  # 使用绝对路径
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': len(self.class_names),
                'names': self.class_names
            }
            
            # 保存data.yaml文件
            data_yaml_path = self.config_paths['configs_dir'] / 'data.yaml'
            ensure_dir(data_yaml_path.parent)
            
            success = write_yaml(data_config, data_yaml_path)
            
            self.profiler.end_timer("生成data.yaml")
            
            if success:
                logger.info(f"data.yaml配置文件已生成: {data_yaml_path}")
                logger.info("配置内容:")
                for key, value in data_config.items():
                    logger.info(f"  {key}: {value}")
            
            return success
            
        except Exception as e:
            logger.error(f"生成data.yaml失败: {e}")
            return False
    
    def validate_dataset_data(self) -> bool:
        """
        验证数据集
        
        Returns:
            bool: 验证是否通过
        """
        logger.info("开始验证数据集...")
        
        try:
            # 验证训练集
            train_stats = validate_annotations(
                label_dir=self.data_paths['train_labels'],
                image_dir=self.data_paths['train_images'],
                class_count=len(self.class_names)
            )
            
            # 验证验证集
            val_stats = validate_annotations(
                label_dir=self.data_paths['val_labels'],
                image_dir=self.data_paths['val_images'],
                class_count=len(self.class_names)
            )
            
            # 验证测试集
            test_stats = validate_annotations(
                label_dir=self.data_paths['test_labels'],
                image_dir=self.data_paths['test_images'],
                class_count=len(self.class_names)
            )
            
            # 汇总验证结果
            total_valid = train_stats['valid_files'] + val_stats['valid_files'] + test_stats['valid_files']
            total_files = train_stats['total_files'] + val_stats['total_files'] + test_stats['total_files']
            
            success_rate = total_valid / total_files if total_files > 0 else 0
            
            logger.info(f"数据集验证完成:")
            logger.info(f"  总文件数: {total_files}")
            logger.info(f"  有效文件数: {total_valid}")
            logger.info(f"  有效率: {success_rate:.2%}")
            
            return success_rate >= 0.95  # 95%以上有效率认为通过
            
        except Exception as e:
            logger.error(f"数据集验证失败: {e}")
            return False

    def run_full_pipeline(self, input_dir: str, annotation_format: str = "auto",
                         class_names: Optional[List[str]] = None,
                         train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1,
                         seed: int = 42, validate: bool = True) -> bool:
        """
        运行完整的数据转换流水线

        Args:
            input_dir: 输入目录
            annotation_format: 标注格式
            class_names: 类别名称列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
            validate: 是否验证数据集

        Returns:
            bool: 流水线是否成功
        """
        print("🚀 开始YOLO数据转换流水线...")
        logger.info("开始YOLO数据转换完整流水线")

        self.profiler.start_timer("完整流水线")

        try:
            # 1. 清理之前的数据
            if self.clean_previous:
                self.clean_previous_data()

            # 2. 转换标注格式
            print("📝 步骤1: 转换标注格式...")
            if not self.convert_annotations(input_dir, annotation_format, class_names):
                print("❌ 标注转换失败")
                return False
            print("✅ 标注转换完成")

            # 3. 整理有效数据
            print("📦 步骤2: 整理有效数据...")
            if not self.organize_valid_data():
                print("❌ 有效数据整理失败")
                return False
            print("✅ 有效数据整理完成")

            # 4. 分割数据集
            print("📂 步骤3: 分割数据集...")
            if not self.split_dataset_data(train_ratio, val_ratio, test_ratio, seed):
                print("❌ 数据集分割失败")
                return False
            print("✅ 数据集分割完成")

            # 5. 生成data.yaml
            print("⚙️ 步骤4: 生成data.yaml配置文件...")
            if not self.generate_data_yaml():
                print("❌ data.yaml生成失败")
                return False
            print("✅ data.yaml生成完成")

            # 6. 验证数据集（可选）
            if validate:
                print("🔍 步骤5: 验证数据集...")
                if not self.validate_dataset_data():
                    print("⚠️ 数据集验证未完全通过，但流水线继续")
                else:
                    print("✅ 数据集验证通过")

            self.profiler.end_timer("完整流水线")

            print("\n" + "="*50)
            print("🎉 YOLO数据转换流水线完成！")
            print("="*50)

            # 输出下一步建议
            data_yaml_path = self.config_paths['configs_dir'] / 'data.yaml'
            print("\n📋 下一步操作建议:")
            print(f"1. 检查生成的配置文件: {data_yaml_path}")
            print("2. 开始训练模型:")
            print("   python BTD/main.py train")
            print("3. 或查看转换结果:")
            print(f"   - 训练集: {self.data_paths['train_images']}")
            print(f"   - 验证集: {self.data_paths['val_images']}")
            print(f"   - 测试集: {self.data_paths['test_images']}")

            return True

        except Exception as e:
            logger.error(f"流水线执行失败: {e}")
            return False

    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 0.001:
            return f"{seconds * 1000:.1f} 毫秒"
        elif seconds < 60:
            return f"{seconds:.2f} 秒"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes} 分 {remaining_seconds:.1f} 秒"


def main():
    """主函数 - 简化版，直接运行即可"""
    print("=" * 60)
    print("🚀 YOLO数据转换一站式工具")
    print("=" * 60)

    # 设置默认路径
    project_root = Path(__file__).parent.parent.parent
    default_input = project_root / "yoloserver" / "data" / "raw" / "original_annotations"

    print(f"📁 输入目录: {default_input}")

    # 检查输入目录
    if not default_input.exists():
        print(f"❌ 输入目录不存在: {default_input}")
        print("💡 请先运行: python main.py init")
        return 1

    # 创建转换器并运行
    transformer = YOLODataTransformer(clean_previous=True)

    success = transformer.run_full_pipeline(
        input_dir=str(default_input),
        annotation_format="auto",
        class_names=None,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
        validate=True
    )

    if success:
        print("🎉 数据转换流水线执行成功！")
        return 0
    else:
        print("❌ 数据转换流水线执行失败！")
        return 1


def run_with_params(input_dir: str, annotation_format: str = "auto",
                   class_names: Optional[List[str]] = None,
                   train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1,
                   seed: int = 42, validate: bool = True, clean_previous: bool = True) -> bool:
    """
    供其他代码调用的函数接口

    Args:
        input_dir: 输入标注文件目录
        annotation_format: 标注格式 (auto, coco, pascal, yolo)
        class_names: 类别名称列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        validate: 是否验证数据集
        clean_previous: 是否清理之前的数据

    Returns:
        bool: 转换是否成功
    """
    transformer = YOLODataTransformer(clean_previous=clean_previous)

    return transformer.run_full_pipeline(
        input_dir=input_dir,
        annotation_format=annotation_format,
        class_names=class_names,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        validate=validate
    )


if __name__ == "__main__":
    main()
