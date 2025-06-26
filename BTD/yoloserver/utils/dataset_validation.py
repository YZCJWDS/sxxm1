# -*- coding:utf-8 -*-
# @FileName  :dataset_validation.py
# @Time      :2025/6/25 16:10:00
# @Author    :BTD Team
# @Project   :BrainTumorDetection
# @Function  :验证YOLO数据集配置以及相关文件

import yaml
from pathlib import Path
import logging
import random
from typing import Tuple, List, Dict

# 配置验证模式和参数
SAMPLE_SIZE = 0.1
MIN_SAMPLES = 10


def verify_dataset_config(yaml_path: Path, current_logger: logging.Logger, mode: str, task_type: str) -> Tuple[bool, List[Dict]]:
    """
    验证YOLO数据集配置，检查data.yaml和对应的图像、标签文件。
    根据 task_type 参数验证标签文件格式。

    Args:
        yaml_path (Path): data.yaml文件的路径
        current_logger (logging.Logger): 用于记录日志的logger实例
        mode (str): 验证模式，"FULL" (完整验证) 或 "SAMPLE" (抽样验证)
        task_type (str): 任务类型，"detection" 或 "segmentation"

    Returns:
        Tuple[bool, List[Dict]]:
            - bool: 表示验证是否通过（True为通过，False为未通过）
            - List[Dict]: 包含所有不合法样本信息的列表，每个字典包含 'image_path', 'label_path', 'error_message'
    """
    import time
    start_time = time.time()

    current_logger.info(f"验证data.yaml文件配置,配置文件路径为：{yaml_path}")
    current_logger.info(f"当前验证任务类型为: {task_type.upper()}")
    current_logger.info(f"验证模式: {mode}")

    invalid_samples = []  # 用于收集不合法样本信息

    if not yaml_path.exists():
        current_logger.error(f"data.yaml文件不存在: {yaml_path}，请检查配置文件路径是否正确")
        return False, invalid_samples

    # 读取YAML文件
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        current_logger.error(f"读取data.yaml文件失败: {e}")
        return False, invalid_samples

    classes_names = config.get("names", [])
    nc = config.get("nc", 0)

    if len(classes_names) != nc:
        current_logger.error(f"数据集类别数量与配置文件不一致，{len(classes_names)} != {nc},请检查配置文件")
        return False, invalid_samples
    current_logger.info(f"数据集类别数量与配置文件一致，类别数量为：{nc}，类别为：{classes_names}")

    # 获取数据集根路径（相对于yaml文件的位置）
    yaml_dir = yaml_path.parent
    dataset_path = config.get("path", ".")
    if Path(dataset_path).is_absolute():
        dataset_root = Path(dataset_path).resolve()
    else:
        dataset_root = (yaml_dir / dataset_path).resolve()
    current_logger.info(f"数据集根路径: {dataset_root}")

    # 验证数据集
    splits = ["train", "val"]
    if 'test' in config and config['test'] is not None:
        splits.append('test')
    else:
        current_logger.info("data.yaml中未定义 'test'路径或其路径值为None，跳过test验证")

    overall_validation_status = True  # 用于跟踪总体验证状态

    for split in splits:
        # 正确处理相对于dataset_root的路径
        split_relative_path = config[split]
        split_path = (dataset_root / split_relative_path).resolve()

        current_logger.info(f"验证 {split} 路径为: {split_path}")
        if not split_path.exists():
            current_logger.error(f"{split} 路径不存在: {split_path}")
            overall_validation_status = False
            # 对于路径不存在的情况，不记录到 invalid_samples，因为这不是具体文件的问题
            continue

        # 获取图像文件
        img_paths = (
                list(split_path.glob("*.[jJ][pP][gG]")) +
                list(split_path.glob("*.[pP][nN][gG]")) +
                list(split_path.glob("*.[jJ][pP][eE][gG]")) +
                list(split_path.glob("*.[tT][iI][fF]")) +
                list(split_path.glob("*.[tT][iI][fF][fF]")) +
                list(split_path.glob("*.[bB][mM][pP]")) +
                list(split_path.glob("*.[wW][eE][bB][pP]"))  # 添加webp支持
        )
        if not img_paths:
            current_logger.error(f"图像目录{split_path} 路径下没有图像文件")
            overall_validation_status = False
            continue
        current_logger.info(f"图像目录{split_path} 存在{len(img_paths)}张图像")

        # 动态抽样
        sample_size = max(MIN_SAMPLES, int(len(img_paths) * SAMPLE_SIZE))
        if mode == "FULL":
            current_logger.info(f"{split} 验证模式为FULL，将验证所有图像")
            sample_paths = img_paths
        else:
            current_logger.info(f"{split} 验证模式为SAMPLE，将随机抽取{sample_size}张图像进行验证")
            sample_paths = random.sample(img_paths, min(sample_size, len(img_paths)))

        # 验证每张图像及其标签
        for img_path in sample_paths:
            # 构建标签文件路径 - 将images替换为labels
            label_dir = str(img_path.parent).replace('images', 'labels')
            label_path = Path(label_dir) / (img_path.stem + '.txt')
            
            if not label_path.exists():
                error_msg = f"标签文件不存在: {label_path}"
                current_logger.warning(error_msg)
                invalid_samples.append({
                    'image_path': str(img_path),
                    'label_path': str(label_path),
                    'error_message': error_msg
                })
                overall_validation_status = False
                continue

            # 验证标签文件内容
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                error_msg = f"无法读取标签文件: {e}"
                current_logger.error(error_msg)
                invalid_samples.append({
                    'image_path': str(img_path),
                    'label_path': str(label_path),
                    'error_message': error_msg
                })
                overall_validation_status = False
                continue

            # 跳过空文件
            if not lines:
                continue

            # 清理行内容
            lines = [line.strip() for line in lines if line.strip()]

            # 标记当前标签文件是否有错误，避免重复添加 invalid_samples
            current_label_has_error = False
            for line_idx, line in enumerate(lines):
                parts = line.split(" ")

                is_format_correct = True
                error_detail = ""
                if task_type == "detection":
                    # 检测任务：class_id x_center y_center width height (5个值)
                    if len(parts) != 5:
                        error_detail = "不符合检测 YOLO 格式 (应为5个浮点数)"
                        is_format_correct = False
                elif task_type == "segmentation":
                    # 分割任务：class_id x1 y1 x2 y2 ... xN yN (至少7个值，即 1 + 2*N, N >= 3)
                    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                        error_detail = "不符合分割 YOLO 格式 (应为至少7个值，且类别ID后坐标对数量为偶数)"
                        is_format_correct = False
                else:
                    error_detail = f"未知的任务类型 '{task_type}'"
                    is_format_correct = False

                if not is_format_correct:
                    error_msg = f"第{line_idx + 1}行格式错误: {error_detail}"
                    current_logger.warning(f"{label_path} {error_msg}")
                    if not current_label_has_error:
                        invalid_samples.append({
                            'image_path': str(img_path),
                            'label_path': str(label_path),
                            'error_message': error_msg
                        })
                        current_label_has_error = True
                        overall_validation_status = False
                    continue

                # 验证数值
                try:
                    values = [float(part) for part in parts]
                except ValueError:
                    error_msg = f"第{line_idx + 1}行包含非数字值"
                    current_logger.warning(f"{label_path} {error_msg}")
                    if not current_label_has_error:
                        invalid_samples.append({
                            'image_path': str(img_path),
                            'label_path': str(label_path),
                            'error_message': error_msg
                        })
                        current_label_has_error = True
                        overall_validation_status = False
                    continue

                # 验证类别ID
                class_id = int(values[0])
                if class_id < 0 or class_id >= nc:
                    error_msg = f"第{line_idx + 1}行类别ID {class_id} 超出范围 [0, {nc-1}]"
                    current_logger.warning(f"{label_path} {error_msg}")
                    if not current_label_has_error:
                        invalid_samples.append({
                            'image_path': str(img_path),
                            'label_path': str(label_path),
                            'error_message': error_msg
                        })
                        current_label_has_error = True
                        overall_validation_status = False
                    continue

                # 验证坐标值范围 [0, 1]
                coords = values[1:]
                for coord in coords:
                    if coord < 0 or coord > 1:
                        error_msg = f"第{line_idx + 1}行坐标值 {coord} 超出范围 [0, 1]"
                        current_logger.warning(f"{label_path} {error_msg}")
                        if not current_label_has_error:
                            invalid_samples.append({
                                'image_path': str(img_path),
                                'label_path': str(label_path),
                                'error_message': error_msg
                            })
                            current_label_has_error = True
                            overall_validation_status = False
                        break

    # 输出验证结果总结
    if overall_validation_status:
        current_logger.info("数据集配置验证通过")
    else:
        current_logger.error(f"数据集配置验证失败，发现 {len(invalid_samples)} 个不合法样本")

    # 记录性能信息
    end_time = time.time()
    elapsed_time = end_time - start_time
    current_logger.info(f"性能测试：'数据集配置验证' 执行耗时: {elapsed_time:.3f}秒")

    return overall_validation_status, invalid_samples


def verify_split_uniqueness(yaml_path: Path, current_logger: logging.Logger) -> bool:
    """
    验证数据集划分（train, val, test）之间是否存在重复图像。

    Args:
        yaml_path (Path): data.yaml文件的路径
        current_logger (logging.Logger): 用于记录日志的logger实例

    Returns:
        bool: 表示分割唯一性验证是否通过（True为无重复，False为存在重复）
    """
    import time
    start_time = time.time()

    current_logger.info("开始验证数据集划分的唯一性（train, val, test 之间无重复图像）。")
    if not yaml_path.exists():
        current_logger.error(f"data.yaml 文件不存在: {yaml_path}，无法进行分割唯一性验证。")
        return False

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        current_logger.error(f"读取data.yaml文件失败: {e}，无法进行分割唯一性验证。")
        return False

    # 获取数据集根路径（相对于yaml文件的位置）
    yaml_dir = yaml_path.parent
    dataset_path = config.get("path", ".")
    if Path(dataset_path).is_absolute():
        dataset_root = Path(dataset_path).resolve()
    else:
        dataset_root = (yaml_dir / dataset_path).resolve()

    # 获取所有分割的路径
    splits = ["train", "val"]
    if 'test' in config and config['test'] is not None:
        splits.append('test')

    # 收集每个分割的图像文件名
    split_images = {}
    for split in splits:
        # 正确处理相对于dataset_root的路径
        split_relative_path = config[split]
        split_path = (dataset_root / split_relative_path).resolve()

        if not split_path.exists():
            current_logger.warning(f"{split} 路径不存在: {split_path}，跳过该分割的唯一性验证")
            continue

        # 获取图像文件
        img_paths = (
                list(split_path.glob("*.[jJ][pP][gG]")) +
                list(split_path.glob("*.[pP][nN][gG]")) +
                list(split_path.glob("*.[jJ][pP][eE][gG]")) +
                list(split_path.glob("*.[tT][iI][fF]")) +
                list(split_path.glob("*.[tT][iI][fF][fF]")) +
                list(split_path.glob("*.[bB][mM][pP]")) +
                list(split_path.glob("*.[wW][eE][bB][pP]"))
        )

        # 只保存文件名（不包含路径）
        split_images[split] = {img_path.name for img_path in img_paths}
        current_logger.info(f"{split} 分割包含 {len(split_images[split])} 张图像")

    # 检查分割之间的重复
    overall_uniqueness_status = True
    splits_list = list(split_images.keys())

    for i in range(len(splits_list)):
        for j in range(i + 1, len(splits_list)):
            split1, split2 = splits_list[i], splits_list[j]

            # 找到重复的图像文件名
            duplicates = split_images[split1] & split_images[split2]

            if duplicates:
                current_logger.error(f"发现 {split1} 和 {split2} 之间存在 {len(duplicates)} 个重复图像:")
                for duplicate in sorted(list(duplicates)[:10]):  # 只显示前10个
                    current_logger.error(f"  重复图像: {duplicate}")
                if len(duplicates) > 10:
                    current_logger.error(f"  ... 还有 {len(duplicates) - 10} 个重复图像")
                overall_uniqueness_status = False

    if overall_uniqueness_status:
        current_logger.info("数据集分割唯一性验证通过，各分割之间无重复图像。")
    else:
        current_logger.error("数据集分割唯一性验证未通过，存在重复图像。请检查日志。")

    # 记录性能信息
    end_time = time.time()
    elapsed_time = end_time - start_time
    current_logger.info(f"性能测试：'数据集分割验证' 执行耗时: {elapsed_time:.3f}秒")

    return overall_uniqueness_status


def delete_invalid_files(invalid_data_list: list, current_logger: logging.Logger):
    """
    删除列表中指定的不合法图像和标签文件。
    注意：删除操作不可逆，应谨慎执行。

    Args:
        invalid_data_list (list): verify_dataset_config返回的不合法样本列表
        current_logger (logging.Logger): 用于记录日志的logger实例
    """
    if not invalid_data_list:
        current_logger.info("没有需要删除的不合法文件")
        return

    current_logger.warning(f"准备删除 {len(invalid_data_list)} 个不合法文件对")

    deleted_count = 0
    error_count = 0

    for invalid_data in invalid_data_list:
        image_path = Path(invalid_data['image_path'])
        label_path = Path(invalid_data['label_path'])
        error_message = invalid_data['error_message']

        current_logger.info(f"删除不合法文件对: {image_path.name} (原因: {error_message})")

        # 删除图像文件
        try:
            if image_path.exists():
                image_path.unlink()
                current_logger.debug(f"已删除图像文件: {image_path}")
            else:
                current_logger.warning(f"图像文件不存在，跳过删除: {image_path}")
        except Exception as e:
            current_logger.error(f"删除图像文件失败: {image_path}, 错误: {e}")
            error_count += 1
            continue

        # 删除标签文件
        try:
            if label_path.exists():
                label_path.unlink()
                current_logger.debug(f"已删除标签文件: {label_path}")
            else:
                current_logger.warning(f"标签文件不存在，跳过删除: {label_path}")
        except Exception as e:
            current_logger.error(f"删除标签文件失败: {label_path}, 错误: {e}")
            error_count += 1
            continue

        deleted_count += 1

    current_logger.info(f"文件删除完成: 成功删除 {deleted_count} 个文件对，失败 {error_count} 个")

    if error_count > 0:
        current_logger.warning(f"有 {error_count} 个文件删除失败，请检查文件权限或手动删除")
