"""
路径工具模块
提供路径相关的实用函数
"""

import os
import shutil
from pathlib import Path
from typing import Union, List, Optional


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
        
    Returns:
        Path: 创建的目录路径对象
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def clean_dir(path: Union[str, Path], keep_dir: bool = True) -> bool:
    """
    清理目录内容
    
    Args:
        path: 要清理的目录路径
        keep_dir: 是否保留目录本身
        
    Returns:
        bool: 清理是否成功
    """
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            return True
            
        if path_obj.is_file():
            path_obj.unlink()
            return True
            
        # 清理目录内容
        for item in path_obj.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
                
        # 如果不保留目录，删除目录本身
        if not keep_dir:
            path_obj.rmdir()
            
        return True
    except Exception as e:
        print(f"清理目录失败 {path}: {e}")
        return False


def copy_file(src: Union[str, Path], dst: Union[str, Path], 
              create_dirs: bool = True) -> bool:
    """
    复制文件
    
    Args:
        src: 源文件路径
        dst: 目标文件路径
        create_dirs: 是否自动创建目标目录
        
    Returns:
        bool: 复制是否成功
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            print(f"源文件不存在: {src_path}")
            return False
            
        if create_dirs:
            ensure_dir(dst_path.parent)
            
        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        print(f"复制文件失败 {src} -> {dst}: {e}")
        return False


def move_file(src: Union[str, Path], dst: Union[str, Path], 
              create_dirs: bool = True) -> bool:
    """
    移动文件
    
    Args:
        src: 源文件路径
        dst: 目标文件路径
        create_dirs: 是否自动创建目标目录
        
    Returns:
        bool: 移动是否成功
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            print(f"源文件不存在: {src_path}")
            return False
            
        if create_dirs:
            ensure_dir(dst_path.parent)
            
        shutil.move(str(src_path), str(dst_path))
        return True
    except Exception as e:
        print(f"移动文件失败 {src} -> {dst}: {e}")
        return False


def get_file_size(path: Union[str, Path]) -> int:
    """
    获取文件大小（字节）
    
    Args:
        path: 文件路径
        
    Returns:
        int: 文件大小，如果文件不存在返回-1
    """
    try:
        return Path(path).stat().st_size
    except:
        return -1


def list_files(directory: Union[str, Path], 
               extensions: Optional[List[str]] = None,
               recursive: bool = False) -> List[Path]:
    """
    列出目录中的文件
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表，如 ['.jpg', '.png']
        recursive: 是否递归搜索子目录
        
    Returns:
        List[Path]: 文件路径列表
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []
            
        files = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
            
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                if extensions is None:
                    files.append(file_path)
                else:
                    if any(file_path.suffix.lower() == ext.lower() for ext in extensions):
                        files.append(file_path)
                        
        return sorted(files)
    except Exception as e:
        print(f"列出文件失败 {directory}: {e}")
        return []


def get_relative_path(path: Union[str, Path], 
                     base: Union[str, Path]) -> Path:
    """
    获取相对路径
    
    Args:
        path: 目标路径
        base: 基准路径
        
    Returns:
        Path: 相对路径
    """
    try:
        return Path(path).relative_to(Path(base))
    except ValueError:
        # 如果无法计算相对路径，返回绝对路径
        return Path(path).resolve()


def is_empty_dir(path: Union[str, Path]) -> bool:
    """
    检查目录是否为空
    
    Args:
        path: 目录路径
        
    Returns:
        bool: 目录是否为空
    """
    try:
        path_obj = Path(path)
        if not path_obj.exists() or not path_obj.is_dir():
            return True
        return not any(path_obj.iterdir())
    except:
        return True


def safe_filename(filename: str) -> str:
    """
    生成安全的文件名（移除特殊字符）
    
    Args:
        filename: 原始文件名
        
    Returns:
        str: 安全的文件名
    """
    import re
    # 移除或替换不安全的字符
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除多余的空格和点
    safe_name = re.sub(r'\s+', '_', safe_name.strip())
    safe_name = safe_name.strip('.')
    
    # 确保文件名不为空
    if not safe_name:
        safe_name = "unnamed_file"
        
    return safe_name


def get_project_root() -> Path:
    """
    获取项目根目录

    Returns:
        Path: 项目根目录路径
    """
    # 从当前文件位置向上查找项目根目录
    current_path = Path(__file__).resolve()

    # 向上查找包含BTD目录的根目录
    for parent in current_path.parents:
        if (parent / "BTD").exists():
            return parent

    # 如果找不到，返回BTD目录的父目录
    btd_path = current_path
    while btd_path.name != "BTD" and btd_path.parent != btd_path:
        btd_path = btd_path.parent

    if btd_path.name == "BTD":
        return btd_path.parent

    # 最后的备选方案
    return Path(__file__).resolve().parent.parent.parent


def get_data_paths() -> dict:
    """
    获取所有数据相关路径

    Returns:
        dict: 包含所有数据路径的字典
    """
    project_root = get_project_root()
    # 如果project_root已经是BTD目录，直接使用；否则添加BTD路径
    if project_root.name == "BTD":
        yoloserver_root = project_root / "yoloserver"
    else:
        btd_root = project_root / "BTD"
        yoloserver_root = btd_root / "yoloserver"
    data_root = yoloserver_root / "data"

    return {
        'project_root': project_root,
        'btd_root': btd_root,
        'yoloserver_root': yoloserver_root,
        'data_root': data_root,
        'raw_data': data_root / "raw",
        'raw_images': data_root / "raw" / "images",
        'raw_annotations': data_root / "raw" / "original_annotations",
        'yolo_labels': data_root / "raw" / "yolo_staged_labels",
        'processed_data': data_root / "processed",
        'processed_images': data_root / "processed" / "images",
        'processed_labels': data_root / "processed" / "labels",
        'train_images': data_root / "train" / "images",
        'train_labels': data_root / "train" / "labels",
        'val_images': data_root / "val" / "images",
        'val_labels': data_root / "val" / "labels",
        'test_images': data_root / "test" / "images",
        'test_labels': data_root / "test" / "labels",
    }


def get_config_paths() -> dict:
    """
    获取配置文件路径

    Returns:
        dict: 包含配置路径的字典
    """
    project_root = get_project_root()
    # 如果project_root已经是BTD目录，直接使用；否则添加BTD路径
    if project_root.name == "BTD":
        configs_dir = project_root / "yoloserver" / "configs"
    else:
        configs_dir = project_root / "BTD" / "yoloserver" / "configs"

    return {
        'configs_dir': configs_dir,
        'data_yaml': configs_dir / "data.yaml",
        'dataset_config': configs_dir / "dataset_config.yaml",
        'model_config': configs_dir / "model_config.yaml",
    }


def get_model_paths() -> dict:
    """
    获取模型相关路径

    Returns:
        dict: 包含模型路径的字典
    """
    project_root = get_project_root()
    # 如果project_root已经是BTD目录，直接使用；否则添加BTD路径
    if project_root.name == "BTD":
        models_dir = project_root / "yoloserver" / "models"
    else:
        models_dir = project_root / "BTD" / "yoloserver" / "models"

    return {
        'models_dir': models_dir,
        'checkpoints': models_dir / "checkpoints",
        'pretrained': models_dir / "pretrained",
        'exports': models_dir / "exports",
    }


def create_project_structure() -> bool:
    """
    创建项目目录结构

    Returns:
        bool: 创建是否成功
    """
    try:
        data_paths = get_data_paths()
        config_paths = get_config_paths()
        model_paths = get_model_paths()

        # 创建所有必要的目录
        all_paths = list(data_paths.values()) + list(config_paths.values())[:-3] + list(model_paths.values())

        for path in all_paths:
            if isinstance(path, Path):
                ensure_dir(path)

        return True
    except Exception as e:
        print(f"创建项目结构失败: {e}")
        return False


def validate_project_structure() -> bool:
    """
    验证项目结构是否完整

    Returns:
        bool: 项目结构是否完整
    """
    try:
        data_paths = get_data_paths()

        # 检查关键目录是否存在
        key_dirs = [
            data_paths['data_root'],
            data_paths['raw_data'],
            data_paths['raw_images'],
            data_paths['raw_annotations'],
        ]

        return all(path.exists() for path in key_dirs)
    except:
        return False


def is_project_initialized() -> bool:
    """
    检查项目是否已初始化

    Returns:
        bool: 项目是否已初始化
    """
    return validate_project_structure()


# 导出所有公共函数
__all__ = [
    'ensure_dir',
    'clean_dir',
    'copy_file',
    'move_file',
    'get_file_size',
    'list_files',
    'get_relative_path',
    'is_empty_dir',
    'safe_filename',
    'get_project_root',
    'get_data_paths',
    'get_config_paths',
    'get_model_paths',
    'create_project_structure',
    'validate_project_structure',
    'is_project_initialized'
]
