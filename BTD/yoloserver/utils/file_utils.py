"""
文件操作工具模块
提供文件读写、复制、移动等操作
"""

import json
import yaml
import shutil
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import hashlib
import os

from .logger import get_logger

logger = get_logger(__name__)

def read_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    读取YAML文件
    
    Args:
        file_path: YAML文件路径
        
    Returns:
        Dict: YAML文件内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"读取YAML文件失败 {file_path}: {e}")
        return {}

def write_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    写入YAML文件
    
    Args:
        data: 要写入的数据
        file_path: YAML文件路径
        
    Returns:
        bool: 写入是否成功
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
        return True
    except Exception as e:
        logger.error(f"写入YAML文件失败 {file_path}: {e}")
        return False

def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    读取JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        Dict: JSON文件内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"读取JSON文件失败 {file_path}: {e}")
        return {}

def write_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    写入JSON文件
    
    Args:
        data: 要写入的数据
        file_path: JSON文件路径
        indent: 缩进空格数
        
    Returns:
        bool: 写入是否成功
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return True
    except Exception as e:
        logger.error(f"写入JSON文件失败 {file_path}: {e}")
        return False

def copy_files(src_files: List[Union[str, Path]], 
               dst_dir: Union[str, Path],
               preserve_structure: bool = False) -> bool:
    """
    复制文件到目标目录
    
    Args:
        src_files: 源文件列表
        dst_dir: 目标目录
        preserve_structure: 是否保持目录结构
        
    Returns:
        bool: 复制是否成功
    """
    try:
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        for src_file in src_files:
            src_file = Path(src_file)
            if not src_file.exists():
                logger.warning(f"源文件不存在: {src_file}")
                continue
            
            if preserve_structure:
                # 保持相对路径结构
                rel_path = src_file.relative_to(src_file.anchor)
                dst_file = dst_dir / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                # 直接复制到目标目录
                dst_file = dst_dir / src_file.name
            
            shutil.copy2(src_file, dst_file)
            logger.debug(f"复制文件: {src_file} -> {dst_file}")
        
        logger.info(f"成功复制{len(src_files)}个文件到 {dst_dir}")
        return True
        
    except Exception as e:
        logger.error(f"复制文件失败: {e}")
        return False

def move_files(src_files: List[Union[str, Path]], 
               dst_dir: Union[str, Path],
               preserve_structure: bool = False) -> bool:
    """
    移动文件到目标目录
    
    Args:
        src_files: 源文件列表
        dst_dir: 目标目录
        preserve_structure: 是否保持目录结构
        
    Returns:
        bool: 移动是否成功
    """
    try:
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        for src_file in src_files:
            src_file = Path(src_file)
            if not src_file.exists():
                logger.warning(f"源文件不存在: {src_file}")
                continue
            
            if preserve_structure:
                # 保持相对路径结构
                rel_path = src_file.relative_to(src_file.anchor)
                dst_file = dst_dir / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                # 直接移动到目标目录
                dst_file = dst_dir / src_file.name
            
            shutil.move(str(src_file), str(dst_file))
            logger.debug(f"移动文件: {src_file} -> {dst_file}")
        
        logger.info(f"成功移动{len(src_files)}个文件到 {dst_dir}")
        return True
        
    except Exception as e:
        logger.error(f"移动文件失败: {e}")
        return False

def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> Optional[str]:
    """
    计算文件哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法 (md5, sha1, sha256)
        
    Returns:
        Optional[str]: 文件哈希值，失败返回None
    """
    try:
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
        
    except Exception as e:
        logger.error(f"计算文件哈希失败 {file_path}: {e}")
        return None

def find_duplicate_files(directory: Union[str, Path], 
                        extensions: Optional[List[str]] = None) -> Dict[str, List[Path]]:
    """
    查找目录中的重复文件
    
    Args:
        directory: 搜索目录
        extensions: 文件扩展名过滤器
        
    Returns:
        Dict[str, List[Path]]: 哈希值到文件路径列表的映射
    """
    try:
        directory = Path(directory)
        file_hashes = {}
        
        # 获取所有文件
        if extensions:
            files = []
            for ext in extensions:
                files.extend(directory.rglob(f"*.{ext}"))
        else:
            files = [f for f in directory.rglob("*") if f.is_file()]
        
        # 计算每个文件的哈希值
        for file_path in files:
            file_hash = calculate_file_hash(file_path)
            if file_hash:
                if file_hash not in file_hashes:
                    file_hashes[file_hash] = []
                file_hashes[file_hash].append(file_path)
        
        # 只返回有重复的文件
        duplicates = {h: files for h, files in file_hashes.items() if len(files) > 1}
        
        if duplicates:
            logger.info(f"找到{len(duplicates)}组重复文件")
        else:
            logger.info("未找到重复文件")
        
        return duplicates
        
    except Exception as e:
        logger.error(f"查找重复文件失败: {e}")
        return {}

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    获取文件详细信息
    
    Args:
        file_path: 文件路径
        
    Returns:
        Dict: 文件信息
    """
    try:
        file_path = Path(file_path)
        stat = file_path.stat()
        
        return {
            'name': file_path.name,
            'stem': file_path.stem,
            'suffix': file_path.suffix,
            'size': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': stat.st_ctime,
            'modified': stat.st_mtime,
            'accessed': stat.st_atime,
            'is_file': file_path.is_file(),
            'is_dir': file_path.is_dir(),
            'absolute_path': str(file_path.absolute()),
            'parent': str(file_path.parent)
        }
        
    except Exception as e:
        logger.error(f"获取文件信息失败 {file_path}: {e}")
        return {}

def clean_empty_directories(directory: Union[str, Path], 
                           remove_root: bool = False) -> int:
    """
    清理空目录
    
    Args:
        directory: 要清理的目录
        remove_root: 是否删除根目录（如果为空）
        
    Returns:
        int: 删除的目录数量
    """
    try:
        directory = Path(directory)
        removed_count = 0
        
        # 从最深层开始删除空目录
        for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
            dirpath = Path(dirpath)
            
            # 跳过根目录（除非明确指定）
            if dirpath == directory and not remove_root:
                continue
            
            # 检查目录是否为空
            try:
                if not any(dirpath.iterdir()):
                    dirpath.rmdir()
                    removed_count += 1
                    logger.debug(f"删除空目录: {dirpath}")
            except OSError:
                # 目录不为空或无法删除
                pass
        
        if removed_count > 0:
            logger.info(f"清理了{removed_count}个空目录")
        
        return removed_count
        
    except Exception as e:
        logger.error(f"清理空目录失败: {e}")
        return 0

def backup_file(file_path: Union[str, Path], 
                backup_dir: Optional[Union[str, Path]] = None,
                add_timestamp: bool = True) -> Optional[Path]:
    """
    备份文件
    
    Args:
        file_path: 要备份的文件路径
        backup_dir: 备份目录，默认为原文件目录下的backup子目录
        add_timestamp: 是否在备份文件名中添加时间戳
        
    Returns:
        Optional[Path]: 备份文件路径，失败返回None
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"要备份的文件不存在: {file_path}")
            return None
        
        # 确定备份目录
        if backup_dir is None:
            backup_dir = file_path.parent / 'backup'
        else:
            backup_dir = Path(backup_dir)
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成备份文件名
        if add_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        else:
            backup_name = f"{file_path.stem}_backup{file_path.suffix}"
        
        backup_path = backup_dir / backup_name
        
        # 复制文件
        shutil.copy2(file_path, backup_path)
        logger.info(f"文件备份成功: {file_path} -> {backup_path}")
        
        return backup_path
        
    except Exception as e:
        logger.error(f"文件备份失败: {e}")
        return None

if __name__ == "__main__":
    # 测试文件工具函数
    test_data = {"test": "data", "number": 123}
    
    # 测试YAML读写
    write_yaml(test_data, "test.yaml")
    loaded_data = read_yaml("test.yaml")
    print("YAML测试:", loaded_data)
    
    # 测试JSON读写
    write_json(test_data, "test.json")
    loaded_data = read_json("test.json")
    print("JSON测试:", loaded_data)
    
    # 清理测试文件
    Path("test.yaml").unlink(missing_ok=True)
    Path("test.json").unlink(missing_ok=True)
