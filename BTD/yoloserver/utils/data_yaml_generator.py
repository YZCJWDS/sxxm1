"""
data.yaml配置文件生成器
提供YOLO训练配置文件的生成功能
"""

import yaml
from pathlib import Path
from typing import List, Dict, Union, Optional

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .logger import get_logger
    from .file_utils import write_yaml
except ImportError:
    # 直接运行时使用绝对导入
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from yoloserver.utils.logger import get_logger
    from yoloserver.utils.file_utils import write_yaml

logger = get_logger(__name__)


class DataYamlGenerator:
    """data.yaml配置文件生成器"""
    
    def __init__(self, data_root: Union[str, Path] = None):
        """
        初始化生成器
        
        Args:
            data_root: 数据根目录路径
        """
        self.data_root = Path(data_root) if data_root else None
        self.class_names = []
        
    def set_data_root(self, data_root: Union[str, Path]) -> None:
        """
        设置数据根目录
        
        Args:
            data_root: 数据根目录路径
        """
        self.data_root = Path(data_root)
        
    def set_class_names(self, class_names: List[str]) -> None:
        """
        设置类别名称
        
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names
        
    def generate_config(self, 
                       train_path: str = "train/images",
                       val_path: str = "val/images", 
                       test_path: str = "test/images",
                       use_relative_paths: bool = True) -> Dict:
        """
        生成data.yaml配置内容
        
        Args:
            train_path: 训练集路径
            val_path: 验证集路径
            test_path: 测试集路径
            use_relative_paths: 是否使用相对路径
            
        Returns:
            Dict: 配置字典
        """
        if not self.class_names:
            logger.warning("类别名称未设置，使用默认类别")
            self.class_names = ['class_0']
            
        config = {
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        if use_relative_paths:
            # 使用相对路径
            if self.data_root:
                config['path'] = str(self.data_root)
            else:
                config['path'] = './data'
            config['train'] = train_path
            config['val'] = val_path
            config['test'] = test_path
        else:
            # 使用绝对路径
            if self.data_root:
                data_root_abs = self.data_root.resolve()
                config['path'] = str(data_root_abs)
                config['train'] = str(data_root_abs / train_path)
                config['val'] = str(data_root_abs / val_path)
                config['test'] = str(data_root_abs / test_path)
            else:
                logger.error("使用绝对路径时必须设置data_root")
                raise ValueError("使用绝对路径时必须设置data_root")
                
        return config
        
    def save_config(self, 
                   output_path: Union[str, Path],
                   train_path: str = "train/images",
                   val_path: str = "val/images",
                   test_path: str = "test/images",
                   use_relative_paths: bool = True,
                   overwrite: bool = True) -> bool:
        """
        保存data.yaml配置文件
        
        Args:
            output_path: 输出文件路径
            train_path: 训练集路径
            val_path: 验证集路径
            test_path: 测试集路径
            use_relative_paths: 是否使用相对路径
            overwrite: 是否覆盖已存在的文件
            
        Returns:
            bool: 保存是否成功
        """
        try:
            output_path = Path(output_path)
            
            # 检查文件是否存在
            if output_path.exists() and not overwrite:
                logger.warning(f"文件已存在且不允许覆盖: {output_path}")
                return False
                
            # 生成配置
            config = self.generate_config(train_path, val_path, test_path, use_relative_paths)
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存配置文件
            success = write_yaml(config, output_path)
            
            if success:
                logger.info(f"data.yaml配置文件已保存: {output_path}")
                logger.info("配置内容:")
                for key, value in config.items():
                    logger.info(f"  {key}: {value}")
            else:
                logger.error(f"保存data.yaml失败: {output_path}")
                
            return success
            
        except Exception as e:
            logger.error(f"保存data.yaml配置文件失败: {e}")
            return False
            
    def validate_config(self, config_path: Union[str, Path]) -> bool:
        """
        验证data.yaml配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            bool: 配置是否有效
        """
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                logger.error(f"配置文件不存在: {config_path}")
                return False
                
            # 读取配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # 检查必需字段
            required_fields = ['path', 'train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in config:
                    logger.error(f"配置文件缺少必需字段: {field}")
                    return False
                    
            # 检查类别数量一致性
            if config['nc'] != len(config['names']):
                logger.error(f"类别数量不一致: nc={config['nc']}, names长度={len(config['names'])}")
                return False
                
            # 检查路径是否存在
            data_root = Path(config['path'])
            paths_to_check = ['train', 'val']
            if 'test' in config:
                paths_to_check.append('test')
                
            for path_key in paths_to_check:
                if path_key in config:
                    path_value = config[path_key]
                    if Path(path_value).is_absolute():
                        full_path = Path(path_value)
                    else:
                        full_path = data_root / path_value
                        
                    if not full_path.exists():
                        logger.warning(f"路径不存在: {full_path} (来自 {path_key})")
                        
            logger.info(f"配置文件验证通过: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"验证配置文件失败: {e}")
            return False


def create_data_yaml(data_root: Union[str, Path],
                    class_names: List[str],
                    output_path: Union[str, Path],
                    train_path: str = "train/images",
                    val_path: str = "val/images",
                    test_path: str = "test/images",
                    use_relative_paths: bool = True) -> bool:
    """
    快速创建data.yaml配置文件
    
    Args:
        data_root: 数据根目录
        class_names: 类别名称列表
        output_path: 输出文件路径
        train_path: 训练集路径
        val_path: 验证集路径
        test_path: 测试集路径
        use_relative_paths: 是否使用相对路径
        
    Returns:
        bool: 创建是否成功
    """
    generator = DataYamlGenerator(data_root)
    generator.set_class_names(class_names)
    
    return generator.save_config(
        output_path=output_path,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        use_relative_paths=use_relative_paths
    )


# 导出公共函数和类
__all__ = [
    'DataYamlGenerator',
    'create_data_yaml'
]
