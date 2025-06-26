# -*- coding:utf-8 -*-
# @FileName  :yolo_validate.py
# @Time      :2025/6/25 16:15:00
# @Author    :BTD Team
# @Project   :BrainTumorDetection
# @Function  :YOLO数据集验证入口脚本

import argparse
import sys
import logging
from pathlib import Path

# 添加项目路径到sys.path
project_root = Path(__file__).parent.parent  # BTD/yoloserver
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "utils"))

# 导入验证函数
try:
    from utils.dataset_validation import verify_dataset_config, verify_split_uniqueness, delete_invalid_files
except ImportError:
    try:
        from dataset_validation import verify_dataset_config, verify_split_uniqueness, delete_invalid_files
    except ImportError as e:
        print(f"无法导入验证模块: {e}")
        print(f"当前Python路径: {sys.path}")
        print(f"项目根目录: {project_root}")
        print(f"utils目录: {project_root / 'utils'}")
        sys.exit(1)

# 尝试导入项目的日志工具
try:
    from utils.logger import setup_logger
except ImportError:
    # 如果导入失败，创建简单的日志设置函数
    def setup_logger(name="yolo_validate", console_output=True, file_output=True):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台输出
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 文件输出
        if file_output:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


def find_data_yaml():
    """
    自动查找data.yaml文件
    
    Returns:
        Path: data.yaml文件路径，如果找不到返回None
    """
    # 获取脚本所在目录
    script_dir = Path(__file__).parent

    # 常见的data.yaml文件位置
    possible_paths = [
        script_dir / "../configs/data.yaml",  # BTD/yoloserver/configs/data.yaml
        script_dir / "../data/data.yaml",     # BTD/yoloserver/data/data.yaml
        script_dir / "../configs/dataset.yaml",
        script_dir / "../../data.yaml",       # BTD/data.yaml
        script_dir / "../../dataset.yaml",    # BTD/dataset.yaml
        Path("data.yaml"),                    # 当前目录
        Path("dataset.yaml"),                 # 当前目录
        Path("configs/data.yaml"),            # 相对于当前目录
        Path("configs/dataset.yaml"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path.resolve()
    
    return None


def confirm_deletion(invalid_count: int) -> bool:
    """
    询问用户是否确认删除不合法文件
    
    Args:
        invalid_count: 不合法文件数量
        
    Returns:
        bool: 用户确认结果
    """
    print(f"\n⚠️  警告：发现 {invalid_count} 个不合法文件对")
    print("删除操作不可逆，请谨慎确认！")
    print("建议先备份数据集后再执行删除操作。")
    
    while True:
        response = input("\n是否确认删除这些不合法文件？(yes/no): ").lower().strip()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("请输入 'yes' 或 'no'")


def main():
    """主函数，处理命令行参数并执行验证"""
    parser = argparse.ArgumentParser(
        description="YOLO数据集验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础验证（完整模式，检测任务）
  python scripts/yolo_validate.py

  # 指定data.yaml文件路径
  python scripts/yolo_validate.py --yaml-path data/dataset.yaml

  # 抽样验证模式
  python scripts/yolo_validate.py --mode SAMPLE
  
  # 分割任务验证
  python scripts/yolo_validate.py --task segmentation
  
  # 启用删除不合法文件选项
  python scripts/yolo_validate.py --delete-invalid
  
  # 组合使用
  python scripts/yolo_validate.py --yaml-path data.yaml --mode FULL --task detection --delete-invalid
        """
    )
    
    # 命令行参数
    parser.add_argument(
        "--yaml-path", 
        type=str, 
        help="data.yaml文件路径（如果不指定，将自动查找）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["FULL", "SAMPLE"],
        default="FULL",
        help="验证模式：FULL（完整验证，默认）或 SAMPLE（抽样验证）"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        choices=["detection", "segmentation"], 
        default="detection",
        help="任务类型：detection（检测，默认）或 segmentation（分割）"
    )
    parser.add_argument(
        "--delete-invalid", 
        action="store_true",
        help="是否在验证失败后启用删除不合法文件选项"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别（默认：INFO）"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger("yolo_validate", console_output=True, file_output=True)
    logger.setLevel(getattr(logging, args.log_level))
    
    logger.info("=" * 60)
    logger.info("YOLO数据集验证工具启动")
    logger.info("=" * 60)
    logger.info(f"验证模式: {args.mode}")
    logger.info(f"任务类型: {args.task}")
    logger.info(f"删除选项: {'启用' if args.delete_invalid else '禁用'}")
    
    # 查找data.yaml文件
    if args.yaml_path:
        yaml_path = Path(args.yaml_path)
        if not yaml_path.exists():
            logger.error(f"指定的data.yaml文件不存在: {yaml_path}")
            sys.exit(1)
    else:
        yaml_path = find_data_yaml()
        if yaml_path is None:
            logger.error("未找到data.yaml文件，请使用 --yaml-path 参数指定文件路径")
            sys.exit(1)
        logger.info(f"自动找到data.yaml文件: {yaml_path}")
    
    # 执行验证
    validation_success = True
    
    try:
        # 1. 基础数据集配置验证
        logger.info("\n" + "=" * 40)
        logger.info("开始基础数据集配置验证")
        logger.info("=" * 40)
        
        config_valid, invalid_samples = verify_dataset_config(
            yaml_path=yaml_path,
            current_logger=logger,
            mode=args.mode,
            task_type=args.task
        )
        
        if not config_valid:
            validation_success = False
            logger.error("基础数据集配置验证失败")
            
            # 如果启用删除选项且存在不合法文件
            if args.delete_invalid and invalid_samples:
                if confirm_deletion(len(invalid_samples)):
                    logger.info("用户确认删除不合法文件")
                    delete_invalid_files(invalid_samples, logger)
                else:
                    logger.info("用户取消删除操作")
        else:
            logger.info("基础数据集配置验证通过")
        
        # 2. 数据集分割唯一性验证
        logger.info("\n" + "=" * 40)
        logger.info("开始数据集分割唯一性验证")
        logger.info("=" * 40)
        
        split_unique = verify_split_uniqueness(
            yaml_path=yaml_path,
            current_logger=logger
        )
        
        if not split_unique:
            validation_success = False
            logger.error("数据集分割唯一性验证失败")
        else:
            logger.info("数据集分割唯一性验证通过")
            
    except Exception as e:
        logger.error(f"验证过程中发生异常: {e}")
        validation_success = False
    
    # 输出最终结果
    logger.info("\n" + "=" * 60)
    if validation_success:
        logger.info("🎉 数据集验证全部通过！")
        logger.info("数据集已准备就绪，可以开始训练模型")
        sys.exit(0)
    else:
        logger.error("❌ 数据集验证失败！")
        logger.error("请根据上述日志信息修复数据集问题后重新验证")
        sys.exit(1)


if __name__ == "__main__":
    main()
