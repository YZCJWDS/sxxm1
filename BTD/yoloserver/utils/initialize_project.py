#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @FileName: initialize_project.py
# @Time: 2025/6/24 09:09:24
# @Author: BTD Team
# @Project: BTD项目初始化脚本，检查并创建必要的项目结构，提示用户将原始数据存放到指定的位置
# @Function: 项目初始化

import logging
import sys
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

try:
    from logger import setup_logger
    from logger import time_it
except ImportError:
    # 如果工具模块还不存在，使用基础实现
    def setup_logger(base_path=None, log_type="init", model_name=None,
                    log_level=logging.INFO, logger_name="BTD_Initialize"):
        """基础日志设置"""
        # model_name 参数保留以兼容接口，但在基础实现中不使用
        if base_path:
            Path(base_path).mkdir(parents=True, exist_ok=True)
            log_file = Path(base_path) / f"{log_type}.log"
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(level=log_level)
        return logging.getLogger(logger_name)

    def time_it(iterations=1, name="操作", logger_instance=None):
        """基础计时装饰器"""
        # iterations 参数保留以兼容接口，但在基础实现中不使用
        def decorator(func):
            def wrapper(*args, **kwargs):
                if logger_instance:
                    logger_instance.info(f"开始执行: {name}")
                result = func(*args, **kwargs)
                if logger_instance:
                    logger_instance.info(f"完成执行: {name}")
                return result
            return wrapper
        return decorator

# 定义项目路径常量
YOLOSERVER_ROOT = Path(__file__).parent.parent.parent  # 项目根目录
CONFIGS_DIR = YOLOSERVER_ROOT / "yoloserver" / "configs"  # 配置文件目录
DATA_DIR = YOLOSERVER_ROOT / "yoloserver" / "data"  # 数据集目录
RUNS_DIR = YOLOSERVER_ROOT / "yoloserver" / "runs"  # 模型运行结果目录
LOGS_DIR = YOLOSERVER_ROOT / "logs"  # 日志目录
MODEL_DIR = YOLOSERVER_ROOT / "yoloserver" / "models"  # 模型目录
PRETRAINED_DIR = MODEL_DIR / "pretrained"  # 预训练模型存放位置
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"  # 检查点目录
SCRIPTS_DIR = YOLOSERVER_ROOT / "yoloserver" / "scripts"  # 脚本目录
RAW_IMAGES_DIR = DATA_DIR / "raw" / "images"  # 原始图像目录
ORIGINAL_ANNOTATIONS_DIR = DATA_DIR / "raw" / "original_annotations"  # 原始标注目录

# 第一步：配置日志记录
logger = setup_logger(
    base_path=LOGS_DIR,
    log_type="init_project",
    model_name=None,
    log_level=logging.INFO,
    logger_name="YOLO_Initialize_Project"
)

# 第二步：定义项目初始化函数
@time_it(iterations=1, name="项目初始化", logger_instance=logger)
def initialize_project():
    """
    检查并创建项目所需的文件夹结构
    :return: None
    """
    logger.info("开始初始化项目".center(60, "="))
    logger.info(f"当前项目的根目录为: {YOLOSERVER_ROOT.resolve()}")
    
    created_dirs = []
    existing_dirs = []
    raw_data_status = []
    
    # 定义需要创建的标准目录结构
    standard_dirs_to_create = [
        CONFIGS_DIR,
        DATA_DIR,
        RUNS_DIR,
        MODEL_DIR,
        CHECKPOINTS_DIR,
        PRETRAINED_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        DATA_DIR / "train" / "images",
        DATA_DIR / "val" / "images",
        DATA_DIR / "test" / "images",
        DATA_DIR / "train" / "labels",
        DATA_DIR / "val" / "labels",
        DATA_DIR / "test" / "labels",
    ]
    
    logger.info("检查并创建核心项目目录结构".center(80, "="))
    
    for d in standard_dirs_to_create:
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
                logger.info(f"已经创建的目录: {d.relative_to(YOLOSERVER_ROOT)}")
                created_dirs.append(d)
            except Exception as e:
                logger.error(f"创建目录: {d.relative_to(YOLOSERVER_ROOT)} 失败: {e}")
                created_dirs.append(f"创建目录: {d.relative_to(YOLOSERVER_ROOT)} 失败: {e}")
        else:
            logger.info(f"检测到已存在的目录: {d.relative_to(YOLOSERVER_ROOT)}")
            existing_dirs.append(d.relative_to(YOLOSERVER_ROOT))
    
    logger.info("核心项目文件夹结构检查以及创建完成".center(60, "="))
    
    # 第三步：检查原始数据集目录并给出提示
    logger.info("开始检查原始数据集目录".center(60, "="))
    
    raw_dirs_to_check = {
        "原始图像文件": RAW_IMAGES_DIR,
        "原始标注文件": ORIGINAL_ANNOTATIONS_DIR,
    }
    
    for desc, raw_dir in raw_dirs_to_check.items():
        if not raw_dir.exists():
            msg = (
                f"!! 原始{desc}目录不存在，请将原始数据集数据放置此目录下，"
                f"并确保目录结构正确，以便后续数据集转换正常执行，期望结构为: {raw_dir.resolve()}"
            )
            logger.warning(msg)
            raw_data_status.append(f"{raw_dir.relative_to(YOLOSERVER_ROOT)}: 不存在，需要手动创建并放置原始数据")
        else:
            if not any(raw_dir.iterdir()):
                msg = f"原始{desc}目录已经存在，但内容为空，请将原始{desc}放在此目录下，以便后续数据集转换"
                logger.warning(msg)
                raw_data_status.append(f"{raw_dir.relative_to(YOLOSERVER_ROOT)}: 已经存在，但内容为空，需要放置原始数据")
            else:
                logger.info(f"原始{desc}目录已经存在，{raw_dir.relative_to(YOLOSERVER_ROOT)} 包含原始文件")
                raw_data_status.append(f"{raw_dir.relative_to(YOLOSERVER_ROOT)}: 已经存在")
    
    # 第四步：汇总所有的检查结果和创建结果
    logger.info("项目初始化结果汇总".center(80, "="))
    
    if created_dirs:
        logger.info(f"一共创建了 {len(created_dirs)} 个目录")
        for d in created_dirs:
            logger.info(f"- {d}")
    else:
        logger.info("本次初始化没有创建任何目录")
    
    if existing_dirs:
        logger.info(f"一共检查到 {len(existing_dirs)} 个已经存在的目录")
        for d in existing_dirs:
            logger.info(f"- {d}")
    
    if raw_data_status:
        logger.info(f"原始数据目录状态:")
        for status in raw_data_status:
            logger.info(f"- {status}")
    
    logger.info("项目初始化完成".center(60, "="))
    
    # 给出下一步操作建议
    logger.info("下一步操作建议:")
    logger.info(f"1. 将原始图像文件放入: {RAW_IMAGES_DIR.relative_to(YOLOSERVER_ROOT)}")
    logger.info(f"2. 将原始标注文件放入: {ORIGINAL_ANNOTATIONS_DIR.relative_to(YOLOSERVER_ROOT)}")
    logger.info("3. 运行数据处理脚本进行格式转换")
    logger.info("4. 使用统一管理工具: python main.py --help")
    logger.info("5. 开始训练: python main.py train dataset.yaml")


def create_config_files():
    """创建基础配置文件"""
    
    # YOLO模型配置
    yolo_config = {
        'model': {
            'name': 'yolov8n',
            'nc': 80,  # 类别数量
            'names': ['person', 'bicycle', 'car'],  # 示例类别名称
        },
        'train': {
            'epochs': 100,
            'batch': 16,  # 修正为正确的YOLO参数名
            'imgsz': 640,  # 修正为正确的YOLO参数名
            'lr0': 0.01,
            'weight_decay': 0.0005,
        },
        'data': {
            'train': './data/train',
            'val': './data/val',
            'test': './data/test',
        }
    }
    
    # 数据集配置 - 使用绝对路径避免路径解析问题
    data_dir = DATA_DIR  # 使用已定义的DATA_DIR常量
    dataset_config = {
        'path': str(data_dir),  # 使用绝对路径
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 4,  # 脑瘤检测4个类别
        'names': ['objects', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
    }
    
    # 保存配置文件
    config_files = [
        (CONFIGS_DIR / 'model_config.yaml', yolo_config),
        (CONFIGS_DIR / 'dataset_config.yaml', dataset_config)
    ]
    
    for file_path, config_data in config_files:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        print(f"✓ 创建配置文件: {file_path}")


def create_init_files():
    """创建Python包初始化文件"""
    
    init_files = [
        YOLOSERVER_ROOT / "yoloserver" / "__init__.py",
        YOLOSERVER_ROOT / "yoloserver" / "utils" / "__init__.py",
        YOLOSERVER_ROOT / "yoloserver" / "scripts" / "__init__.py"
    ]
    
    for init_file in init_files:
        init_file.touch()
        print(f"✓ 创建初始化文件: {init_file.relative_to(YOLOSERVER_ROOT)}")


def create_gitignore():
    """创建.gitignore文件"""
    
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
yoloserver/runs/
yoloserver/data/raw/images/*
yoloserver/data/train/images/*
yoloserver/data/val/images/*
yoloserver/data/test/images/*
yoloserver/models/checkpoints/*
yoloserver/models/pretrained/*
!yoloserver/data/**/README.md
!yoloserver/models/**/README.md

# Logs
*.log
logs/

# Node modules (for web frontend)
BTDWeb/node_modules/
BTDWeb/dist/
BTDWeb/.env.local
BTDWeb/.env.production

# Build outputs
BTDUi/build/
BTDUi/dist/
"""
    
    gitignore_file = YOLOSERVER_ROOT / '.gitignore'
    with open(gitignore_file, 'w', encoding='utf-8') as f:
        f.write(gitignore_content.strip())
    print(f"✓ 创建.gitignore文件: {gitignore_file.relative_to(YOLOSERVER_ROOT)}")


def create_readme_files():
    """创建README文件"""
    
    readme_files = {
        DATA_DIR / 'README.md': """# 数据目录说明

## 目录结构
- `raw/`: 原始数据
  - `images/`: 原始图片
  - `original_annotations/`: 原始标注文件
  - `yolo_staged_labels/`: YOLO格式标注文件
- `train/`: 训练数据
- `val/`: 验证数据  
- `test/`: 测试数据

## 使用说明
1. 将原始图片放入 `raw/images/` 目录
2. 将原始标注文件放入 `raw/original_annotations/` 目录
3. 运行数据转换脚本生成YOLO格式标注
4. 运行数据分割脚本将数据分配到train/val/test目录
""",
        
        MODEL_DIR / 'README.md': """# 模型目录说明

## 目录结构
- `checkpoints/`: 训练检查点文件
- `pretrained/`: 预训练模型文件

## 使用说明
1. 下载预训练模型到 `pretrained/` 目录
2. 训练过程中的检查点会自动保存到 `checkpoints/` 目录
3. 最佳模型会保存为 `best.pt`
""",
        
        RUNS_DIR / 'README.md': """# 运行结果目录

## 目录结构
- `train/`: 训练结果
- `val/`: 验证结果
- `detect/`: 检测结果

## 说明
此目录用于保存模型训练、验证和检测的结果文件
"""
    }
    
    for file_path, content in readme_files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 创建说明文件: {file_path.relative_to(YOLOSERVER_ROOT)}")


def main():
    """主函数"""
    print("=" * 50)
    print("BTD项目初始化脚本")
    print("=" * 50)
    
    try:
        # 执行项目初始化
        initialize_project()
        
        # 创建配置文件
        print("\n创建基础配置文件...")
        create_config_files()
        
        # 创建初始化文件
        print("\n创建Python包初始化文件...")
        create_init_files()
        
        # 创建.gitignore
        print("\n创建.gitignore文件...")
        create_gitignore()
        
        # 创建README文件
        print("\n创建说明文件...")
        create_readme_files()
        
        print("\n" + "=" * 50)
        print("✅ 项目初始化完成！")
        print("=" * 50)
        
        print("\n💡 提示: 现在可以使用 'python main.py --help' 查看所有可用功能！")
        
    except Exception as e:
        logger.error(f"初始化过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
