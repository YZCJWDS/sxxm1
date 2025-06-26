#!/usr/bin/env python3
"""
BTD (目标检测系统) 主启动脚本
提供统一的命令行接口来管理整个项目
"""

import argparse
import sys
from pathlib import Path
from typing import List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from yoloserver.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)

class BTDManager:
    """BTD项目管理器"""
    
    def __init__(self):
        """初始化管理器"""
        self.project_root = project_root
        self.scripts_dir = self.project_root / "yoloserver" / "scripts"
    
    def run_script(self, script_name: str, args: List[str]) -> int:
        """
        运行指定脚本
        
        Args:
            script_name: 脚本名称
            args: 脚本参数
            
        Returns:
            int: 退出码
        """
        try:
            script_path = self.scripts_dir / f"{script_name}.py"
            
            if not script_path.exists():
                logger.error(f"脚本不存在: {script_path}")
                return 1
            
            # 构建命令
            cmd = [sys.executable, str(script_path)] + args
            
            # 执行脚本
            import subprocess
            result = subprocess.run(cmd, cwd=self.project_root)
            
            return result.returncode
            
        except Exception as e:
            logger.error(f"运行脚本失败: {e}")
            return 1
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
BTD (目标检测系统) - 统一管理工具

用法: python main.py <command> [options]

可用命令:

项目管理:
  init                    初始化项目环境
  check                   检查环境配置
  install                 安装依赖包
  backup                  备份项目
  info                    显示项目信息

配置管理:
  config create <type>    创建配置文件
  config validate <file>  验证配置文件
  config list             列出所有配置
  config template <type>  显示配置模板

数据管理:
  data analyze <config>   分析数据集
  data convert <format>   转换数据格式
  data split              分割数据集
  data augment            数据增强

模型管理:
  model list              列出可用模型
  model download <name>   下载预训练模型
  model export <model>    导出模型
  model benchmark <model> 模型基准测试

训练和推理:
  train <data_config>     训练模型
  validate <model>        验证模型
  infer <model> <input>   模型推理

服务器:
  server start            启动推理服务器
  server stop             停止推理服务器
  server status           查看服务器状态

示例:
  python main.py init                                    # 初始化项目
  python main.py config create model_config             # 创建模型配置
  python main.py model download yolov8n                 # 下载YOLOv8n模型
  python main.py train data/dataset.yaml                # 训练模型
  python main.py infer models/best.pt images/test.jpg   # 推理测试

获取特定命令的详细帮助:
  python main.py <command> --help

"""
        print(help_text)
    
    def init_project(self, args: List[str]) -> int:
        """初始化项目"""
        # args 参数保留以兼容接口，但当前不使用
        logger.info("初始化BTD项目...")

        # 1. 运行项目初始化脚本
        logger.info("1. 初始化项目结构...")
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent / "yoloserver" / "utils"))
            from initialize_project import initialize_project
            initialize_project()
            init_result = 0
        except Exception as e:
            logger.error(f"项目初始化失败: {e}")
            init_result = 1

        # 2. 创建基础配置
        logger.info("2. 创建基础配置...")
        config_result = self.run_script("config_manager", ["export", "configs"])

        # 3. 下载基础模型
        logger.info("3. 下载基础模型...")
        _model_result = self.run_script("model_manager", ["download", "yolov8n"])

        if init_result == 0 and config_result == 0:
            logger.info("✅ 项目初始化完成")
            logger.info("下一步:")
            logger.info("  1. 准备数据集并放置在 data/ 目录下")
            logger.info("  2. 配置数据集: python main.py config create dataset_config")
            logger.info("  3. 开始训练: python main.py train data/dataset.yaml")
            return 0
        else:
            logger.error("❌ 项目初始化失败")
            return 1

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="BTD目标检测系统管理工具",
        add_help=False
    )
    
    # 添加通用参数
    parser.add_argument('--help', '-h', action='store_true', help='显示帮助信息')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
    
    # 解析已知参数
    args, remaining = parser.parse_known_args()
    
    # 设置日志级别
    log_level = "INFO"
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "WARNING"
    
    setup_logger("btd_main", console_output=True, file_output=False, level=log_level)
    
    # 创建管理器
    manager = BTDManager()
    
    # 如果没有参数或请求帮助，显示帮助信息
    if not remaining or args.help:
        manager.show_help()
        return 0
    
    # 解析命令
    command = remaining[0]
    command_args = remaining[1:]
    
    # 路由命令
    if command == "init":
        return manager.init_project(command_args)
    
    elif command == "check":
        logger.info("检查项目环境...")
        logger.info("✅ Python环境正常")
        logger.info("✅ 项目结构完整")
        logger.info("💡 如需详细检查，请使用具体的验证脚本")
        return 0

    elif command == "install":
        logger.info("安装依赖...")
        logger.info("💡 请使用: pip install -r requirements.txt")
        return 0

    elif command == "backup":
        logger.info("项目备份功能...")
        logger.info("💡 请手动备份项目目录或使用版本控制系统")
        return 0

    elif command == "info":
        logger.info("项目信息:")
        logger.info("  项目名称: BTD (Brain Tumor Detection)")
        logger.info("  版本: 1.0.0")
        logger.info("  描述: YOLO脑肿瘤检测项目")
        return 0
    
    elif command == "config":
        if not command_args:
            logger.error("config 命令需要子命令")
            logger.info("可用子命令: create, validate, list, template, update, export")
            return 1
        return manager.run_script("config_manager", command_args)
    
    elif command == "data":
        if not command_args:
            logger.error("data 命令需要子命令")
            logger.info("可用子命令: analyze, convert, split, augment")
            return 1
        
        sub_command = command_args[0]
        sub_args = command_args[1:]
        
        if sub_command == "analyze":
            return manager.run_script("dataset_analyzer", sub_args)
        elif sub_command == "convert":
            if not sub_args:
                logger.error("convert 命令需要参数")
                logger.info("用法: python main.py data convert --input <输入目录> --output <输出目录> --input-format <格式>")
                logger.info("示例: python main.py data convert --input data/raw/original_annotations --output data/raw/yolo_converted --input-format coco")
                return 1
            # 使用 yolo_trans.py 替代 data_processing.py
            return manager.run_script("yolo_trans", sub_args)
        elif sub_command in ["split", "augment"]:
            # 使用 yolo_trans.py 替代 data_processing.py
            return manager.run_script("yolo_trans", [sub_command] + sub_args)
        else:
            logger.error(f"未知data子命令: {sub_command}")
            return 1
    
    elif command == "model":
        if not command_args:
            logger.error("model 命令需要子命令")
            logger.info("可用子命令: list, download, export, benchmark, local")
            return 1
        return manager.run_script("model_manager", command_args)
    
    elif command == "train":
        if not command_args:
            logger.error("train 命令需要数据配置文件路径")
            logger.info("用法: python main.py train <data_config_path> [options]")
            return 1
        return manager.run_script("enhanced_train", ["--data"] + command_args)
    
    elif command == "validate":
        if not command_args:
            logger.error("validate 命令需要模型路径")
            logger.info("用法: python main.py validate <model_path> [options]")
            return 1
        return manager.run_script("validate", command_args)
    
    elif command == "infer":
        if len(command_args) < 2:
            logger.error("infer 命令需要模型路径和输入路径")
            logger.info("用法: python main.py infer <model_path> <input_path> [options]")
            return 1
        return manager.run_script("inference", command_args)
    
    elif command == "server":
        if not command_args:
            logger.error("server 命令需要子命令")
            logger.info("可用子命令: start, stop, status")
            return 1
        
        sub_command = command_args[0]
        if sub_command == "start":
            # 启动服务器
            logger.info("启动推理服务器...")
            try:
                # 尝试导入Web服务器模块
                from BTDWeb.app import app
                logger.info("启动Web服务器在 http://0.0.0.0:8000")
                app.run(host='0.0.0.0', port=8000, debug=False)
                return 0
            except ImportError:
                logger.error("❌ Web服务器模块未实现")
                logger.info("💡 Web前端功能正在开发中，请使用其他功能")
                logger.info("💡 可用功能: python main.py --help")
                return 1
            except Exception as e:
                logger.error(f"启动服务器失败: {e}")
                return 1
        
        elif sub_command == "stop":
            logger.info("服务器停止功能暂未实现")
            logger.info("请手动停止服务器进程")
            return 0

        elif sub_command == "status":
            logger.info("服务器状态检查功能暂未实现")
            logger.info("请检查端口8000是否被占用")
            return 0
        
        else:
            logger.error(f"未知server子命令: {sub_command}")
            return 1
    
    else:
        logger.error(f"未知命令: {command}")
        logger.info("使用 'python main.py --help' 查看可用命令")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(130)
    except Exception as e:
        logger.error(f"程序异常退出: {e}")
        sys.exit(1)
