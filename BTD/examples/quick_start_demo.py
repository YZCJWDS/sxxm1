#!/usr/bin/env python3
"""
BTD快速开始演示脚本
展示如何使用新的统一管理工具进行完整的目标检测工作流程
"""

import sys
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"执行命令: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        
        if result.stdout:
            print("输出:")
            print(result.stdout)
        
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ 命令执行成功")
        else:
            print(f"❌ 命令执行失败 (退出码: {result.returncode})")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 命令执行异常: {e}")
        return False

def main():
    """主演示流程"""
    print("🎯 BTD目标检测系统 - 快速开始演示")
    print("本演示将展示BTD的主要功能和工作流程")
    
    # 演示命令列表
    demos = [
        {
            "cmd": ["python", "main.py", "--help"],
            "desc": "查看统一管理工具帮助信息"
        },
        {
            "cmd": ["python", "main.py", "check"],
            "desc": "检查项目环境配置"
        },
        {
            "cmd": ["python", "main.py", "info"],
            "desc": "显示项目信息"
        },
        {
            "cmd": ["python", "main.py", "config", "list"],
            "desc": "列出项目配置文件"
        },
        {
            "cmd": ["python", "main.py", "config", "template", "model_config"],
            "desc": "查看模型配置模板"
        },
        {
            "cmd": ["python", "main.py", "model", "list"],
            "desc": "列出可用的预训练模型"
        },
        {
            "cmd": ["python", "main.py", "model", "local"],
            "desc": "查看本地模型文件"
        }
    ]
    
    # 可选演示 (需要用户确认)
    optional_demos = [
        {
            "cmd": ["python", "main.py", "model", "download", "yolov8n"],
            "desc": "下载YOLOv8n预训练模型 (约6MB)"
        },
        {
            "cmd": ["python", "main.py", "config", "create", "dataset_config", "--output", "examples/demo_dataset.yaml"],
            "desc": "创建示例数据集配置文件"
        },
        {
            "cmd": ["python", "main.py", "backup", "--output", "examples/demo_backup", "--no-models"],
            "desc": "创建项目备份 (不包含模型文件)"
        }
    ]
    
    # 执行基础演示
    print("\n📋 基础功能演示:")
    success_count = 0
    
    for i, demo in enumerate(demos, 1):
        print(f"\n[{i}/{len(demos)}]", end="")
        if run_command(demo["cmd"], demo["desc"]):
            success_count += 1
        
        # 暂停让用户查看结果
        input("\n按Enter键继续下一个演示...")
    
    # 可选演示
    print(f"\n📊 基础演示完成: {success_count}/{len(demos)} 个命令成功执行")
    
    if input("\n是否继续可选演示? (这些操作会下载文件或创建文件) [y/N]: ").lower() == 'y':
        print("\n📋 可选功能演示:")
        
        for i, demo in enumerate(optional_demos, 1):
            print(f"\n[{i}/{len(optional_demos)}]", end="")
            if run_command(demo["cmd"], demo["desc"]):
                success_count += 1
            
            input("\n按Enter键继续下一个演示...")
    
    # 显示总结
    print(f"\n{'='*60}")
    print("🎉 演示完成!")
    print(f"{'='*60}")
    print("通过这个演示，您已经了解了BTD的主要功能:")
    print("✅ 环境检查和项目信息")
    print("✅ 配置文件管理")
    print("✅ 模型管理")
    print("✅ 项目备份")
    print("\n下一步建议:")
    print("1. 准备您的数据集")
    print("2. 创建数据集配置: python main.py config create dataset_config")
    print("3. 分析数据集: python main.py data analyze dataset.yaml")
    print("4. 开始训练: python main.py train dataset.yaml")
    print("5. 启动Web服务: python main.py server start")
    print("\n📚 更多信息请查看 README.md 文件")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示过程中发生错误: {e}")
