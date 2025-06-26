#!/usr/bin/env python3
"""
BTD项目依赖安装脚本
自动检测和安装项目所需的依赖包
"""

import subprocess
import sys
import importlib
from pathlib import Path

def check_package(package_name: str, import_name: str = None) -> bool:
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package: str) -> bool:
    """安装单个包"""
    try:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package} 安装失败")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("BTD项目依赖安装脚本")
    print("=" * 60)
    
    # 核心依赖包
    core_packages = [
        ("ultralytics", "ultralytics"),
        ("opencv-python", "cv2"),
        ("pyyaml", "yaml"),
        ("pillow", "PIL"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("tqdm", "tqdm"),
        ("pandas", "pandas"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
    ]
    
    # 可选依赖包
    optional_packages = [
        ("psutil", "psutil"),
        ("requests", "requests"),
    ]
    
    print("\n🔍 检查核心依赖...")
    missing_core = []
    for package, import_name in core_packages:
        if check_package(package, import_name):
            print(f"✅ {package} 已安装")
        else:
            print(f"❌ {package} 未安装")
            missing_core.append(package)
    
    print("\n🔍 检查可选依赖...")
    missing_optional = []
    for package, import_name in optional_packages:
        if check_package(package, import_name):
            print(f"✅ {package} 已安装")
        else:
            print(f"⚠️  {package} 未安装 (可选)")
            missing_optional.append(package)
    
    # 安装缺失的核心依赖
    if missing_core:
        print(f"\n📦 安装缺失的核心依赖 ({len(missing_core)}个)...")
        failed_core = []
        for package in missing_core:
            if not install_package(package):
                failed_core.append(package)
        
        if failed_core:
            print(f"\n❌ 以下核心依赖安装失败: {', '.join(failed_core)}")
            print("请手动安装这些依赖包")
            return 1
    else:
        print("\n✅ 所有核心依赖已安装")
    
    # 询问是否安装可选依赖
    if missing_optional:
        print(f"\n📦 发现 {len(missing_optional)} 个可选依赖未安装:")
        for package in missing_optional:
            print(f"  - {package}")
        
        response = input("\n是否安装可选依赖？(y/N): ").strip().lower()
        if response == 'y':
            print("\n安装可选依赖...")
            for package in missing_optional:
                install_package(package)
        else:
            print("跳过可选依赖安装")
    else:
        print("\n✅ 所有可选依赖已安装")
    
    print("\n" + "=" * 60)
    print("✅ 依赖检查和安装完成！")
    print("=" * 60)
    
    print("\n💡 下一步:")
    print("1. 运行项目初始化: python main.py init")
    print("2. 查看帮助信息: python main.py --help")
    print("3. 开始使用BTD: python main.py check")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ 安装被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ 安装过程中发生错误: {e}")
        sys.exit(1)
