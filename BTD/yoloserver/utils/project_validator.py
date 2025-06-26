#!/usr/bin/env python3
"""
项目结构验证工具
验证项目是否符合设计要求，检查核心功能完整性
"""

from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util
import sys

class ProjectValidator:
    """项目结构验证器"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.yoloserver_root = project_root / "yoloserver"
        
    def validate_directory_structure(self) -> Dict[str, bool]:
        """验证目录结构"""
        required_dirs = {
            "yoloserver": self.yoloserver_root,
            "yoloserver/configs": self.yoloserver_root / "configs",
            "yoloserver/data": self.yoloserver_root / "data",
            "yoloserver/data/raw": self.yoloserver_root / "data" / "raw",
            "yoloserver/data/raw/images": self.yoloserver_root / "data" / "raw" / "images",
            "yoloserver/data/raw/original_annotations": self.yoloserver_root / "data" / "raw" / "original_annotations",
            "yoloserver/data/train": self.yoloserver_root / "data" / "train",
            "yoloserver/data/val": self.yoloserver_root / "data" / "val",
            "yoloserver/data/test": self.yoloserver_root / "data" / "test",
            "yoloserver/models": self.yoloserver_root / "models",
            "yoloserver/models/checkpoints": self.yoloserver_root / "models" / "checkpoints",
            "yoloserver/models/pretrained": self.yoloserver_root / "models" / "pretrained",
            "yoloserver/runs": self.yoloserver_root / "runs",
            "yoloserver/scripts": self.yoloserver_root / "scripts",
            "yoloserver/utils": self.yoloserver_root / "utils",
            "logs": self.project_root / "logs",
            "docs": self.project_root / "docs",
            "examples": self.project_root / "examples",
            "BTDWeb": self.project_root / "BTDWeb",
            "BTDUi": self.project_root / "BTDUi",
        }
        
        results = {}
        for name, path in required_dirs.items():
            results[name] = path.exists() and path.is_dir()
            
        return results
    
    def validate_core_files(self) -> Dict[str, bool]:
        """验证核心文件"""
        required_files = {
            "main.py": self.project_root / "main.py",
            "README.md": self.project_root / "README.md",
            "yoloserver/__init__.py": self.yoloserver_root / "__init__.py",
            "yoloserver/utils/__init__.py": self.yoloserver_root / "utils" / "__init__.py",
            "yoloserver/utils/initialize_project.py": self.yoloserver_root / "utils" / "initialize_project.py",
            "yoloserver/utils/logger.py": self.yoloserver_root / "utils" / "logger.py",
            "yoloserver/utils/path_utils.py": self.yoloserver_root / "utils" / "path_utils.py",
            "yoloserver/utils/data_converter.py": self.yoloserver_root / "utils" / "data_converter.py",
            "yoloserver/scripts/__init__.py": self.yoloserver_root / "scripts" / "__init__.py",
            "yoloserver/scripts/enhanced_train.py": self.yoloserver_root / "scripts" / "enhanced_train.py",
            "yoloserver/scripts/model_manager.py": self.yoloserver_root / "scripts" / "model_manager.py",
            "yoloserver/scripts/dataset_analyzer.py": self.yoloserver_root / "scripts" / "dataset_analyzer.py",
            "yoloserver/scripts/project_manager.py": self.yoloserver_root / "scripts" / "project_manager.py",
            "yoloserver/scripts/config_manager.py": self.yoloserver_root / "scripts" / "config_manager.py",
        }
        
        results = {}
        for name, path in required_files.items():
            results[name] = path.exists() and path.is_file()
            
        return results
    
    def validate_module_imports(self) -> Dict[str, Tuple[bool, str]]:
        """验证模块导入"""
        modules_to_test = [
            ("yoloserver.utils.logger", "get_logger"),
            ("yoloserver.utils.path_utils", "ensure_dir"),
            ("yoloserver.utils.data_converter", "convert_coco_to_yolo"),
            ("yoloserver.utils.file_utils", "copy_files"),
        ]
        
        results = {}
        original_path = sys.path.copy()
        
        try:
            # 添加项目根目录到路径
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            
            for module_name, function_name in modules_to_test:
                try:
                    module = importlib.import_module(module_name)
                    if hasattr(module, function_name):
                        results[module_name] = (True, f"✅ {function_name} 可用")
                    else:
                        results[module_name] = (False, f"❌ 缺少函数 {function_name}")
                except ImportError as e:
                    results[module_name] = (False, f"❌ 导入失败: {e}")
                except Exception as e:
                    results[module_name] = (False, f"❌ 其他错误: {e}")
        
        finally:
            # 恢复原始路径
            sys.path = original_path
            
        return results
    
    def check_configuration_files(self) -> Dict[str, bool]:
        """检查配置文件"""
        config_files = {
            "model_config.yaml": self.yoloserver_root / "configs" / "model_config.yaml",
            "dataset_config.yaml": self.yoloserver_root / "configs" / "dataset_config.yaml",
        }
        
        results = {}
        for name, path in config_files.items():
            results[name] = path.exists()
            
        return results
    
    def generate_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("=" * 80)
        report.append("BTD 项目结构验证报告")
        report.append("=" * 80)
        
        # 目录结构验证
        report.append("\n📁 目录结构验证:")
        dir_results = self.validate_directory_structure()
        passed_dirs = sum(1 for v in dir_results.values() if v)
        total_dirs = len(dir_results)
        
        report.append(f"通过: {passed_dirs}/{total_dirs} 个目录")
        
        for name, passed in dir_results.items():
            status = "✅" if passed else "❌"
            report.append(f"  {status} {name}")
        
        # 核心文件验证
        report.append("\n📄 核心文件验证:")
        file_results = self.validate_core_files()
        passed_files = sum(1 for v in file_results.values() if v)
        total_files = len(file_results)
        
        report.append(f"通过: {passed_files}/{total_files} 个文件")
        
        for name, passed in file_results.items():
            status = "✅" if passed else "❌"
            report.append(f"  {status} {name}")
        
        # 模块导入验证
        report.append("\n🔧 模块导入验证:")
        import_results = self.validate_module_imports()
        passed_imports = sum(1 for v, _ in import_results.values() if v)
        total_imports = len(import_results)
        
        report.append(f"通过: {passed_imports}/{total_imports} 个模块")
        
        for module_name, (passed, message) in import_results.items():
            report.append(f"  {message}")
        
        # 配置文件验证
        report.append("\n⚙️ 配置文件验证:")
        config_results = self.check_configuration_files()
        passed_configs = sum(1 for v in config_results.values() if v)
        total_configs = len(config_results)
        
        report.append(f"通过: {passed_configs}/{total_configs} 个配置文件")
        
        for name, passed in config_results.items():
            status = "✅" if passed else "❌"
            report.append(f"  {status} {name}")
        
        # 总体评估
        total_checks = total_dirs + total_files + total_imports + total_configs
        total_passed = passed_dirs + passed_files + passed_imports + passed_configs
        
        report.append("\n" + "=" * 80)
        report.append("📊 总体评估:")
        report.append(f"总检查项: {total_checks}")
        report.append(f"通过项: {total_passed}")
        report.append(f"通过率: {total_passed/total_checks*100:.1f}%")
        
        if total_passed == total_checks:
            report.append("🎉 项目结构完全符合要求！")
        elif total_passed / total_checks >= 0.8:
            report.append("✅ 项目结构基本符合要求，有少量问题需要修复")
        else:
            report.append("⚠️ 项目结构存在较多问题，需要重点关注")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """主函数"""
    project_root = Path(__file__).parent.parent.parent
    validator = ProjectValidator(project_root)
    
    print(validator.generate_report())

if __name__ == "__main__":
    main()
