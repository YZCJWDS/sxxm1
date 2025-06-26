"""
日志清理工具
定期清理过期的日志文件，保持日志目录整洁
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

# 设置控制台编码为UTF-8
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except:
        pass

def clean_empty_logs(log_dir: Path, dry_run: bool = False) -> List[str]:
    """
    清理空的日志文件（0字节或只有空白内容）

    Args:
        log_dir: 日志目录
        dry_run: 是否为试运行

    Returns:
        List[str]: 被删除的空文件列表
    """
    if not log_dir.exists():
        return []

    deleted_files = []

    for log_file in log_dir.glob("*.log"):
        if not log_file.exists():
            continue

        try:
            # 检查文件大小
            if log_file.stat().st_size == 0:
                deleted_files.append(f"{log_file.name} (空文件)")
                if not dry_run:
                    log_file.unlink()
            else:
                # 检查文件内容是否只有空白
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if not content:
                        deleted_files.append(f"{log_file.name} (空内容)")
                        if not dry_run:
                            log_file.unlink()
        except Exception as e:
            print(f"检查文件失败 {log_file}: {e}")

    return deleted_files


def clean_old_logs(log_dir: Path,
                   days_to_keep: int = 7,
                   max_files_per_type: int = 10,
                   dry_run: bool = False) -> List[str]:
    """
    清理过期的日志文件
    
    Args:
        log_dir: 日志目录
        days_to_keep: 保留天数
        max_files_per_type: 每种类型最多保留文件数
        dry_run: 是否为试运行（不实际删除）
        
    Returns:
        List[str]: 被删除（或将被删除）的文件列表
    """
    if not log_dir.exists():
        return []
    
    deleted_files = []
    cutoff_time = time.time() - (days_to_keep * 24 * 3600)
    
    # 按文件类型分组
    file_groups = {}
    
    for log_file in log_dir.glob("*.log"):
        # 提取文件类型（去掉时间戳部分）
        name_parts = log_file.stem.split('_')
        if len(name_parts) >= 2:
            # 移除最后的日期时间部分
            file_type = '_'.join(name_parts[:-1])
        else:
            file_type = log_file.stem
            
        if file_type not in file_groups:
            file_groups[file_type] = []
        file_groups[file_type].append(log_file)
    
    # 对每种类型的文件进行清理
    for file_type, files in file_groups.items():
        # 按修改时间排序（最新的在前）
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # 删除过期文件
        for log_file in files:
            # 检查文件是否还存在
            if not log_file.exists():
                continue

            should_delete = False

            # 检查文件年龄
            if log_file.stat().st_mtime < cutoff_time:
                should_delete = True
                reason = f"超过{days_to_keep}天"
            
            # 检查数量限制
            elif len([f for f in files if f.exists() and f.stat().st_mtime >= cutoff_time]) > max_files_per_type:
                # 如果该类型文件过多，删除最旧的
                if files.index(log_file) >= max_files_per_type:
                    should_delete = True
                    reason = f"超过{max_files_per_type}个文件限制"
            
            if should_delete:
                deleted_files.append(f"{log_file.name} ({reason})")
                if not dry_run:
                    try:
                        log_file.unlink()
                    except Exception as e:
                        print(f"删除文件失败 {log_file}: {e}")
    
    return deleted_files

def get_log_statistics(log_dir: Path) -> dict:
    """获取日志目录统计信息"""
    if not log_dir.exists():
        return {"total_files": 0, "total_size": 0, "file_types": {}}
    
    stats = {
        "total_files": 0,
        "total_size": 0,
        "file_types": {},
        "oldest_file": None,
        "newest_file": None
    }
    
    oldest_time = float('inf')
    newest_time = 0
    
    for log_file in log_dir.glob("*.log"):
        stats["total_files"] += 1
        file_size = log_file.stat().st_size
        stats["total_size"] += file_size
        
        # 文件类型统计
        name_parts = log_file.stem.split('_')
        if len(name_parts) >= 2:
            file_type = '_'.join(name_parts[:-1])
        else:
            file_type = log_file.stem
            
        if file_type not in stats["file_types"]:
            stats["file_types"][file_type] = {"count": 0, "size": 0}
        stats["file_types"][file_type]["count"] += 1
        stats["file_types"][file_type]["size"] += file_size
        
        # 时间统计
        mtime = log_file.stat().st_mtime
        if mtime < oldest_time:
            oldest_time = mtime
            stats["oldest_file"] = log_file.name
        if mtime > newest_time:
            newest_time = mtime
            stats["newest_file"] = log_file.name
    
    return stats

def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="BTD日志清理工具")
    parser.add_argument("--auto", "-a", action="store_true", help="自动执行清理，不询问确认")
    parser.add_argument("--dry-run", "-d", action="store_true", help="只显示将要删除的文件，不实际删除")
    args = parser.parse_args()

    # 获取项目根目录 - 从 BTD/yoloserver/utils/ 向上4级到项目根目录
    project_root = Path(__file__).parent.parent.parent.parent
    log_dir = project_root / "logs"
    
    print("=" * 60)
    print("BTD 日志清理工具")
    print("=" * 60)

    # 显示当前统计
    stats = get_log_statistics(log_dir)
    print(f"日志目录: {log_dir}")
    print(f"总文件数: {stats['total_files']}")
    print(f"总大小: {format_size(stats['total_size'])}")
    
    if stats['total_files'] > 0:
        print(f"最旧文件: {stats['oldest_file']}")
        print(f"最新文件: {stats['newest_file']}")
        
        print("\n文件类型分布:")
        for file_type, info in stats['file_types'].items():
            print(f"  {file_type}: {info['count']}个文件, {format_size(info['size'])}")
    
    # 试运行清理空文件
    print("\n" + "=" * 60)
    print("试运行清理空文件（不会实际删除文件）")
    print("=" * 60)

    empty_files = clean_empty_logs(log_dir, dry_run=True)
    if empty_files:
        print(f"发现 {len(empty_files)} 个空文件:")
        for file_info in empty_files:
            print(f"  - {file_info}")
    else:
        print("[OK] 没有发现空文件")

    # 试运行清理旧文件
    print("\n" + "=" * 60)
    print("试运行清理旧文件（不会实际删除文件）")
    print("=" * 60)

    deleted_files = clean_old_logs(log_dir, days_to_keep=7, max_files_per_type=2, dry_run=True)
    
    total_to_delete = len(empty_files) + len(deleted_files)

    if total_to_delete > 0:
        print(f"\n总计将删除 {total_to_delete} 个文件:")
        print(f"  - 空文件: {len(empty_files)} 个")
        print(f"  - 旧文件: {len(deleted_files)} 个")

        # 决定是否执行实际清理
        should_clean = False

        if args.dry_run:
            print("\n[试运行模式] 不会实际删除文件")
            should_clean = False
        elif args.auto:
            print("\n[自动模式] 开始执行清理...")
            should_clean = True
        else:
            # 询问是否执行实际清理
            print("\n是否执行实际清理？")
            print("输入 'y' 确认删除，输入其他任何内容取消:")
            try:
                response = input(">>> ").strip().lower()
                should_clean = (response == 'y' or response == 'yes')
            except (EOFError, KeyboardInterrupt):
                print("\n操作已取消")
                return

        if should_clean:
            print("\n执行实际清理...")

            # 清理空文件
            if empty_files:
                actual_empty_deleted = clean_empty_logs(log_dir, dry_run=False)
                print(f"[OK] 已删除 {len(actual_empty_deleted)} 个空文件")

            # 清理旧文件
            if deleted_files:
                actual_deleted = clean_old_logs(log_dir, days_to_keep=7, max_files_per_type=2, dry_run=False)
                print(f"[OK] 已删除 {len(actual_deleted)} 个旧文件")

            print(f"[OK] 总计删除 {len(actual_empty_deleted if empty_files else []) + len(actual_deleted if deleted_files else [])} 个文件")
        else:
            print("取消清理操作")
    else:
        print("[OK] 没有需要清理的文件")
    
    # 显示清理后统计
    final_stats = get_log_statistics(log_dir)
    print(f"\n清理后统计:")
    print(f"总文件数: {final_stats['total_files']}")
    print(f"总大小: {format_size(final_stats['total_size'])}")

if __name__ == "__main__":
    main()
