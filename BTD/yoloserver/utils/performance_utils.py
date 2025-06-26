#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :performance_utils.py
# @Time      :2025/6/24 15:37:47
# @Author    :BTD Team
# @Project   :BTD
# @Function  :放一些性能测试的工具函数
import logging
import time
from functools import wraps
from typing import Optional, Callable, Any


_default_logger = logging.getLogger(__name__)


def time_it(iterations: int = 1, name: Optional[str] = None, logger_instance: Optional[logging.Logger] = None):
    """
    一个用于记录函数执行耗时的装饰器函数，实际使用中会传入一个日志记录器
    
    Args:
        iterations: 函数执行次数，如果大于1，记录平均耗时，等于1，单次执行耗时
        name: 用于日志输出的函数类别名称
        logger_instance: 日志记录器实例
        
    Returns:
        装饰器函数
    """
    _logger_to_use = logger_instance if logger_instance is not None else _default_logger

    # 辅助函数：根据总秒数格式化为最合适的单位
    def _format_time_auto_unit(total_seconds: float) -> str:
        """
        根据总秒数自动选择并格式化为最合适的单位（微秒、毫秒、秒、分钟、小时）。
        
        Args:
            total_seconds: 总秒数
            
        Returns:
            str: 格式化后的时间字符串
        """
        if total_seconds < 0.000001:  # 小于1微秒
            return f"{total_seconds * 1_000_000:.3f} 微秒"
        elif total_seconds < 0.001:  # 小于1毫秒
            return f"{total_seconds * 1_000_000:.3f} 微秒"  # 保持微秒精度
        elif total_seconds < 1.0:  # 小于1秒
            return f"{total_seconds * 1000:.3f} 毫秒"
        elif total_seconds < 60.0:  # 小于1分钟
            return f"{total_seconds:.3f} 秒"
        elif total_seconds < 3600:  # 小于1小时
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            return f"{minutes} 分 {seconds:.3f} 秒"
        else:  # 大于等于1小时
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = total_seconds % 60
            return f"{hours} 小时 {minutes} 分 {seconds:.3f} 秒"

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_display_name = name if name is not None else func.__name__
            total_elapsed_time = 0.0
            result = None

            for i in range(iterations):
                start_time = time.perf_counter()  # 获取当前时间
                result = func(*args, **kwargs)
                end_time = time.perf_counter()  # 获取结束的时间
                total_elapsed_time += end_time - start_time
                
            avg_elapsed_time = total_elapsed_time / iterations
            formatted_avg_time = _format_time_auto_unit(avg_elapsed_time)
            
            if iterations == 1:
                _logger_to_use.info(f"性能测试：'{func_display_name}' 执行耗时: {formatted_avg_time}")
            else:
                _logger_to_use.info(f"性能测试：'{func_display_name}' 执行: {iterations} 次, 单次平均耗时: {formatted_avg_time}")
            return result
        return wrapper
    return decorator


class PerformanceProfiler:
    """
    性能分析器类，用于更复杂的性能测试场景
    """
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        初始化性能分析器
        
        Args:
            logger_instance: 日志记录器实例
        """
        self.logger = logger_instance if logger_instance is not None else _default_logger
        self.timers = {}
        self.results = {}
    
    def start_timer(self, name: str) -> None:
        """
        开始计时
        
        Args:
            name: 计时器名称
        """
        self.timers[name] = time.perf_counter()
        self.logger.debug(f"开始计时: {name}")
    
    def end_timer(self, name: str) -> float:
        """
        结束计时并返回耗时
        
        Args:
            name: 计时器名称
            
        Returns:
            float: 耗时（秒）
        """
        if name not in self.timers:
            self.logger.error(f"计时器 '{name}' 未启动")
            return 0.0
        
        elapsed_time = time.perf_counter() - self.timers[name]
        self.results[name] = elapsed_time
        
        formatted_time = self._format_time_auto_unit(elapsed_time)
        self.logger.info(f"计时结束: {name}, 耗时: {formatted_time}")
        
        del self.timers[name]
        return elapsed_time
    
    def _format_time_auto_unit(self, total_seconds: float) -> str:
        """格式化时间单位"""
        if total_seconds < 0.000001:
            return f"{total_seconds * 1_000_000:.3f} 微秒"
        elif total_seconds < 0.001:
            return f"{total_seconds * 1_000_000:.3f} 微秒"
        elif total_seconds < 1.0:
            return f"{total_seconds * 1000:.3f} 毫秒"
        elif total_seconds < 60.0:
            return f"{total_seconds:.3f} 秒"
        elif total_seconds < 3600:
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            return f"{minutes} 分 {seconds:.3f} 秒"
        else:
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = total_seconds % 60
            return f"{hours} 小时 {minutes} 分 {seconds:.3f} 秒"
    
    def get_summary(self) -> dict:
        """
        获取性能测试摘要
        
        Returns:
            dict: 性能测试结果摘要
        """
        return self.results.copy()
    
    def clear_results(self) -> None:
        """清除所有结果"""
        self.results.clear()
        self.timers.clear()


def benchmark_function(func: Callable, iterations: int = 100, 
                      warmup_iterations: int = 10,
                      logger_instance: Optional[logging.Logger] = None) -> dict:
    """
    对函数进行基准测试
    
    Args:
        func: 要测试的函数
        iterations: 测试迭代次数
        warmup_iterations: 预热迭代次数
        logger_instance: 日志记录器实例
        
    Returns:
        dict: 基准测试结果
    """
    logger = logger_instance if logger_instance is not None else _default_logger
    
    # 预热
    logger.info(f"开始预热 {func.__name__}，预热次数: {warmup_iterations}")
    for _ in range(warmup_iterations):
        func()
    
    # 正式测试
    logger.info(f"开始基准测试 {func.__name__}，测试次数: {iterations}")
    times = []
    
    for i in range(iterations):
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # 计算统计信息
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # 计算标准差
    variance = sum((t - avg_time) ** 2 for t in times) / len(times)
    std_dev = variance ** 0.5
    
    results = {
        'function_name': func.__name__,
        'iterations': iterations,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_dev': std_dev,
        'total_time': sum(times)
    }
    
    # 格式化输出
    def format_time(seconds):
        if seconds < 0.000001:
            return f"{seconds * 1_000_000:.3f} 微秒"
        elif seconds < 0.001:
            return f"{seconds * 1_000_000:.3f} 微秒"
        elif seconds < 1.0:
            return f"{seconds * 1000:.3f} 毫秒"
        else:
            return f"{seconds:.3f} 秒"
    
    logger.info(f"基准测试结果 - {func.__name__}:")
    logger.info(f"  平均耗时: {format_time(avg_time)}")
    logger.info(f"  最小耗时: {format_time(min_time)}")
    logger.info(f"  最大耗时: {format_time(max_time)}")
    logger.info(f"  标准差: {format_time(std_dev)}")
    logger.info(f"  总耗时: {format_time(sum(times))}")
    
    return results


if __name__ == "__main__":
    # 测试代码
    try:
        from .logger import setup_logger
        from .path_utils import get_project_root

        # 设置日志
        project_root = get_project_root()
        logs_dir = project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger("performance_test", console_output=True, file_output=True)

    except ImportError:
        # 如果导入失败，使用基础日志配置
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("performance_test")

    # 测试time_it装饰器
    @time_it(iterations=5, name="测试函数", logger_instance=logger)
    def test_function():
        time.sleep(0.1)
        return "测试完成"

    # 测试性能分析器
    profiler = PerformanceProfiler(logger)

    def test_profiler():
        profiler.start_timer("测试操作")
        time.sleep(0.05)
        profiler.end_timer("测试操作")

    # 测试基准测试
    def simple_calculation():
        return sum(range(1000))

    logger.info("=" * 50)
    logger.info("性能工具测试开始")
    logger.info("=" * 50)

    # 运行测试
    result = test_function()
    logger.info(f"装饰器测试结果: {result}")

    test_profiler()

    benchmark_results = benchmark_function(simple_calculation, iterations=50, warmup_iterations=5, logger_instance=logger)

    logger.info("=" * 50)
    logger.info("性能工具测试完成")
    logger.info("=" * 50)
