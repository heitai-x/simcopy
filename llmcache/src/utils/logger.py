#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志工具模块

提供统一的日志配置和管理功能。
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }
    
    def format(self, record):
        # 添加颜色
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class LoggerManager:
    """日志管理器"""
    
    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False
    _log_dir = None
    _log_level = logging.INFO
    _enable_file_logging = True
    _enable_console_logging = True
    _max_file_size = 10 * 1024 * 1024  # 10MB
    _backup_count = 5
    
    @classmethod
    def initialize(
        cls,
        log_dir: Optional[str] = None,
        log_level: str = "INFO",
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        max_file_size: int = 10 * 1024 * 1024,
        backup_count: int = 5
    ):
        """初始化日志管理器
        
        Args:
            log_dir: 日志目录
            log_level: 日志级别
            enable_file_logging: 是否启用文件日志
            enable_console_logging: 是否启用控制台日志
            max_file_size: 最大文件大小
            backup_count: 备份文件数量
        """
        if cls._initialized:
            return
        
        # 设置日志目录
        if log_dir is None:
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"
        
        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志级别
        cls._log_level = getattr(logging, log_level.upper(), logging.INFO)
        cls._enable_file_logging = enable_file_logging
        cls._enable_console_logging = enable_console_logging
        cls._max_file_size = max_file_size
        cls._backup_count = backup_count
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """获取日志器
        
        Args:
            name: 日志器名称
            
        Returns:
            logging.Logger: 配置好的日志器
        """
        # 确保已初始化
        if not cls._initialized:
            cls.initialize()
        
        # 如果已存在，直接返回
        if name in cls._loggers:
            return cls._loggers[name]
        
        # 创建新的日志器
        logger = logging.getLogger(name)
        logger.setLevel(cls._log_level)
        
        # 避免重复添加处理器
        if logger.handlers:
            logger.handlers.clear()
        
        # 创建格式化器
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        colored_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # 添加控制台处理器
        if cls._enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(cls._log_level)
            
            # 在支持颜色的终端使用彩色格式
            if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
                console_handler.setFormatter(colored_formatter)
            else:
                console_handler.setFormatter(simple_formatter)
            
            logger.addHandler(console_handler)
        
        # 添加文件处理器
        if cls._enable_file_logging:
            # 主日志文件
            main_log_file = cls._log_dir / f"{name}.log"
            file_handler = RotatingFileHandler(
                main_log_file,
                maxBytes=cls._max_file_size,
                backupCount=cls._backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(cls._log_level)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
            
            # 错误日志文件
            error_log_file = cls._log_dir / f"{name}_error.log"
            error_handler = RotatingFileHandler(
                error_log_file,
                maxBytes=cls._max_file_size,
                backupCount=cls._backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            logger.addHandler(error_handler)
        
        # 缓存日志器
        cls._loggers[name] = logger
        
        return logger
    
    @classmethod
    def set_level(cls, level: str):
        """设置所有日志器的级别
        
        Args:
            level: 日志级别
        """
        cls._log_level = getattr(logging, level.upper(), logging.INFO)
        
        for logger in cls._loggers.values():
            logger.setLevel(cls._log_level)
            for handler in logger.handlers:
                handler.setLevel(cls._log_level)
    
    @classmethod
    def get_log_files(cls) -> Dict[str, Path]:
        """获取所有日志文件路径
        
        Returns:
            Dict[str, Path]: 日志文件路径字典
        """
        if not cls._initialized:
            return {}
        
        log_files = {}
        for log_file in cls._log_dir.glob("*.log"):
            log_files[log_file.stem] = log_file
        
        return log_files
    
    @classmethod
    def cleanup_old_logs(cls, days: int = 7):
        """清理旧日志文件
        
        Args:
            days: 保留天数
        """
        if not cls._initialized:
            return
        
        import time
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        for log_file in cls._log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    print(f"删除旧日志文件: {log_file}")
                except Exception as e:
                    print(f"删除日志文件失败 {log_file}: {e}")
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """获取日志统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not cls._initialized:
            return {}
        
        stats = {
            "initialized": cls._initialized,
            "log_dir": str(cls._log_dir),
            "log_level": logging.getLevelName(cls._log_level),
            "loggers_count": len(cls._loggers),
            "logger_names": list(cls._loggers.keys()),
            "enable_file_logging": cls._enable_file_logging,
            "enable_console_logging": cls._enable_console_logging
        }
        
        # 统计日志文件
        if cls._log_dir.exists():
            log_files = list(cls._log_dir.glob("*.log*"))
            total_size = sum(f.stat().st_size for f in log_files if f.is_file())
            
            stats.update({
                "log_files_count": len(log_files),
                "total_log_size_mb": total_size / (1024 * 1024),
                "log_files": [f.name for f in log_files]
            })
        
        return stats


# 便捷函数
def get_logger(name: str = None) -> logging.Logger:
    """获取日志器的便捷函数
    
    Args:
        name: 日志器名称，如果为None则使用调用者的模块名
        
    Returns:
        logging.Logger: 配置好的日志器
    """
    if name is None:
        # 获取调用者的模块名
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return LoggerManager.get_logger(name)


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    enable_file_logging: bool = True,
    enable_console_logging: bool = True
):
    """设置日志系统的便捷函数
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别
        enable_file_logging: 是否启用文件日志
        enable_console_logging: 是否启用控制台日志
    """
    LoggerManager.initialize(
        log_dir=log_dir,
        log_level=log_level,
        enable_file_logging=enable_file_logging,
        enable_console_logging=enable_console_logging
    )


def set_log_level(level: str):
    """设置日志级别的便捷函数
    
    Args:
        level: 日志级别
    """
    LoggerManager.set_level(level)


def get_log_stats() -> Dict[str, Any]:
    """获取日志统计信息的便捷函数
    
    Returns:
        Dict[str, Any]: 统计信息
    """
    return LoggerManager.get_stats()


# 自动初始化
if not LoggerManager._initialized:
    # 从环境变量读取配置
    log_level = os.getenv('VLLM_LOG_LEVEL', 'INFO')
    log_dir = os.getenv('VLLM_LOG_DIR')
    
    LoggerManager.initialize(
        log_dir=log_dir,
        log_level=log_level
    )