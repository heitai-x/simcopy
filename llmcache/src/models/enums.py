"""系统枚举定义"""

from enum import Enum


class RequestStatus(Enum):
    """请求状态枚举"""
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_FOR_DUPLICATE = "waiting_for_duplicate"


class CacheStatus(Enum):
    """缓存状态枚举"""
    PROCESSING = "processing"
    COMPLETED = "completed"


class CacheLevel(Enum):
    """缓存级别枚举"""
    MEMORY = "memory"
    REDIS = "redis"
    NONE = "none"


class Priority(Enum):
    """请求优先级枚举"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3