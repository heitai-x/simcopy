"""缓存管理模块"""

from .redis_manager import RedisManager
from .enhanced_vector_search import EnhancedVectorSearchManager
from .multi_level_cache import MultiLevelCache

try:
    from ..utils.similarity_search_helper import SimilaritySearchHelper
except ImportError:
    # similarity_search_helper might not exist yet
    SimilaritySearchHelper = None

# Create an alias for backward compatibility
MultiLevelCacheManager = MultiLevelCache

__all__ = [
    "RedisManager",
    "EnhancedVectorSearchManager", 
    "MultiLevelCache",
    "MultiLevelCacheManager",  # Alias for compatibility
    "SimilaritySearchHelper"
]