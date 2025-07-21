"""文档检索模块

提供多种文档检索器实现：
- DocumentRetriever: 基础文档检索器
- EnhancedDocumentRetriever: 增强文档检索器（推荐）
"""

# from .document_retriever import DocumentRetriever  # 文件不存在，注释掉
from .enhanced_document_retriever import EnhancedDocumentRetriever

__all__ = [
    # 'DocumentRetriever',  # 文件不存在，已注释
    'EnhancedDocumentRetriever'
]