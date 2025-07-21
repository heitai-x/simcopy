"""高级异步连接词提取器

结合了异步设计和高级NLP功能的连接词提取器，专为高并发实时场景设计。
主要特性：
1. 异步并发控制：支持多任务并发处理，适合实时响应
2. 高级NLP处理：句法分析（benepar）、层序遍历
3. 智能连接词识别：支持多种连接词类型和层级分析
4. 性能优化：懒加载、预编译正则表达式
5. 实时处理：专注于单个请求的快速响应，无批量处理开销
"""

import asyncio
import logging
import os
import re
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional

import spacy
import benepar
import pandas as pd
import nltk

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 修改NLTK数据路径为相对路径或环境变量
nltk_data_path = os.getenv('NLTK_DATA', './nltk_data')
if os.path.exists(nltk_data_path):
    nltk.data.path.append(nltk_data_path)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncAdvancedConjunctionExtractor:
    """高级异步连接词提取器"""
    
    # 常量定义
    MAX_TOKEN_LIMIT = 500
    CONJUNCTIONS = ('and', 'or', 'but', ',', ';')
    INITIAL_LEVELS = 5
    GROWTH_FACTOR = 5
    SKIP_LABELS = frozenset({'PP', 'SBAR'})
    
    def __init__(self, max_concurrent_tasks: int = 10, typo_dict_path: str = "llmcache/data/typo_dict.tsv"):
        """初始化异步连接词提取器
        
        Args:
            max_concurrent_tasks: 最大并发任务数
            typo_dict_path: 拼写错误字典文件路径
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._current_tasks = 0
        
        # NLP模型
        self.nlp = None
        self.sent_nlp = None
        self._initialized = False
        
        # 预编译正则表达式
        self._compile_patterns()
        
        # 拼写错误字典路径
        self.typo_dict_path = typo_dict_path
        self.typo_dict = {}
        self.typo_pattern = re.compile('')
        
        # 预计算连接词集合
        self.conjunctions_set = frozenset(self.CONJUNCTIONS)
        self.initialize()
            
        logger.info(f"异步高级连接词提取器创建成功，最大并发: {max_concurrent_tasks}")
    
    def _compile_patterns(self) -> None:
        """预编译所有正则表达式模式"""
        self.conj_pattern = re.compile(r'\b(?:and|or|but)\b|[,;]', re.IGNORECASE)
        self.two_end_pattern = re.compile(r'^[/\]]|[/?`.:;>]+$')
        

    def initialize(self) -> None:
        """同步初始化NLP模型和字典"""
        try:
            logger.info("开始同步初始化NLP模型...")
            
            # 直接加载模型（不使用线程池）
            self.nlp = self._create_main_nlp_model()
            self.sent_nlp = self._create_sentence_nlp_model()
            
            # 加载拼写错误字典
            self._load_typo_dict_sync()
            
            self._initialized = True
            logger.info("同步高级连接词提取器初始化成功")
            
        except Exception as e:
            logger.error(f"初始化同步连接词提取器失败: {e}")
            self._initialized = False
            raise
    
    def _create_main_nlp_model(self):
        """创建主要的NLP模型"""
        nlp = spacy.load(
            'en_core_web_sm', 
            exclude=['lemmatizer', 'ner', 'parser', 'tagger', 'attribute_ruler']
        )
        nlp.add_pipe('sentencizer', first=True)
        
        # 添加benepar组件
        if spacy.__version__.startswith('2'):
            nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            nlp.add_pipe("benepar", config={"model": "benepar_en3"})
        
        return nlp
    
    def _create_sentence_nlp_model(self):
        """创建句子分割模型"""
        sent_nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner', 'tagger'])
        sent_nlp.add_pipe('sentencizer', first=True)
        return sent_nlp
    
    def _load_typo_dict_sync(self) -> None:
        """同步加载拼写错误字典"""
        try:
            if not os.path.exists(self.typo_dict_path):
                logger.warning(f"拼写错误字典未找到: {self.typo_dict_path}")
                return
            
            # 直接加载字典文件（移除异步线程池）
            typo_df = pd.read_csv(self.typo_dict_path, sep='\t', header=None)
            
            # 构建字典
            typo_dict = {}
            for _, row in typo_df.iterrows():
                original, corrected = row[0], row[1]
                typo_dict[original.lower()] = corrected.lower()
                typo_dict[original.capitalize()] = corrected.capitalize()
            
            self.typo_dict = typo_dict
            
            if typo_dict:
                self.typo_pattern = re.compile('|'.join(map(re.escape, typo_dict.keys())))
            
            logger.info(f"拼写错误字典加载成功，包含 {len(typo_dict)} 个条目")
            
        except Exception as e:
            logger.warning(f"加载拼写错误字典失败: {e}")
        
    async def cleanup(self) -> None:
        """清理资源"""
        try:
            # 等待所有任务完成
            while self._current_tasks > 0:
                await asyncio.sleep(0.1)
            
            self.nlp = None
            self.sent_nlp = None
            self._initialized = False
            logger.info("异步高级连接词提取器已清理")
            
        except Exception as e:
            logger.error(f"清理异步连接词提取器失败: {e}")
    
    def set_max_concurrent_tasks(self, max_concurrent: int) -> None:
        """设置最大并发任务数"""
        self.max_concurrent_tasks = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f"最大并发任务数已设置为: {max_concurrent}")
    
    async def extract_and_generate_variants(self, text: str, max_variants: int = 3) -> Optional[Dict[str, Any]]:
        """异步提取连接词并生成变体
        
        Args:
            text: 输入文本
            max_variants: 最大变体数量
            
        Returns:
            包含原文、连接词和变体的字典
        """
        if not self._initialized:
            logger.warning("提取器未初始化")
            return None
        
        async with self._semaphore:
            self._current_tasks += 1
            try:
                return await self._process_text_advanced(text, max_variants)
            finally:
                self._current_tasks -= 1
    
    async def _process_text_advanced(self, text: str, max_variants: int) -> Dict[str, Any]:
        """高级文本处理逻辑"""
        try:
            # 快速预筛选
            if not self.conj_pattern.search(text):
                return {
                    'original_text': text,
                    'conjunctions': [],
                    'subsentences': [],
                    'has_conjunctions': False
                }
            
            # 长度检查
            if len(text) > self.MAX_TOKEN_LIMIT * 5:
                logger.warning(f"文本过长，跳过处理: {len(text)} 字符")
                return {
                    'original_text': text,
                    'conjunctions': [],
                    'subsentences': [],
                    'has_conjunctions': False
                }
            
            # 在线程池中进行NLP处理（避免阻塞）
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = await loop.run_in_executor(
                    executor, 
                    self._extract_internal_sync, 
                    text
                )
            
            if result and result.get('variants'):
                return {
                    'original_text': text,
                    'conjunctions': result.get('conjunctions', []),
                    'subsentences': result.get('variants', []),
                    'has_conjunctions': len(result.get('conjunctions', [])) > 0
                }
            else:
                # 使用简单的正则表达式提取
                conjunctions = await self._extract_conjunctions_regex(text)
                
                return {
                    'original_text': text,
                    'conjunctions': conjunctions,
                    'subsentences': [],
                    'has_conjunctions': len(conjunctions) > 0
                }
                
        except Exception as e:
            logger.error(f"处理文本失败: {e}")
            return {
                'original_text': text,
                'conjunctions': [],
                'subsentences': [],
                'has_conjunctions': False
            }
    
    def _extract_internal_sync(self, text: str) -> Optional[Dict[str, Any]]:
        """同步的内部提取逻辑（在线程池中运行）"""
        try:
            sents = list(self.sent_nlp(text).sents)
            
            if len(sents) > 1:
                return self._process_multi_sentences(sents, None)
            else:
                return self._process_single_sentence_wrapper(text, None)
        except Exception as e:
            logger.warning(f"同步提取失败: {e}")
            return None
    
    def _process_multi_sentences(self, sents: List, id: Optional[str]) -> Optional[Dict[str, Any]]:
        """处理多句子文本"""
        for i, sent in enumerate(sents):
            sent_text = sent.text
            
            if self.conj_pattern.search(sent_text):
                try:
                    doc = self.nlp(sent_text)
                    result = self._process_single_sentence(doc, id)
                    if result and result.get('conjunctions') and result.get('subsentences'):
                        variants = self._build_multi_sentence_variants(sents, i, result)
                        if variants:
                            return {
                                'id': id,
                                'original': ' '.join(s.text for s in sents),
                                'variants': variants,
                                'conjunctions': result.get('conjunctions', [])
                            }
                except Exception as e:
                    logger.warning(f"处理句子 {i} 失败: {e}")
                    continue
        
        return None
    
    def _process_single_sentence_wrapper(self, sentence: str, id: Optional[str]) -> Optional[Dict[str, Any]]:
        """处理单句子的包装方法"""
        try:
            doc = self.nlp(sentence)
            result = self._process_single_sentence(doc, id)
            
            if result and result.get('subsentences'):
                variants = []
                for sub_list in result['subsentences'].values():
                    variants.extend([sub['text'] for sub in sub_list])
                
                if variants:
                    return {
                        'id': id,
                        'original': sentence,
                        'variants': variants,
                        'conjunctions': result.get('conjunctions', [])
                    }
        except Exception as e:
            logger.warning(f"处理单句子失败: {e}")
        
        return None
    
    def _build_multi_sentence_variants(self, sents: List, target_index: int, result: Dict) -> List[str]:
        """构建多句子的变体"""
        variants = []
        subsentences = []
        
        for sub_list in result['subsentences'].values():
            subsentences.extend([sub['text'] for sub in sub_list])
        
        if subsentences:
            modified_sents = [s.text for s in sents]
            for sub in subsentences:
                modified_sents[target_index] = sub
                variants.append(' '.join(modified_sents))
        
        return variants
    
    def _process_single_sentence(self, doc, id: Optional[str]) -> Optional[Dict[str, Any]]:
        """处理单个句子的核心逻辑"""
        conjunctions = []
        subsentences = {}
        
        for sent in doc.sents:
            if not hasattr(sent._, 'constituents'):
                continue
            
            level_nodes = self._level_order_traversal(sent)
            sent_conjunctions = self._find_conjunctions(level_nodes)
            
            if sent_conjunctions:
                conjunctions.extend([
                    c['coordinators'][0] for c in sent_conjunctions 
                    if c.get('coordinators')
                ])
                
                outer_conjunctions = self._find_outer_conjunctions(sent_conjunctions)
                if outer_conjunctions:
                    subsentences.update(self._generate_subsentences(sent.text, outer_conjunctions))
        
        if conjunctions:
            return {
                'conjunctions': conjunctions,
                'subsentences': subsentences
            }
        
        return None
    
    def _level_order_traversal(self, sent) -> List[List[Dict]]:
        """对句法树进行层序遍历"""
        level_nodes = [[] for _ in range(self.INITIAL_LEVELS)]
        max_level = 0
        queue = deque([(sent, 0)])
        
        while queue:
            current_node, current_level = queue.popleft()
            
            if current_level > max_level:
                max_level = current_level

            if current_level >= len(level_nodes):
                level_nodes.extend([[] for _ in range(self.GROWTH_FACTOR)])

            node_underscore = current_node._
            
            if hasattr(node_underscore, 'labels'):
                labels = node_underscore.labels
                if labels and labels[0] in self.SKIP_LABELS:
                    continue
                node_label = labels[0] if labels else None
            else:
                node_label = None

            node_info = {
                'node': current_node,
                'label': node_label,
                'text': current_node.text,
                'start': current_node.start_char,
                'end': current_node.end_char
            }
            level_nodes[current_level].append(node_info)
            
            if hasattr(node_underscore, 'children') and node_underscore.children:
                next_level = current_level + 1
                queue.extend((child, next_level) for child in node_underscore.children)
        
        return level_nodes[:max_level + 1]
    
    def _find_conjunctions(self, level_nodes: List[List[Dict]]) -> List[Dict]:
        """在句法树的各层级中查找连接词模式"""
        conjunctions = []
        level_count = len(level_nodes)
        
        for level in range(level_count - 1, -1, -1):
            nodes = level_nodes[level]
            nodes_len = len(nodes)
            
            if nodes_len <= 1:
                continue
            
            parent_map = self._build_parent_map(level_nodes, level)
            conjunctions.extend(self._find_conjunctions_in_level(nodes, parent_map, level))
        
        return conjunctions
    
    def _build_parent_map(self, level_nodes: List[List[Dict]], level: int) -> Dict[int, Dict]:
        """构建节点的父子关系映射"""
        parent_map = {}
        
        if level > 0:
            parent_children_map = {}
            
            for parent_idx, parent in enumerate(level_nodes[level-1]):
                parent_node_underscore = parent['node']._
                if hasattr(parent_node_underscore, 'children'):
                    parent_children_map[parent_idx] = set(parent_node_underscore.children)
            
            if parent_children_map:
                for node_idx, node in enumerate(level_nodes[level]):
                    node_obj = node['node']
                    for parent_idx, children in parent_children_map.items():
                        if node_obj in children:
                            parent_map[node_idx] = level_nodes[level-1][parent_idx]
                            break
        
        return parent_map
    
    def _find_conjunctions_in_level(self, nodes: List[Dict], parent_map: Dict[int, Dict], level: int) -> List[Dict]:
        """在特定层级中查找连接词"""
        conjunctions = []
        nodes_len = len(nodes)
        i = 0
        
        while i < nodes_len:
            node = nodes[i]
            node_obj = node['node']
            
            if not isinstance(node_obj, spacy.tokens.Span):
                i += 1
                continue
                
            node_text = node_obj.text.strip()
            if node_text not in self.conjunctions_set or i == 0:
                i += 1
                continue
            
            conjunction = self._extract_conjunction_pattern(nodes, i, parent_map, level)
            if conjunction:
                conjunctions.append(conjunction)
                i = conjunction.get('next_index', i + 1)
            else:
                i += 1
        
        return conjunctions
    
    def _extract_conjunction_pattern(self, nodes: List[Dict], conj_index: int, parent_map: Dict[int, Dict], level: int) -> Optional[Dict]:
        """提取连接词模式"""
        left_phrase = nodes[conj_index - 1]
        left_idx = conj_index - 1
        left_parent = parent_map.get(left_idx)
        left_label = left_phrase['label']
        
        phrases = [left_phrase]
        coordinators = []
        positions = [[(left_phrase['start'], left_phrase['end'])]]
        
        j = conj_index
        nodes_len = len(nodes)
        
        while j < nodes_len:
            curr_node = nodes[j]
            curr_obj = curr_node['node']
            
            if isinstance(curr_obj, spacy.tokens.Span):
                curr_text = curr_obj.text.strip()
                if curr_text in self.conjunctions_set:
                    coordinators.append(curr_text)
                    j += 1
                    continue
            
            if (curr_node['label'] == left_label and
                (left_parent is None or parent_map.get(j) == left_parent)):
                phrases.append(curr_node)
                positions.append([(curr_node['start'], curr_node['end'])])
                j += 1
            else:
                break
        
        if len(phrases) > 1:
            return {
                'label': f"{left_label}_{level}",
                'phrases': [p['text'] for p in phrases],
                'coordinators': coordinators,
                'positions': positions,
                'next_index': j
            }
        
        return None
    
    def _find_outer_conjunctions(self, conjunctions: List[Dict]) -> List[Dict]:
        """从找到的连接词中选择最外层的"""
        if not conjunctions:
            return []

        levels = defaultdict(list)
        
        for conj in conjunctions:
            label_parts = conj['label'].split('_')
            if len(label_parts) > 1:
                try:
                    level = int(label_parts[-1])
                    levels[level].append(conj)
                except ValueError:
                    logger.warning(f"无效的层级标签: {conj['label']}")
                    continue
        
        if not levels:
            return []
        
        min_level = min(levels.keys())
        return levels[min_level]
    
    def _generate_subsentences(self, original_text: str, conjunctions: List[Dict]) -> Dict[str, List[Dict]]:
        """根据连接词生成子句"""
        subsentences = {}
        
        for conj in conjunctions:
            label = conj['label']
            phrases = conj['phrases']
            positions = conj['positions']
            coordinators = conj['coordinators']
            
            all_positions = [pos for pos_list in positions for pos in pos_list]
            min_pos = min(pos[0] for pos in all_positions)
            max_pos = max(pos[1] for pos in all_positions)
            
            prefix = original_text[:min_pos]
            suffix = original_text[max_pos:]
            
            sub_sentences = []
            for phrase in phrases:
                subsentence = prefix + phrase + suffix
                sub_info = {
                    'text': subsentence,
                    'coordinator': coordinators,
                    'label': label
                }
                sub_sentences.append(sub_info)
            
            subsentences[label] = sub_sentences
        
        return subsentences
    
    async def _extract_conjunctions_regex(self, text: str) -> List[str]:
        """使用正则表达式提取连接词"""
        try:
            conjunctions = []
            
            # 查找所有连接词
            matches = self.conj_pattern.finditer(text)
            for match in matches:
                conj = match.group().strip()
                if conj and conj not in conjunctions:
                    conjunctions.append(conj)
            
            return list(conjunctions)
            
        except Exception as e:
            logger.error(f"正则表达式提取连接词失败: {e}")
            return []
    

    
    def clean_question(self, question: str, two_end: bool = True, typo: bool = False) -> str:
        """清理问题文本
        
        Args:
            question: 输入问题
            two_end: 是否清理首尾特殊字符
            typo: 是否修正拼写错误
            
        Returns:
            清理后的文本
        """
        if not question:
            return ''
        
        question = question.strip()
        
        if two_end:
            question = self.two_end_pattern.sub('', question)
        
        if typo and self.typo_dict:
            question = question.replace(" 's", "'s")
            question = self.typo_pattern.sub(lambda m: self.typo_dict[m.group()], question)
        
        return ' '.join(question.split())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'initialized': self._initialized,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'current_tasks': self._current_tasks,
            'available_permits': self._semaphore._value if self._semaphore else 0,
            'has_nlp_model': self.nlp is not None,
            'has_sent_nlp_model': self.sent_nlp is not None,
            'typo_dict_size': len(self.typo_dict)
        }
    
    def is_ready(self) -> bool:
        """检查是否准备就绪"""
        return self._initialized and self.nlp is not None and self.sent_nlp is not None


if __name__ == "__main__":
    async def main():
        print("开始异步连接词提取测试...")
        start_time = time.time()
        
        # 创建提取器
        extractor = AsyncAdvancedConjunctionExtractor(max_concurrent_tasks=5)
        
        # 初始化
        await extractor.initialize()
        print(f"初始化完成，耗时: {time.time() - start_time:.2f} 秒")
        
        # 测试单个文本
        test_text = "How can I check the used and total space of each GPU on Ubuntu?"
        begin_time = time.time()
        result = await extractor.extract_and_generate_variants(test_text)
        print(f"单个提取耗时: {time.time() - begin_time:.4f} 秒")
        print(f"结果: {result}")
        
        # 测试多个文本的单独处理（实时场景）
        test_texts = [
            "I like apples and oranges",
            "She runs fast but he walks slowly",
            "We can go to the park or stay at home"
        ]
        
        print("\n测试实时处理多个文本:")
        for i, text in enumerate(test_texts):
            single_start = time.time()
            result = await extractor.extract_and_generate_variants(text)
            single_time = time.time() - single_start
            print("result:",result)
            print(f"文本 {i+1} 处理耗时: {single_time:.4f} 秒 - {result['has_conjunctions'] if result else False}")
            if result:
                print(f"  变体数量: {len(result['enhanced_variants'])}")
        
        # 获取统计信息
        stats = extractor.get_stats()
        print(f"\n统计信息: {stats}")
        
        # 清理资源
        await extractor.cleanup()
        print("测试完成")
    
    # 运行异步主函数
    asyncio.run(main())