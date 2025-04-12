import os
# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import re
from functools import lru_cache
import spacy
import benepar
from transformers import RobertaTokenizer
import pandas as pd
import tqdm
import time

# 全局模型初始化
nlp = None
sent_nlp = None


def initialize_models():
    global nlp, sent_nlp

    nlp = spacy.load('en_core_web_trf', exclude=['lemmatizer', 'ner', 'parser', 'tagger', 'attribute_ruler'])
    nlp.add_pipe('sentencizer', first=True)
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3_large"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3_large"})
    if sent_nlp is None:
        sent_nlp = spacy.load('en_core_web_sm')
        sent_nlp.add_pipe('sentencizer', first=True)

initialize_models()

# 加载拼写错误字典
typo_df = pd.read_csv(r"D:\桌面\typo_dict.tsv", sep='\t', header=None)
typo_dict_lower = {row[0][0].lower() + row[0][1:]: row[1][0].lower() + row[1][1:] for _, row in typo_df.iterrows()}
typo_dict_capital = {row[0][0].upper() + row[0][1:]: row[1][0].upper() + row[1][1:] for _, row in typo_df.iterrows()}
typo_dict = dict(typo_dict_capital, **typo_dict_lower)

from collections import defaultdict, deque

class ConjunctionExtractor:
    MAX_TOKEN_LIMIT = 512
    CONJUNCTIONS = (',', '/', ';', 'and', 'or', 'but')  # 新增连接词规定
    VS = ["VB", "VBG", "VBD", "VBN", "VBP", "VBZ"]
    NS = ["NN", "NNS", "NNP", "NNPS"]
    def is_vs(self,node):
        if node['label'] in self.VS:
            return True
    def is_ns(self,node):
        if node['label'] in self.NS:
            return True

    def __init__(self):
        self.nlp = nlp
        self.sent_nlp = sent_nlp
        # 预编译正则表达式
        self.two_end_pattern = re.compile(r'^[/\]]|[/?`.:;>]+$')
        self.typo_pattern = re.compile('|'.join(map(re.escape, typo_dict.keys())))

    def extract(self, sentence=None, doc=None, id=None):

        if doc is None:
            if not sentence:
                return None
            if not any(conj in sentence for conj in self.CONJUNCTIONS):
                return None            
            sub_sentences = [sent.text for sent in self.sent_nlp(sentence).sents]
            
            all_results = []
            for sub_sent in sub_sentences:
                doc= self.nlp(sub_sent)
                # sent = list(doc.sents)[0] 
                # print(sent._.parse_string)
                if not any(conj in sub_sent for conj in self.CONJUNCTIONS):
                    continue
                try:
                    result = self._process_sub_sentence_text(doc, id)
                    if result:
                        all_results.extend(result)
                except Exception as e:
                    print(f"处理子句时发生错误: {str(e)}")
            
            if all_results:
                return self._generate_output(sentence, all_results, self.sent_nlp(sentence), id)
            return None

        # 如果已经有doc，直接处理
        return self._process_single_sentence(doc, id)

    def _process_sub_sentence_text(self, doc, id):
        """处理单个子句的文本"""
        return self._process_single_sentence(doc, id)

    def _process_single_sentence(self, doc, id):
        """处理单个句子"""
        conjunctions = []
        for sent in doc.sents:
            if not hasattr(sent._, 'constituents'):
                continue
            
            level_nodes = self._level_order_traversal(sent)
            sent_conjunctions = self._find_conjunctions(level_nodes)
            if sent_conjunctions:
                conjunctions.extend(sent_conjunctions)
                
        return conjunctions if conjunctions else None

    def _generate_output(self, text, conjunctions, sent, id):
        """生成输出结果，包含原始句子和ID信息"""
        # 找出最外层的并列结构
        outer_conjunctions = self._find_outer_conjunctions(conjunctions)
        
        # 为最外层并列结构生成子句
        subsentences = self._generate_subsentences(text, outer_conjunctions) if outer_conjunctions else {}
        
        return {
            'id': id,
            'original': text,
            'placeholder': sent.text,
            'conjunctions': {c['label']: c['phrases'] for c in conjunctions},
            'coordinators': {c['label']: c['coordinators'] for c in conjunctions},
            'positions': {c['label']: c['positions'] for c in conjunctions},
            'labels': [c['label'] for c in conjunctions],
            'subsentences': subsentences  # 添加子句到输出
        }
    
    def _find_outer_conjunctions(self, conjunctions):
        """找出最外层的并列结构（通常是层级数字最小的）"""
        if not conjunctions:
            return []
        
        # 提取每个并列结构的层级
        levels = {}
        for conj in conjunctions:
            label_parts = conj['label'].split('_')
            if len(label_parts) > 1:
                try:
                    level = int(label_parts[-1])
                    if level not in levels:
                        levels[level] = []
                    levels[level].append(conj)
                except ValueError:
                    continue
        
        # 如果没有有效的层级，返回空列表
        if not levels:
            return []
        
        # 返回最小层级的并列结构
        min_level = min(levels.keys())
        return levels[min_level]
    
    def _generate_subsentences(self, original_text, conjunctions):
        """为最外层并列结构生成子句"""
        subsentences = {}
        
        for conj in conjunctions:
            label = conj['label']
            phrases = conj['phrases']
            positions = conj['positions']
            coordinators = conj['coordinators']
            
            # 只处理有多个短语的并列结构
            if len(phrases) <= 1 or len(positions) != len(phrases):
                continue
            
            # 为每个短语生成子句
            sub_sentences = []
            for i, (phrase, pos_list) in enumerate(zip(phrases, positions)):
                # 创建原始文本的副本
                subsentence = list(original_text)  # 转换为字符列表以便于修改
                
                # 获取当前短语的位置
                if not pos_list:
                    continue
                
                # 记录需要删除的区间
                to_delete = []
                
                # 收集所有需要删除的区间（除了当前短语）
                for j, other_pos_list in enumerate(positions):
                    if j == i:  # 跳过当前短语
                        continue
                    
                    # 添加其他短语的位置到删除列表
                    for start, end in other_pos_list:
                        to_delete.append((start, end))
                
                # 如果有连接词，也需要删除
                for j, coord in enumerate(coordinators):
                    # 找到连接词在原文中的位置
                    if j < len(phrases) - 1:
                        # 连接词通常在两个短语之间
                        if j < i:  # 当前短语之前的连接词
                            # 找到连接词的位置（在前一个短语结束和后一个短语开始之间）
                            prev_end = positions[j][-1][1] if positions[j] else 0
                            next_start = positions[j+1][0][0] if positions[j+1] else len(original_text)
                            
                            # 在这个范围内查找连接词
                            coord_pos = original_text.find(coord, prev_end, next_start)
                            if coord_pos != -1:
                                to_delete.append((coord_pos, coord_pos + len(coord)))
                        elif j >= i:  # 当前短语之后的连接词
                            # 同样找到连接词位置
                            prev_end = positions[j][-1][1] if positions[j] else 0
                            next_start = positions[j+1][0][0] if j+1 < len(positions) and positions[j+1] else len(original_text)
                            
                            coord_pos = original_text.find(coord, prev_end, next_start)
                            if coord_pos != -1:
                                to_delete.append((coord_pos, coord_pos + len(coord)))
                
                # 合并重叠的删除区间
                if to_delete:
                    to_delete.sort()  # 按开始位置排序
                    merged = [to_delete[0]]
                    for start, end in to_delete[1:]:
                        prev_start, prev_end = merged[-1]
                        if start <= prev_end:  # 有重叠
                            merged[-1] = (prev_start, max(prev_end, end))
                        else:
                            merged.append((start, end))
                    to_delete = merged
                
                # 从后向前删除，避免位置偏移
                for start, end in sorted(to_delete, reverse=True):
                    subsentence[start:end] = []
                
                # 转回字符串
                subsentence = ''.join(subsentence)
                
                # 清理多余的连接词和空格
                subsentence = self._clean_subsentence(subsentence)
                
                # 存储子句信息，包括连接词和标签
                sub_info = {
                    'text': subsentence,
                    'phrase': phrase,
                    'position': pos_list,
                    'coordinator': coordinators[i-1] if i > 0 and i-1 < len(coordinators) else None,
                    'label': label
                }
                sub_sentences.append(sub_info)
            
            subsentences[label] = sub_sentences
        
        return subsentences
    
    def _clean_subsentence(self, sentence):
        """清理子句中的多余连接词和空格"""
        # 清理连续的连接词
        for conj in self.CONJUNCTIONS:
            # 清理句首的连接词
            if sentence.startswith(conj):
                sentence = sentence[len(conj):].lstrip()
            # 清理句尾的连接词
            if sentence.endswith(conj):
                sentence = sentence[:-len(conj)].rstrip()
            # 清理连续的连接词
            pattern = f" {re.escape(conj)} {re.escape(conj)} "
            sentence = re.sub(pattern, f" {conj} ", sentence)
        
        # 清理多余空格和标点
        sentence = ' '.join(sentence.split())
        # 确保句子有正确的结束标点

        
        return sentence.strip()

    def clean_question(self, question, two_end=True, typo=False):
        """清理句子，使用正则表达式优化"""
        if not question:
            return ''
        question = question.strip()
        if two_end:
            question = self.two_end_pattern.sub('', question)
        if typo:
            question = question.replace(" 's", "'s")
            question = self.typo_pattern.sub(lambda m: typo_dict[m.group()], question)
        return ' '.join(question.split())

    def process_json(self, input_json_path, output_json_path):
        """处理 JSON 文件中的句子"""
        buffer = []
        processed_count = 0
        success_count = 0
        
        # 检查是否有已存在的输出文件，如果有则加载
        if os.path.exists(output_json_path):
            try:
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    buffer = json.load(f)
                    # 获取已处理的ID列表
                    processed_ids = {item['id'] for item in buffer}
                    success_count = len(buffer)
                    print(f"已加载 {success_count} 条已处理数据")
            except Exception as e:
                print(f"加载已有数据时出错: {str(e)}")
                buffer = []
        else:
            processed_ids = set()
        
        # 加载输入数据
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_items = len(data)
        print(f"总数据量: {total_items}，已处理: {success_count}")
        
        # 创建进度条
        with tqdm.tqdm(total=total_items, desc='处理进度', initial=success_count) as pbar:
            start_time = time.time()
            save_interval = 100  # 每处理100条数据保存一次
            
            for item in data:
                # 跳过已处理的数据
                if item['id'] in processed_ids:
                    continue
                
                sentence = item['sentences'][-1]
                id = item['id']
                cleaned_sentence = self.clean_question(sentence)
                
                try:
                    result = self.extract(cleaned_sentence, id=id)
                    if result:
                        buffer.append(result)
                        success_count += 1
                    
                    # 更新进度
                    processed_count += 1
                    pbar.update(1)
                    
                    # 定期保存结果
                    if processed_count % save_interval == 0:
                        self.save_to_json(buffer, output_json_path)
                        elapsed_time = time.time() - start_time
                        avg_time = elapsed_time / processed_count
                        remaining = (total_items - success_count - processed_count) * avg_time
                        print(f"\n已处理: {processed_count}，成功: {success_count}，平均时间: {avg_time:.2f}秒/条")
                        print(f"预计剩余时间: {remaining/60:.2f}分钟")
                
                except Exception as e:
                    print(f"\n处理ID {id}时出错: {str(e)}")
                    pbar.update(1)
            
            # 最终保存
            self.save_to_json(buffer, output_json_path)
            
        # 打印最终统计信息
        print(f"处理完成! 总数据: {total_items}, 成功处理: {success_count}, 成功率: {success_count/total_items*100:.2f}%")
        return buffer
    def save_to_json(self, buffer, json_file_path):
        """增量追加 JSON 数据，只保存关键信息"""
        # 提取关键信息
        simplified_buffer = []
        for item in buffer:
            simplified_item = {
                'id': item['id'],
                'original': item['original'],
                'subsentences': item.get('subsentences', {})
            }
            simplified_buffer.append(simplified_item)
        
        # 检查文件是否存在
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    # 使用ID去重
                    id_map = {item['id']: item for item in existing_data}
                    for item in simplified_buffer:
                        id_map[item['id']] = item
                    # 转换回列表
                    simplified_buffer = list(id_map.values())
                except json.JSONDecodeError:
                    print("文件格式错误，将覆盖原文件")
        
        # 写入简化后的数据
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_buffer, f, ensure_ascii=False, indent=4)
        
        print(f"已保存 {len(simplified_buffer)} 条简化记录到 {json_file_path}")

    def _level_order_traversal(self, sent):
        """保证兼容性的改进版本，跳过介词短语和从句节点"""
        level_nodes = []
        queue = deque([(sent, 0)])
        
        while queue:
            current_node, current_level = queue.popleft()
            
            # 检查节点类型，如果是PP(介词短语)或SBAR(从句)则跳过
            if hasattr(current_node._, 'labels') and current_node._.labels:
                node_label = current_node._.labels[0]
                if node_label in ['PP', 'SBAR']:  # 跳过介词短语和从句节点
                    continue
                    
            # 确保层级容器存在
            while current_level >= len(level_nodes):
                level_nodes.append([])
                
            # 生成节点信息
            node_info = {
                'node': current_node,
                'label': current_node._.labels[0] if hasattr(current_node._, 'labels') 
                        and current_node._.labels else None,
                'text': current_node.text,
                'start': current_node.start_char,
                'end': current_node.end_char
            }
            level_nodes[current_level].append(node_info)
            
            # 获取子节点
            children = current_node._.children if hasattr(current_node._, 'children') else []
            for child in children:
                queue.append((child, current_level + 1))
        
        return level_nodes


    def _find_conjunctions(self, level_nodes):
        """从后向前遍历每层寻找并列结构，以连接词为核心"""
        conjunctions = []
        
        for level in reversed(range(len(level_nodes))):
            nodes = level_nodes[level]
            i = 0
            
            # 为每个节点添加父节点信息
            parent_map = {}
            if level > 0 and level < len(level_nodes):
                for node_idx, node in enumerate(nodes):
                    # 寻找父节点
                    for parent in level_nodes[level-1]:
                        if (hasattr(parent['node']._, 'children') and 
                            node['node'] in parent['node']._.children):
                            parent_map[node_idx] = parent
                            break
            
            while i < len(nodes):
                node = nodes[i]
                
                if not (isinstance(node['node'], spacy.tokens.Span) and 
                       node['node'].text.strip() in self.CONJUNCTIONS):
                    i += 1
                    continue
                
                # 向前寻找第一个短语
                left_phrase = None
                left_idx = None
                if i > 0:
                    left_phrase = nodes[i-1]
                    left_idx = i-1
                
                if not left_phrase:
                    i += 1
                    continue
                
                # 获取左侧短语的父节点
                left_parent = parent_map.get(left_idx)
                
                # 收集所有并列成分
                phrases = [left_phrase]
                coordinators = []
                positions = [[(left_phrase['start'], left_phrase['end'])]]
                
                # 从当前连接词开始，收集所有连接词和短语
                j = i
                while j < len(nodes):
                    # 当前位置是连接词
                    if (isinstance(nodes[j]['node'], spacy.tokens.Span) and 
                        nodes[j]['node'].text.strip() in self.CONJUNCTIONS):
                        coordinators.append(nodes[j]['node'].text.strip())
                        j += 1
                    # 连接词后面是短语，且与左侧短语标签相同，并且共享同一个父节点
                    elif (j < len(nodes) and 
                          left_phrase['label'] == nodes[j]['label'] and
                          (left_parent is None or parent_map.get(j) == left_parent)):
                        phrases.append(nodes[j])
                        positions.append([(nodes[j]['start'], nodes[j]['end'])])
                        j += 1
                    else:
                        break
                
                # 如果找到了多个短语和连接词，创建并列结构
                if len(phrases) > 1:
                    phrase_label = left_phrase['label'] if left_phrase['label'] else 'PHRASE'
                    conjunction = {
                        'label': f"{phrase_label}_{level}",
                        'phrases': [p['text'] for p in phrases],
                        'coordinators': coordinators,
                        'positions': positions
                    }
                    conjunctions.append(conjunction)
                    i = j  # 跳过已处理的节点
                else:
                    i += 1
        
        return conjunctions

    def extract_dependencies(self, sentence):
        """输出句子的依存关系分析结果"""
        if not sentence:
            return None
            
        doc = self.nlp(sentence)
        dependencies = []
        
        for token in doc:
            dependencies.append({
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_,
                'head': token.head.text,
                'head_pos': token.head.pos_,
                'children': [child.text for child in token.children]
            })
            
        return {
            'sentence': sentence,
            'dependencies': dependencies
        }

if __name__ == "__main__":
    extractor = ConjunctionExtractor()
    try:
        input_path = r"num2_sim_fails.json"
        output_path = r"data\num2_sim_fails_coors.json"
        
        # 处理数据
        extractor.process_json(input_path, output_path)
        
        # 新增测试依存关系分析
        example = "Boat are full with people on a boat"
        deps = extractor.extract_dependencies(example)
        print(json.dumps(deps, indent=2))
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")