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

from collections import defaultdict, deque
import nltk
nltk.data.path.append('/root/autodl-tmp/nltk_data')
# benepar.download('benepar_en3_large')

class ConjunctionExtractor:
    MAX_TOKEN_LIMIT = 500
    CONJUNCTIONS = ( 'and', 'or', 'but')  


    def __init__(self):
        # 初始化NLP模型
        self.initialize_models()
        # 加载拼写错误字典
        typo_df = pd.read_csv("llma/src/tree/typo_dict.tsv", sep='\t', header=None)
        typo_dict_lower = {row[0][0].lower() + row[0][1:]: row[1][0].lower() + row[1][1:] for _, row in typo_df.iterrows()}
        typo_dict_capital = {row[0][0].upper() + row[0][1:]: row[1][0].upper() + row[1][1:] for _, row in typo_df.iterrows()}
        self.typo_dict = dict(typo_dict_capital, **typo_dict_lower)
        # 预编译正则表达式
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.two_end_pattern = re.compile(r'^[/\]]|[/?`.:;>]+$')
        self.typo_pattern = re.compile('|'.join(map(re.escape, self.typo_dict.keys())))
        
    def initialize_models(self):
        self.nlp = spacy.load('en_core_web_sm', exclude=['lemmatizer', 'ner', 'parser', 'tagger', 'attribute_ruler'])
        self.nlp.add_pipe('sentencizer', first=True)
        if spacy.__version__.startswith('2'):
            self.nlp.add_pipe(benepar.BeneparComponent("benepar_en3_large"))
        else:
            self.nlp.add_pipe("benepar", config={"model": "benepar_en3_large"})
        self.sent_nlp = spacy.load('en_core_web_sm')
        self.sent_nlp.add_pipe('sentencizer', first=True)


    def extract(self, sentence=None, doc=None, id=None):
        if doc is None:
            if not sentence:
                return None
            # 使用正则表达式直接匹配连接词
            conj_pattern = re.compile(r'\b(and|or|but)\b|[,;]', re.IGNORECASE)
            if not conj_pattern.search(sentence):
                return None
                
            sents = list(self.sent_nlp(sentence).sents)
            if len(sents) > 1:  # 长文本处理
                results = []
                found_conjunction = False
                result = None  
                for i, sent in enumerate(sents):
                    sent_text = sent.text

                    if not found_conjunction and conj_pattern.search(sent_text):
                        try:
                            doc = self.nlp(sent_text)
                        except Exception as e:
                            continue
                        result = self._process_single_sentence(doc, id)
                        
                        if result and result.get('conjunctions') and result.get('subsentences'):
                            subsentences = []
                            for sub_list in result['subsentences'].values():
                                subsentences.extend([sub['text'] for sub in sub_list])
                            
                            if subsentences:
                                variants = []
                                modified_sents = [s.text for s in sents]
                                for sub in subsentences:
                                    modified_sents[i] = sub
                                    variants.append(' '.join(modified_sents))
                                
                                results = variants
                                found_conjunction = True
                            break
                
                if not found_conjunction:
                    return None
                
                if results:
                    return {
                        'id': id,
                        'original': sentence,
                        'variants': results,
                        'conjunctions': result.get('conjunctions', []) if result else []
                    }
            else: 
                
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
                            'conjunctions': result.get('conjunctions', []) if result else []
                        }
        return None

    def _process_sub_sentence_text(self, doc, id):
        return self._process_single_sentence(doc, id)

    def _process_single_sentence(self, doc, id):
        conjunctions = []
        subsentences = {} 
        
        for sent in doc.sents:
            if not hasattr(sent._, 'constituents'):
                continue
            
            level_nodes = self._level_order_traversal(sent)
            sent_conjunctions = self._find_conjunctions(level_nodes)
            if sent_conjunctions:
                # 只提取连接词文本
                conjunctions.extend([c['coordinators'][0] for c in sent_conjunctions if c.get('coordinators')])
                outer_conjunctions = self._find_outer_conjunctions(sent_conjunctions)
                if outer_conjunctions:
                    subsentences.update(self._generate_subsentences(sent.text, outer_conjunctions))
        
        if conjunctions:
            return {
                'conjunctions': conjunctions,  # 现在只包含连接词文本列表
                'subsentences': subsentences 
            }
        return None

    def _find_outer_conjunctions(self, conjunctions):
        if not conjunctions:
            return []

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
        if not levels:
            return []
        min_level = min(levels.keys())
        return levels[min_level]
    def _generate_subsentences(self, original_text, conjunctions):
        subsentences = {}
        for conj in conjunctions:
            label = conj['label']
            phrases = conj['phrases']
            positions = conj['positions']
            coordinators = conj['coordinators']
            sub_sentences = []
            all_positions = [pos for pos_list in positions for pos in pos_list]
            min_pos = min(pos[0] for pos in all_positions) 
            max_pos = max(pos[1] for pos in all_positions)  
            prefix = original_text[:min_pos]
            suffix=original_text[max_pos:]
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
    

    def clean_question(self, question, two_end=True, typo=False):
        if not question:
            return ''
        question = question.strip()
        if two_end:
            question = self.two_end_pattern.sub('', question)
        if typo:
            question = question.replace(" 's", "'s")
            question = self.typo_pattern.sub(lambda m: self.typo_dict[m.group()], question)
        return ' '.join(question.split())
    def _level_order_traversal(self, sent):
        initial_levels = 5
        growth_factor = 5
        level_nodes = [[] for _ in range(initial_levels)]

        skip_labels = {'PP', 'SBAR'}
        max_level = 0

        queue = deque([(sent, 0)])
        
        while queue:
            current_node, current_level = queue.popleft()
            
            if current_level > max_level:
                max_level = current_level

            if current_level >= len(level_nodes):
                level_nodes.extend([] for _ in range(growth_factor))

            node_underscore = current_node._

            has_labels = hasattr(node_underscore, 'labels')
            if has_labels:
                labels = node_underscore.labels
                if labels and labels[0] in skip_labels:
                    continue
                node_label = labels[0] if labels else None
            else:
                node_label = None
            start_char = current_node.start_char
            end_char = current_node.end_char

            node_info = {
                'node': current_node,
                'label': node_label,
                'text': current_node.text,
                'start': start_char,
                'end': end_char
            }
            level_nodes[current_level].append(node_info)
            
            if hasattr(node_underscore, 'children') and node_underscore.children:
                children = node_underscore.children
                next_level = current_level + 1
                queue.extend((child, next_level) for child in children)
        
        return level_nodes[:max_level + 1]


    def _find_conjunctions(self, level_nodes):
        conjunctions = []
        conjunctions_set = set(self.CONJUNCTIONS)
        
        level_count = len(level_nodes)
        
        for level in range(level_count - 1, -1, -1):  
            nodes = level_nodes[level]
            nodes_len = len(nodes)
            if nodes_len <= 1:  
                continue
                
            parent_map = {}
            
            if level > 0:
                parent_children_map = {}
                for parent_idx, parent in enumerate(level_nodes[level-1]):
                    parent_node_underscore = parent['node']._
                    if hasattr(parent_node_underscore, 'children'):
                        parent_children_map[parent_idx] = set(parent_node_underscore.children)
            
                if parent_children_map:
                    for node_idx, node in enumerate(nodes):
                        node_obj = node['node']
                        for parent_idx, children in parent_children_map.items():
                            if node_obj in children:
                                parent_map[node_idx] = level_nodes[level-1][parent_idx]
                                break
            
            i = 0
            while i < nodes_len:
                node = nodes[i]
                node_obj = node['node']
                
                is_span = isinstance(node_obj, spacy.tokens.Span)
                if not is_span:
                    i += 1
                    continue
                    
                node_text = node_obj.text.strip()

                if node_text not in conjunctions_set:
                    i += 1
                    continue
                if i == 0:
                    i += 1
                    continue
                    
                left_phrase = nodes[i-1]
                left_idx = i-1
                left_parent = parent_map.get(left_idx)
                left_label = left_phrase['label']
                phrases = [left_phrase]
                coordinators = []
                positions = [[(left_phrase['start'], left_phrase['end'])]]

                j = i
                while j < nodes_len:
                    curr_node = nodes[j]
                    curr_obj = curr_node['node']

                    curr_is_span = isinstance(curr_obj, spacy.tokens.Span)

                    if curr_is_span:
                        curr_text = curr_obj.text.strip()
                        if curr_text in conjunctions_set:
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
                    phrase_label = left_label
                    conjunction = {
                        'label': f"{phrase_label}_{level}",
                        'phrases': [p['text'] for p in phrases],
                        'coordinators': coordinators,
                        'positions': positions
                    }
                    conjunctions.append(conjunction)
                    i = j 
                else:
                    i += 1

        return conjunctions



if __name__ == "__main__":
    print("Starting extraction...")
    start_time = time.time()
    extractor = ConjunctionExtractor()
    print("Initialization completed.")
    print(f"Initialization time: {time.time() - start_time} seconds")
    begin_time = time.time()
    text="write a story that includes these words \"blubber, bruit, portentous, kilter, brusque, chaffed, cluck, immaculate, dulcet, asperous, clop, prodigious, equine, blase, lapis lazuli, chenille, jive, brunt, cochineal, foothill, chiffonier, pleistocene, fatuous, skittles, soliloquy, rushes, brash, cleft, forfeiture, rickety, Bacchic, chenille, premonition, portend, Dionysus, dishevel, déshabillé, Bacchanal, equipage, dissolute, brunt, marzipan, chiffon, palaeontology, blob, pallor, cadence, bruit, frayed, effrontery, peonies, propagate, schism, quadrupeds, pepsin, insinuating, expostulating, hauberk, sagittarius, Faustian, nihilism, congeries, penury, ghoulish, vaunted, satyr, bedew, claustrophobic, polypeptides, Salves, snigger, antics, lather, hooch, Pyrrhic, peccadilloes, pith, puritanical, anchorite, inhibitions, insets, onyx, gentry, veronal, fret, potage, felicitous, incarceration, pauper, bauble, fop, sycophant, obsequious, Agate, Barbiturates, effervescent, jig, carefree, troupe, mimicry, Invective, proprietorial, smudge, undulating, gibbous, filial, portico, drew nigh, spilt, pestilence, peon, hanger-on, cloying, redolent, pander, portly, wheeze, gramophone, vacuity, preposterous, heaved, button-downs, sculling, mitten, denarius, encirclement, regurgitate, ashram, heathen\".\n"
    result=extractor.extract(text)
    print(f"Extraction time: {time.time() - begin_time} seconds")
    print(result)
    # try:
    #     input_path = "New Folder/llma/sample_data/output.json"
    #     output_path = "New Folder/llma/sample_data/coors.json"

    #     extractor.process_json(input_path, output_path)
        
        
    # except Exception as e:
    #     print(f"处理过程中发生错误: {str(e)}")