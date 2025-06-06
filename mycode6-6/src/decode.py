import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
cache_dir = "/mnt/sevenT/debinx/huggingface_models"
import torch.cuda
import time
import json
import re
import os
from collections import defaultdict
from sentence_transformers import CrossEncoder
from transformers import Qwen2ForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch
import json
from collections import defaultdict
from tqdm import tqdm
from retriever import DocumentRetriever
from sentence_transformers import SentenceTransformer
import time
import argparse
import spacy
import numpy as np
from tree import ConjunctionExtractor
from transformers.cache_utils import DynamicCache

# 修改spacy初始化部分
nlp_model = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp_model.add_pipe("sentencizer") 

def position_my(query, draft_ids, matcheds, position_ids, begin_idx, draft_all=None):
    # 进一步优化版本：减少内存分配、复制操作和计算量
    # draft_ids为当前草稿
    # query为正常推理时的输入token
    # position_ids为已经处理过的位置信息，根据当前草稿进行更新
    # begin_idx为kvcache长度
    # draft_all为上一步处理完的输入
    # matcheds存储已处理过的matched_ids和matched_pos，用于前缀搜索
    
    # 快速获取关键维度信息
    query_len = query.shape[-1]  # 一般为1
    draft_len = len(draft_ids)  # 当前草稿长度
    draft_all_len = draft_all.shape[-1] if draft_all is not None else 0  # 上一步处理完的输入长度
    start = begin_idx + query_len
    device = query.device
    
    # 预分配空列表，避免后续重新分配
    prefix_pos = []
    draft_new_pos = []
    
    # 查找最长匹配前缀 - 优化循环结构
    if matcheds:
        max_prefix_len = 0
        best_matched_pos = None
        
        for matched_ids, matched_pos, _ in matcheds:
            # 计算可比较的最小长度
            compare_len = min(draft_len, len(matched_ids))
            if compare_len == 0:
                continue
                
            # 使用更高效的方式查找匹配前缀长度
            prefix_len = 0
            for i in range(compare_len):
                if draft_ids[i] != matched_ids[i]:
                    break
                prefix_len = i + 1
            
            # 只在找到更长前缀时更新
            if prefix_len > max_prefix_len:
                max_prefix_len = prefix_len
                if prefix_len > 0 and prefix_len <= len(matched_pos):
                    best_matched_pos = matched_pos[:prefix_len]
        
        # 只在找到匹配时设置prefix_pos
        if best_matched_pos is not None:
            prefix_pos = best_matched_pos
    
    pre_len = len(prefix_pos)
    
    # 高效创建position_ids张量 - 避免不必要的计算
    if draft_len > pre_len:
        new_positions = torch.arange(start+pre_len, start+draft_len, device=device)
        position_ids.append(new_positions)
    
    # 合并position_ids - 只在有内容时执行
    if position_ids:
        position_ids = torch.cat(position_ids).unsqueeze(0)
    else:
        position_ids = torch.tensor([], device=device).unsqueeze(0)

    # 只在需要时创建新的张量 - 避免不必要的计算
    if pre_len < draft_len:
        # 优化张量创建
        if isinstance(draft_ids, torch.Tensor):
            remaining_ids = draft_ids[pre_len:].unsqueeze(0)
            if remaining_ids.device != device:
                remaining_ids = remaining_ids.to(device)
        else:
            remaining_ids = torch.tensor([draft_ids[pre_len:]], device=device)
        
        # 高效地更新draft_all - 避免不必要的连接操作
        if draft_all is None:
            draft_all = remaining_ids
        else:
            draft_all = torch.cat([draft_all, remaining_ids], dim=1)
        
        # 预计算偏移量 - 使用列表推导式一次性创建
        offset = 1 + draft_all_len - pre_len
        draft_new_pos = [offset + i for i in range(pre_len, draft_len)]

    # 高效地创建new_matched_ids - 避免不必要的转换
    if isinstance(draft_ids, list):
        new_matched_ids = draft_ids.copy()
    else:
        new_matched_ids = draft_ids.tolist()
    
    # 合并位置信息
    new_matched_pos = prefix_pos + draft_new_pos
    matcheds.append((new_matched_ids, new_matched_pos, draft_new_pos))
        
    return draft_all, matcheds, position_ids
    
def mybuild_parallel_attention_mask(
    q_length,
    kv_length,
    copy_length,
    dtype,
    device,
    matcheds,
    batch_size=1
):
    # 优化版本：减少内存分配和张量操作
    total_length = q_length + copy_length
    min_dtype = torch.finfo(dtype).min
    
    # 预分配掩码张量，避免多次分配内存
    causal_mask = torch.full(
        (total_length, total_length), fill_value=min_dtype, dtype=dtype, device=device
    )
    
    # 使用in-place操作设置下三角部分
    if q_length > 0:
        # 只创建一次零张量和min_dtype张量
        zero_tensor = torch.tensor(0.0, dtype=dtype, device=device)
        min_tensor = torch.tensor(min_dtype, dtype=dtype, device=device)
        
        # 使用预计算的下三角矩阵
        causal_indices = torch.tril(torch.ones((q_length, total_length), device=device))
        # 将前q_length行的下三角部分设置为0（可见）
        causal_mask[:q_length, :total_length] = torch.where(
            causal_indices > 0,
            zero_tensor,
            min_tensor
        )
    
    # 使用matched_pos和draft_new_pos进行掩码处理
    if matcheds and len(matcheds) > 0:
        # 批量处理所有匹配信息，减少循环次数
        all_rows = []
        all_cols = []
        
        for _, matched_pos, draft_new_pos in matcheds:
            if not draft_new_pos or not matched_pos:
                continue
                
            # 直接使用列表推导式，避免额外的列表创建
            if len(draft_new_pos) > 0 and len(matched_pos) > 0:
                # 预计算索引数组大小
                row_indices = []
                col_indices = []
                
                # 使用嵌套循环代替repeat_interleave和repeat，减少内存使用
                for row in draft_new_pos:
                    for col in matched_pos:
                        row_indices.append(row)
                        col_indices.append(col)
                
                all_rows.extend(row_indices)
                all_cols.extend(col_indices)
        
        # 只在有匹配时才创建张量和设置掩码
        if all_rows:
            # 一次性创建索引张量并设置掩码
            causal_mask[all_rows, all_cols] = 0.0
    
    # 创建kv部分的掩码并连接
    kv_mask = torch.zeros((total_length, kv_length), dtype=dtype, device=device)
    causal_mask = torch.cat([kv_mask, causal_mask], dim=-1)
    
    # 扩展到批次维度
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, total_length, total_length+kv_length)
    
    return causal_mask


def is_complete_sentence(text):
    """使用 NLP 解析句子，判断是否完整"""
    doc = nlp_model.nlp(text)
    if not doc:
        return False
    last_token = doc[-1]
    return last_token.is_punct


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--type", type=str, default="llma")
    parser.add_argument("--input_data_fn", type=str, default="llma/sample_data/output_sample.json")
    parser.add_argument("--forced_decoding", action="store_true")
    parser.add_argument("--retriever_metadata", type=str, help="检索器元数据路径", default="llma/src/faiss_dataset/metadata/metadata.json")
    parser.add_argument("--retriever_index", type=str, help="检索器索引路径", default="llma/src/faiss_dataset/metadata/vector_index_cosine.faiss")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    return args
def get_ngrams(tokens, n):
    ngram_list = []
    for i in range(len(tokens)-n+1):
        ngram = ' '.join(tokens[i:i+n])
        ngram_list.append(ngram)
    return ngram_list


def match_length(seq_a, seq_b):
    l = 0
    for i in range(min(len(seq_a), len(seq_b))):
        if seq_a[i] != seq_b[i]:
            break
        l += 1
    return l

def prepare_ngrams(s ,n,tokenizer, max_n=5):

    if max_n < n:
        max_n = n
    docs = s["similar_answer"]
    doc_list = [tokenizer.tokenize(x) for x in docs]
    # if generate_ids is not None:
    #     doc_token_id_list=generate_ids
    # else:
    
    doc_token_id_list = [tokenizer.convert_tokens_to_ids(x) for x in doc_list]
    
    per_doc_ngrams_list = []
    for doc_idx, doc_tokens in enumerate(doc_list):
        doc_ngrams_list = []
        # 为每个长度生成n-grams
        for l in range(n, max_n+1):
            doc_ngrams = defaultdict(list)
            ngram_list = get_ngrams(doc_tokens, l)
            for pos, ngram in enumerate(ngram_list):
                doc_ngrams[ngram].append((doc_idx, pos))  # 存储文档索引和位置
            doc_ngrams_list.append([l, doc_ngrams])
        doc_ngrams_list.reverse()  # 长的n-grams优先
        per_doc_ngrams_list.append(doc_ngrams_list)

    return { "doc_ngrams": per_doc_ngrams_list,"doc_token_id_list": doc_token_id_list}

def get_tokenizer_and_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    

    model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16, 
        device_map="auto" ,          
        cache_dir=cache_dir,       

    )
    
    return tokenizer, model

def truncate(doc, tokenizer, max_tokens=1024):

    if max_tokens <= 0:
        return doc
        
    tokens = tokenizer.tokenize(doc)[:max_tokens] 
    
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    doc = tokenizer.decode(token_ids)
    
    return doc

def load_data(input_fn, tokenizer, retriever=None, model=None,ConjunctionExtractor=None):
    s_list = []
    gen_texts_ids=None
    forced_decoding=False
    
    # 导入多线程模块
    import threading
    
    with open(input_fn, 'r',encoding='utf-8') as file:
        data = json.load(file)
        i=0
        for s in data:
            i+=1
            print(i)    
            if retriever is not None:
                results = {"sub_queries": None, "similar_docs": None}
                def extract_sub_queries():
                    results["sub_queries"] = ConjunctionExtractor.extract(sentence=s['query'][0], doc=None, id=s['id'])
                def search_similar_docs():
                    results["similar_docs"] = retriever.search(s['query'], s['id'], k=3)
                extract_sub_queries()
                search_similar_docs()
                similar_docs = results["similar_docs"][0]
                
                s['high_similarity'] = False
                s['similar_answer'] = []
                s['sim_query'] = []
                high_sim_found = False
                
                for similar_doc in similar_docs:
                    print(similar_doc)
                    if similar_doc['similarity'] >= 0.95:
                        prompt = similar_doc['text']
                        messages = [
                            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                            {"role": "user", "content": prompt}]
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        inputs = tokenizer(text, return_tensors="pt")
                        generate_ids,token_num = base_generate(model, tokenizer, inputs.input_ids, gen_texts_ids,
                                    forced_decoding=forced_decoding,past_key_values=None)
                        generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        s['similar_answer'].append(generated)
                        s['high_similarity'] = True
                        s['sim_query'].append(similar_doc['text'])
                        high_sim_found = True
                        break
                if not high_sim_found:
                    if  results["sub_queries"] is not None:
                        sub_queries = results["sub_queries"]['variants']
                        all_similar_docs = []
                        seen_texts = set()
                        for similar_doc in similar_docs:
                            if similar_doc['similarity']>=0.8 and similar_doc['text'] not in seen_texts:
                                all_similar_docs.append(similar_doc)
                                seen_texts.add(similar_doc['text'])
                        if sub_queries:
                            for sub_query in sub_queries:
                                sub_similar_doc = retriever.search([sub_query], s['id'], k=2)
                                if sub_similar_doc != []:
                                    if sub_similar_doc[0]['similarity'] >= 0.8 and sub_similar_doc[0]['text'] not in seen_texts:
                                        all_similar_docs.append(sub_similar_doc[0])
                                        seen_texts.add(sub_similar_doc[0]['text'])
                    else:
                        all_similar_docs = similar_docs
                    for similar_doc in all_similar_docs:
                        prompt = similar_doc['text']
                        messages = [
                            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                            {"role": "user", "content": prompt}]
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        inputs = tokenizer(text, return_tensors="pt")
                        generate_ids,token_num = base_generate(model, tokenizer, inputs.input_ids, gen_texts_ids,
                            forced_decoding=forced_decoding,past_key_values=None)
                        generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        s['similar_answer'].append(generated)    
                        s['sim_query'].append(similar_doc['text'])
                s_list.append(s)    
            else:
                s_list.append(s)
    return s_list

        # for doc in docs:
        #     doc_ids=tokenizer(doc, add_special_tokens=False,return_tensors="pt")['input_ids'].tolist()[0]
        #     print(doc_ids)
        #     chunks = []
        #     for i in range(0, len(doc_ids), max_tokens):
        #         chunk = doc_ids[i:i+max_tokens]
        #         chunks.append(chunk)
        #     for chunk in chunks:
        #             # 将token ids转换回文本
        #         text = tokenizer.decode(chunk, skip_special_tokens=True)
        #         sim_ans_sentences.append(text)
        #         sim_ans_token_id_lists.append(chunk)
                    
        #     if embedding_model is not None:
        #         embeddings = embedding_model.encode(
        #             sim_ans_sentences,
        #             batch_size=8,
        #             convert_to_tensor=True,
        #             normalize_embeddings=True
        #         )
        #         sim_embeddings.append(embeddings.cpu().numpy())            

def match_prefix(g_ngrams_list,per_doc_ngrams_list, step):

    matches = []
    num = []
    for doc_ngrams_list in per_doc_ngrams_list:
        
        for g_ngrams, doc_ngrams in zip(g_ngrams_list, doc_ngrams_list):
            n = g_ngrams[0]
            g_ngrams = g_ngrams[1]
            doc_ngrams = doc_ngrams[1]
            if step < n:
                continue

            if g_ngrams[step-n] in doc_ngrams.keys():
                matches.append(doc_ngrams[g_ngrams[step-n]])
                num.append(n)
                break
    if matches:
        return num, matches
    
    return [], []

def _compute_match_matrix(tokenizer, probs, prepend_tensor, output_tensor, matcheds, max_candidate_len):
    """计算匹配矩阵,判断每个候选序列的匹配程度 - 优化版本"""
    # 预分配内存并使用设备内存
    device = prepend_tensor.device
    match_matrix = torch.zeros((len(matcheds), max_candidate_len), 
                            dtype=torch.bool, device=device)
    
    # 预先计算EOS token ID
    eos_token_id = tokenizer.eos_token_id
    
    # 阈值常量
    threshold = 0.8
    
    # 批量处理所有候选序列
    for i, (matched_ids, matched_pos, draft_new_pos) in enumerate(matcheds):
        if not matched_pos:
            continue
            
        # 预先获取所有需要的位置索引
        out_positions = [matched_pos[pre_idx-1] if pre_idx > 0 else 0 for pre_idx in range(len(matched_pos))]
        
        # 批量获取当前token和概率
        for pre_idx, (pos, out_pos) in enumerate(zip(matched_pos, out_positions)):
            # 提前检查索引边界
            if pos >= len(prepend_tensor) or out_pos >= probs.shape[1]:
                break
                
            # 获取当前token和概率
            current_token = prepend_tensor[pos].item()
            
            # 检查索引是否有效
            if current_token >= probs.shape[2]:
                break
                
            current_token_prob = probs[0, out_pos, current_token].item()
            
            # 使用torch.max的返回值直接获取最大概率
            max_prob = torch.max(probs[0, out_pos]).item()
            
            # 检查前一个token是否是EOS
            prev_token_id = output_tensor[out_pos].item() if out_pos < len(output_tensor) else -1
            
            # 设置匹配结果
            is_match = (current_token_prob >= max_prob * threshold and prev_token_id != eos_token_id)
            match_matrix[i, pre_idx] = is_match
            
            # 提前终止不匹配的序列
            if not is_match:
                break
                
    return match_matrix

def _get_best_candidate(candidate_match_lengths, matcheds, prepend_tensor, output_tensor):
    """根据匹配矩阵选择最佳候选序列 - 优化版本"""
    # 使用numpy操作找到最大匹配长度和索引
    max_match_length = max(candidate_match_lengths)
    best_candidate_idx = candidate_match_lengths.index(max_match_length)
    
    # 获取最佳匹配位置
    real_copy_matched_pos = matcheds[best_candidate_idx][1]
    
    # 预分配结果数组
    real_output_ids = [output_tensor[0].item()]
    
    # 一次性获取所有需要的token ID
    if max_match_length > 0:
        # 更新第一个token
        if len(real_copy_matched_pos) > 0:
            real_output_ids[0] = prepend_tensor[real_copy_matched_pos[0]].item()
        
        # 添加剩余的token
        for j in range(1, max_match_length):
            if j < len(real_copy_matched_pos) and real_copy_matched_pos[j] < len(prepend_tensor):
                pos = real_copy_matched_pos[j]
                if pos < len(output_tensor):
                    real_output_ids.append(output_tensor[pos].item())
    
    return real_output_ids, best_candidate_idx
    
def make_past_key_values(past_key_values, step_length, accepted_step_length, real_copy_matched_pos):
    # 优化版本：减少内存分配和张量操作
    
    # 快速路径：如果只接受一个token，直接裁剪
    kv_length = past_key_values.get_seq_length()
    if accepted_step_length == 1:
        past_key_values.crop(kv_length - step_length + accepted_step_length)
        return past_key_values
    
    # 只保留需要的位置索引
    if accepted_step_length > 1 and len(real_copy_matched_pos) >= accepted_step_length - 1:
        real_copy_matched_pos = real_copy_matched_pos[:accepted_step_length-1]
    elif len(real_copy_matched_pos) == 0:
        # 如果没有匹配位置，直接裁剪并返回
        past_key_values.crop(kv_length - step_length + accepted_step_length)
        return past_key_values
    
    # 更新已见token数量
    past_key_values._seen_tokens = kv_length - step_length + accepted_step_length
    
    # 计算旧缓存结束位置和复制索引
    old_end = kv_length - step_length + 1
    
    # 一次性创建索引张量，避免在循环中重复创建
    device = None
    for cache in past_key_values.key_cache:
        if cache is not None and cache.numel() > 0:
            device = cache.device
            break
    
    if device is None:
        return past_key_values
    
    # 预计算复制索引
    copy_indices = torch.tensor([kv_length - step_length + idx for idx in real_copy_matched_pos], 
                               device=device)
    
    # 批量处理所有缓存层
    for idx, (key_cache, value_cache) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
        if key_cache is None or key_cache.numel() == 0:
            continue
            
        # 使用视图操作而不是复制，减少内存使用
        old_key_cache = key_cache[..., :old_end, :]
        old_value_cache = value_cache[..., :old_end, :]
        
        # 使用预计算的索引进行选择
        copy_key_cache = torch.index_select(key_cache, -2, copy_indices)
        copy_value_cache = torch.index_select(value_cache, -2, copy_indices)
        
        # 使用torch.cat合并缓存
        past_key_values.key_cache[idx] = torch.cat([old_key_cache, copy_key_cache], dim=-2)
        past_key_values.value_cache[idx] = torch.cat([old_value_cache, copy_value_cache], dim=-2)
    
    return past_key_values

def copy_generate(model, tokenizer, input_ids, trigger_N=1, block_K=20, min_block_K=2,
                     forced_decoding=False, ngrams_cache=None, max_new_tokens=1024,
                     past_key_values=DynamicCache(), min_candidate_num=5, max_n=5):
        # 优化版本：减少内存分配和计算量
        past_key_values = DynamicCache()
        prepend_ids = input_ids.cuda()
        device = prepend_ids.device
        dtype = model.dtype
        
        # 初始化计数器和状态变量
        llma_num = 0
        generate_ids = None
        generate_ids_num = 0
        step = 0
        number = 0
        
        # 获取文档n-gram和token列表
        doc_ngrams_list = ngrams_cache["doc_ngrams"]
        doc_token_id_list = ngrams_cache["doc_token_id_list"]

        # 初始化n-gram列表和状态标志
        gtokens = []
        g_ngrams_list = [(n, []) for n in range(max_n, 0, -1)]
        modify_mode = True
        copy_mode = False
        n_mode = False
        
        # 预分配文档token数量数组
        doc_token_num = [min_candidate_num] * len(doc_token_id_list)
        
        # 获取EOS token ID，避免重复查询
        eos_token_id = tokenizer.eos_token_id
        
        while True:
            # 重置每次迭代的变量
            position_ids = None
            attention_mask = None
            candidate_lens = []
            context_ids = []
            matcheds = []
            begin_idx = past_key_values.get_seq_length() if past_key_values is not None else 0
            ori_ids = prepend_ids
            
            # 处理多token输入的情况
            if prepend_ids.shape[-1] > 1:
                used = set()
                number += 1
                draft_all = None
                postion_ids = [torch.arange(begin_idx, begin_idx+prepend_ids.shape[-1], device=device)]
                
                # 批量处理所有文档token
                for i, doc_token_ids in enumerate(doc_token_id_list):
                    # 获取当前文档的token序列
                    sent = doc_token_ids[:doc_token_num[i]]
                    sent_tuple = tuple(sent)
                    
                    # 避免重复处理相同的序列
                    if sent_tuple not in used:
                        used.add(sent_tuple)
                        
                        # 更新draft_all和位置信息
                        draft_all, matcheds, position_ids = position_my(
                            prepend_ids,
                            sent,
                            matcheds,
                            postion_ids,
                            begin_idx=begin_idx,
                            draft_all=draft_all
                        )
                        
                        # 记录上下文ID和候选长度
                        context_ids.append(i)
                        candidate_lens.append(len(sent))
                
                # 计算步长和注意力掩码
                if draft_all is not None:
                    step_length = 1 + draft_all.shape[-1]
                    attention_mask = mybuild_parallel_attention_mask(
                        prepend_ids.shape[-1],
                        begin_idx,
                        draft_all.shape[-1],
                        dtype=dtype,
                        device=device,
                        matcheds=matcheds
                    )
                    
                    # 连接输入和草稿
                    prepend_ids = torch.cat([prepend_ids, draft_all], dim=-1)
                    copy_mode = True
                else:
                    step_length = 1
                    copy_mode = False
            else:
                step_length = 1
                copy_mode = False
            
            # 处理非复制模式
            if not copy_mode:
                prefix_n, matches = match_prefix(g_ngrams_list, doc_ngrams_list, step)
                
                # 如果找到前缀匹配
                if prefix_n:
                    draft_all = None
                    number += 1
                    used = set()
                    postion_ids = [torch.arange(begin_idx, begin_idx+prepend_ids.shape[-1], device=device)]
                    
                    # 批量处理所有匹配
                    for i, match in enumerate(matches):
                        n = prefix_n[i]
                        for mat in match:
                            doc_idx, pos = mat
                            
                            # 获取匹配的token序列
                            matched_ids = doc_token_id_list[doc_idx][pos+n:pos+n+doc_token_num[doc_idx]]
                            matched_tuple = tuple(matched_ids)
                            
                            # 避免重复处理相同的序列
                            if matched_tuple not in used:
                                used.add(matched_tuple)
                                
                                # 记录上下文ID和候选长度
                                context_ids.append(doc_idx)
                                candidate_lens.append(len(matched_ids))
                                
                                # 更新draft_all和位置信息
                                draft_all, matcheds, position_ids = position_my(
                                    prepend_ids,
                                    matched_ids,
                                    matcheds,
                                    postion_ids,
                                    begin_idx=begin_idx,
                                    draft_all=draft_all
                                )
                    
                    # 计算步长和注意力掩码
                    if draft_all is not None:
                        step_length = 1 + draft_all.shape[-1]
                        attention_mask = mybuild_parallel_attention_mask(
                            prepend_ids.shape[-1],
                            begin_idx,
                            draft_all.shape[-1],
                            dtype=dtype,
                            device=device,
                            matcheds=matcheds
                        )
                        
                        # 连接输入和草稿
                        prepend_ids = torch.cat([prepend_ids, draft_all], dim=-1)
                        n_mode = True
                    else:
                        step_length = 1
                        n_mode = False
                else:
                    step_length = 1
                    n_mode = False
            # 使用torch.no_grad()减少内存使用
            with torch.no_grad():
                # 模型推理
                output = model(
                    input_ids=prepend_ids,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    position_ids=position_ids,
                    attention_mask=attention_mask
                )
                
                # 获取最后step_length个位置的logits
                logits = output['logits'][:, -step_length:, :]
                
                # 获取最可能的token ID
                output_ids = torch.argmax(logits, dim=-1)
                accepted_step_length = step_length
                
                # 只保留最后step_length个token
                prepend_ids = prepend_ids[:, -step_length:]
                past_key_values = output.past_key_values
                
                # 处理复制模式或n-gram模式
                if copy_mode or n_mode:
                    # 计算token概率
                    probs = F.softmax(logits, dim=-1)
                    prepend_tensor = prepend_ids[0]
                    output_tensor = output_ids[0]
                    
                    # 计算最大候选长度
                    max_candidate_len = max(candidate_lens) if candidate_lens else 0
                    
                    # 计算匹配矩阵
                    match_matrix = _compute_match_matrix(
                        tokenizer,
                        probs, 
                        prepend_tensor, 
                        output_tensor, 
                        matcheds, 
                        max_candidate_len
                    )
                    
                    # 计算每个候选序列的匹配长度
                    candidate_match_lengths = match_matrix.sum(dim=1).cpu().tolist()
                    
                    # 获取最佳候选序列
                    real_output_ids, best_candidate_idx = _get_best_candidate(
                        candidate_match_lengths, 
                        matcheds,
                        prepend_tensor, 
                        output_tensor
                    )
                    
                    # 更新LLMA计数
                    llma_num += len(real_output_ids) - 1
                    
                    # 动态调整文档token数量
                    for i, idx in enumerate(context_ids):
                        if i < len(candidate_match_lengths):
                            if candidate_match_lengths[i] == doc_token_num[idx]:
                                # 如果完全匹配，增加token数量
                                doc_token_num[idx] = min(doc_token_num[idx] * 2, block_K)
                            else:
                                # 否则调整为匹配长度和最小候选数的最大值
                                doc_token_num[idx] = max(min_candidate_num, candidate_match_lengths[i])
                    
                    # 更新接受的步长
                    accepted_step_length = len(real_output_ids)
                    
                    # 更新past_key_values
                    if best_candidate_idx < len(matcheds):
                        past_key_values = make_past_key_values(
                            past_key_values, 
                            step_length, 
                            accepted_step_length, 
                            matcheds[best_candidate_idx][1]
                        )
                    
                    # 创建新的output_ids张量
                    output_ids = torch.tensor([real_output_ids], dtype=prepend_ids.dtype, device=device)
                
                # 更新步数和prepend_ids
                step += accepted_step_length
                prepend_ids = output_ids[:, -1:]
                
                # 更新生成的ID
                if generate_ids is None:
                    generate_ids = output_ids
                else:
                    generate_ids = torch.cat([generate_ids, output_ids], dim=1)
                
                # 更新生成的token数量
                generate_ids_num += output_ids.size(1)
                
                # 转换为CPU张量并获取token
                output_ids_cpu = output_ids.cpu().numpy()
                output_tokens = tokenizer.convert_ids_to_tokens(output_ids_cpu[0])
                
                # 非强制解码模式下更新n-gram
                if not forced_decoding:
                    gtokens += output_tokens
                    for pos in range(len(g_ngrams_list)):
                        l = g_ngrams_list[pos][0]
                        g_ngrams_list[pos] = (l, get_ngrams(gtokens, l))
                
                # 检查是否达到终止条件
                if output_ids_cpu[0, -1] == eos_token_id or generate_ids_num >= max_new_tokens:
                    break

        result = generate_ids.cpu()
        del output
        del prepend_ids
        del ori_ids
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return result, generate_ids_num, llma_num,number


def base_generate(model,tokenizer,input_ids,gen_texts_ids,forced_decoding=False, max_new_tokens=1024,past_key_values=DynamicCache() ):#
    prepend_ids = input_ids.cuda()

    generate_ids = None
    generated_tokens = 0
    if forced_decoding and gen_texts_ids is not None:
        eos = torch.tensor([[tokenizer.eos_token_id]], dtype=gen_texts_ids.dtype, device=gen_texts_ids.device)
        gen_texts_ids = torch.cat([gen_texts_ids, eos], dim=-1)

    step=0
    step_length = 1
    
    while True:  
        with torch.no_grad():
            output = model(
                input_ids=prepend_ids,
                past_key_values=past_key_values,  
                return_dict=True,
                use_cache=True     
            )

            logits = output['logits'][:, -1:, :]
            output_ids = torch.argmax(logits, dim=-1)

            if forced_decoding:
                output_ids = gen_texts_ids[:, step:step+step_length].to(output_ids.device)
            prepend_ids= output_ids
            
            if generate_ids is None:
                generate_ids = output_ids
            else:
                generate_ids = torch.concat([generate_ids, output_ids],dim=1)
            past_key_values = output['past_key_values']
            output_ids_cpu = output_ids.cpu().numpy()
            generated_tokens += 1
            step += 1
            if output_ids_cpu[0][-1] == tokenizer.eos_token_id or generated_tokens >= max_new_tokens:
                break
    final_result = generate_ids.cpu()
    return final_result, generated_tokens

def run_time_base_test(s_list, decoding_fn, model, tokenizer,trigger_N, block_K, forced_decoding=False,ngrams_cache=None):

    print("预处理输入数据...")
    for s in s_list:
        if s['similar_answer']!=[] :
            ngrams_cache = prepare_ngrams(s, trigger_N, tokenizer, max_n=5)
            s['ngrams_cache'] = ngrams_cache
            query = s['query'][0]
            docs = ""
            for q, a in zip(s['sim_query'], s['similar_answer']):
                docs += f"similar query: {q}\nAnswer: {a}\n"
            prompt = f"docs:{docs}\n\nquery: {query}\nanswer:"
            # prompt=query
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
                ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt")
            s['inputs'] = inputs
        else:
            prompt = s['query'][0]
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
                ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True       
            )
            inputs = tokenizer(text, return_tensors="pt")
            s['inputs'] = inputs
    results = []
    print("\n开始生成回答...")
    acc_time = 0  
    total_length = 0  #
    total_start_time = time.time()  
    for idx, s in enumerate(tqdm(s_list)):
        if 'ngrams_cache' in s:
            start_time = time.time() 
            inputs = s["inputs"]
            ngrams_cache = s["ngrams_cache"]
            gen_texts_ids = None
            generate_ids ,num1,llma_num= llma_generate(model, tokenizer, inputs.input_ids, gen_texts_ids, trigger_N=1, block_K=20, 
                                                forced_decoding=forced_decoding,
                                                ngrams_cache=ngrams_cache)
                    
            total_length = generate_ids.shape[-1] + total_length  
            end_time = time.time()  
            s_time = end_time-start_time  
            acc_time = s_time + acc_time 
            generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            s["output"] = generated
            result={
                "id":s["id"],
                "query":s["query"],
                "output":s["output"],
                "s_time":s_time,
                "token_num":num1,
                'speed':num1/s_time,
                'llma_num':llma_num
            }
            results.append(result)
            print("speed:",num1/s_time)
        else:
            start_time = time.time() 
            inputs = s["inputs"]
            gen_texts_ids = None
            generate_ids,token_num = base_generate(model, tokenizer, inputs.input_ids, gen_texts_ids,
                                    forced_decoding=forced_decoding,
                                    )
            total_length = generate_ids.shape[-1] + total_length
            end_time = time.time()
            s_time = end_time-start_time
            acc_time = s_time + acc_time
            generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            s["output"] = generated
            result={
                "id":s["id"],
                "query":s["query"],
                "output":s["output"],
                "s_time":s_time,
                "token_num":token_num,
                'speed':token_num/s_time
            }
            results.append(result)
            print("speed:",token_num/s_time)
    total_end_time = time.time()
    total_time = total_end_time-total_start_time
    print(f"\n所有查询处理完成，总耗时: {total_time:.2f}秒，平均每个查询: {total_time/len(s_list):.2f}秒")
    return total_time,results


def run_time_test(s_list,  model, tokenizer,trigger_N, block_K, min_block_K=5,forced_decoding=False, similarity_model=None):
    
    print("预处理输入数据...")
    for s in s_list:
        if s['similar_answer']!=[] and s['high_similarity'] is not True:

            ngrams_cache = prepare_ngrams(s, trigger_N, tokenizer, max_n=5)
            s['ngrams_cache'] = ngrams_cache
            query = s['query'][0]
            docs = ""
            for q, a in zip(s['sim_query'], s['similar_answer']):
                docs += f"Query: {q}\nAnswer: {a}\n"
            prompt = f"query: {query}\nanswer:"
            # prompt=query
            messages = [
                {"role": "system", "content": f"{docs}\n The above are the similarity queries and their answer.\n"},
                {"role": "user", "content": prompt}
                ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt")
            s['inputs'] = inputs
        else:
            prompt = s['query'][0]
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
                ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True       
            )
            inputs = tokenizer(text, return_tensors="pt")
            s['inputs'] = inputs
    results = []
    print("\n开始生成回答...")
    acc_time = 0  
    total_length = 0 
    total_start_time = time.time() 
    for idx, s in enumerate(tqdm(s_list)):
        if 'ngrams_cache' in s:
            start_time = time.time() 
            if s['high_similarity'] :
                s["output"] = s["similar_answer"]

                end_time = time.time()
                s_time = end_time-start_time 
                acc_time = s_time + acc_time  
                result={
                    "id":s["id"],
                    "query":s["query"],
                    "output":s["output"],
                    "s_time":s_time
                }
            else:
                inputs = s["inputs"]
                ngrams_cache = s["ngrams_cache"]
                generate_ids ,num1,llma_num,number= copy_generate(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=inputs.input_ids,

                    trigger_N=2,
                    block_K=20,
                    min_block_K=5,
                    forced_decoding=forced_decoding,
                    ngrams_cache=ngrams_cache,
                    past_key_values=DynamicCache()
                )
                    

                total_length = generate_ids.shape[-1] + total_length  
                end_time = time.time()  
                s_time = end_time-start_time  
                acc_time = s_time + acc_time 

                generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                s["output"] = generated
                result={
                    "id":s["id"],
                    "query":s["query"],
                    "output":s["output"],
                    "s_time":s_time,
                    "token_num":num1,
                    'speed':num1/s_time,
                    'llma_num':llma_num,
                   'number':number
                }
                results.append(result)
                print("speed:",num1/s_time)
            
                del generate_ids, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if 'similarity_cache' in s:
                    del s['similarity_cache']
                if 'ngrams_cache' in s:
                    del s['ngrams_cache']
        else:
            start_time = time.time() 
            inputs = s["inputs"]
            gen_texts_ids = None
            generate_ids,token_num = base_generate(model, tokenizer, inputs.input_ids, gen_texts_ids,
                                    forced_decoding=forced_decoding,
                                    )
            total_length = generate_ids.shape[-1] + total_length
            end_time = time.time()
            s_time = end_time-start_time
            acc_time = s_time + acc_time
            generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            s["output"] = generated
            result={
                "id":s["id"],
                "query":s["query"],
                "output":s["output"],
                "s_time":s_time,
                "token_num":token_num,
               'speed':token_num/s_time

            }
            results.append(result)
            print("speed:",token_num/s_time)
            del generate_ids, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if 'similarity_cache' in s:
                del s['similarity_cache']
            if 'ngrams_cache' in s:
                del s['ngrams_cache']
    total_end_time = time.time()
    total_time = total_end_time-total_start_time
    print(f"\n所有查询处理完成，总耗时: {total_time:.2f}秒，平均每个查询: {total_time/len(s_list):.2f}秒")
    return total_time,results

def main():
    args = get_args()
    print(args)
    # answer_model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2",
    #                             device="cuda" if torch.cuda.is_available() else "cpu")
    # retriever = None
    if args.retriever_metadata and args.retriever_index:
        try:
            print(f"初始化检索器..."
                  f"\n - 元数据: {args.retriever_metadata}"
                  f"\n - 索引: {args.retriever_index}")
            retriever = DocumentRetriever(
                metadata_path=args.retriever_metadata,
                index_path=args.retriever_index
            )
            print("检索器初始化成功")
        except Exception as e:
            print(f"检索器初始化失败: {e}")
            retriever = None
    
    # 初始化ConjunctionExtractor实例
    print("初始化连接词提取器...")
    conjunction_extractor = ConjunctionExtractor()
    print("连接词提取器初始化成功")

    model_path = args.model_path
    print(f"加载模型: {model_path}")
    tokenizer, model = get_tokenizer_and_model(model_path)
    

    print("初始化句子相似度模型...")
    try:
        similarity_model = SentenceTransformer(
            'all-MiniLM-L6-v2', 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"句子相似度模型初始化成功 (维度: {similarity_model.get_sentence_embedding_dimension()})")
    except Exception as e:
        print(f"句子相似度模型初始化失败: {e}")
        similarity_model = None
    
    input_fn = args.input_data_fn
    trigger_N = args.n
    block_K = args.k
    print(f"加载数据...")
    s_list = load_data(input_fn, tokenizer, retriever, model,conjunction_extractor)
    with open("s_list.json","w",encoding="utf-8") as f:
        json.dump(s_list,f,ensure_ascii=False,indent=4)
    print(f"加载了 {len(s_list)} 条数据")
    print("\n开始COPY解码")
    total_time,results = run_time_test(
            s_list, 
            model, 
            tokenizer,            
            trigger_N, 
            block_K,
            min_block_K=5,
            forced_decoding=args.forced_decoding,
            similarity_model=similarity_model,
        )
    print(f"COPY总耗时: {total_time:.2f}秒")
    print("清理资源...")
    with open("sample_copy.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # print("\n开始LLMA解码")
    # base_time,results_base=run_time_base_test(
    #         s_list, 
    #         llma_generate, 
    #         model, 
    #         tokenizer,
    #         forced_decoding=args.forced_decoding,
    #         trigger_N=trigger_N,
    #         block_K=block_K,
    #         ngrams_cache=None
    #     )
    # with open("sample_llma.json", "w") as f:
    #     json.dump(results_base, f, ensure_ascii=False, indent=4)
    # print(f"LLMA总耗时: {base_time:.2f}秒")
    if retriever is not None:
        del retriever
    
if __name__ == "__main__":
    main()
