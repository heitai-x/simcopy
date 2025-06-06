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
    # parser.add_argument("--input_data_fn", type=str, default="sample_data/output_sample.json")
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

def position(input_ids : torch.Tensor,candidate_lens=None,begin_idx=0):#input_ids=query,begin_idx=kv cache长度
    input_len = input_ids.shape[-1]
    position_ids_list = [torch.arange(begin_idx, begin_idx+input_len, device=input_ids.device)]
    if candidate_lens!=[]:
        for num in candidate_lens:
            position_ids_list.append(torch.arange(begin_idx+input_len, begin_idx+input_len + num, device=input_ids.device))
    position_ids = torch.cat(position_ids_list).unsqueeze(0)
    return position_ids

def build_parallel_attention_mask(
    q_length,
    kv_length,
    copy_length,
    dtype,
    device,
    candidate_lens=None,
    batch_size=1
):
    total_length = kv_length + q_length + copy_length
    cache_position = torch.arange(
        kv_length, total_length, device=device
    )
    min_dtype = torch.finfo(dtype).min
    
    causal_mask = torch.full(
        (q_length+copy_length, total_length), fill_value=min_dtype, dtype=dtype, device=device
    )
    diagonal_attend_mask = torch.arange(total_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask *= diagonal_attend_mask
    if candidate_lens!=[] and len(candidate_lens) > 0:
        start_positions = [q_length]
        for i in range(1, len(candidate_lens)):
            start_positions.append(start_positions[-1] + candidate_lens[i-1])
        for i, draft_len in enumerate(candidate_lens):
            if i > 0:
                start_pos = start_positions[i]
                end_pos = start_pos + draft_len
                prev_start = start_positions[0]
                rows = torch.arange(start_pos, end_pos, device=device)
                cols = torch.arange(kv_length+prev_start, kv_length+start_pos, device=device)
                row_indices = rows.repeat_interleave(len(cols))
                col_indices = cols.repeat(len(rows))

                causal_mask[row_indices, col_indices] = min_dtype
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, q_length+copy_length, total_length)
    return causal_mask

def match_length(seq_a, seq_b):
    l = 0
    for i in range(min(len(seq_a), len(seq_b))):
        if seq_a[i] != seq_b[i]:
            break
        l += 1
    return l

def prepare_ngrams(s, n, tokenizer, max_n=5):
    if max_n < n:
        max_n = n
    docs = s['similar_answer']

    gtokens = None
    g_ngrams_list = None
    doc_list = [tokenizer.tokenize(x) for x in docs]
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

    return {"doc_ngrams": per_doc_ngrams_list,"doc_token_id_list": doc_token_id_list}

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

def match_prefix(g_ngrams_list, per_doc_ngrams_list, step):
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

def make_past_key_values(past_key_values, step_length, accepted_step_length,real_copy_start):#step_length当前新的kv cache长度, accepted_step_length接受的kv cache长度,real_copy_start接受的开始位置
    if isinstance(past_key_values, DynamicCache):
        kv_length = past_key_values.get_seq_length()
        # for idx in range(len(past_key_values.key_cache)):
        #     if past_key_values.key_cache[idx] != []:
        if accepted_step_length==1:
            past_key_values.crop(past_key_values.get_seq_length() - step_length + accepted_step_length)
        else:
            past_key_values._seen_tokens=past_key_values.get_seq_length()-step_length+accepted_step_length
            for idx in range(len(past_key_values.key_cache)):
                if past_key_values.key_cache[idx] != []:
                    old_key_cache = past_key_values.key_cache[idx][..., :kv_length - step_length + 1, :]
                    old_value_cache = past_key_values.value_cache[idx][..., :kv_length - step_length + 1, :]
                    start_pos = kv_length - step_length + real_copy_start
                    end_pos = start_pos + accepted_step_length - 1 
                    copy_key_cache = past_key_values.key_cache[idx][..., start_pos:end_pos, :]
                    copy_value_cache = past_key_values.value_cache[idx][..., start_pos:end_pos, :]
                    past_key_values.key_cache[idx] = torch.cat([old_key_cache, copy_key_cache], dim=-2)
                    past_key_values.value_cache[idx] = torch.cat([old_value_cache, copy_value_cache], dim=-2)

    return past_key_values
# def save_kvcache(past_key_values, save_path):
    
def copy_generate(model, tokenizer, input_ids,trigger_N=1, block_K=20,min_block_K=2,
                     forced_decoding=False,  ngrams_cache=None, max_new_tokens=1024,
                     past_key_values=DynamicCache(),min_candidate_num=5,max_n=5):
    past_key_values=DynamicCache()
    prepend_ids = input_ids.cuda()
    trigger_N = trigger_N
    block_K = block_K
    max_n = max_n
    llma_num=0
    min_candidate_num = min_candidate_num
    generate_ids = None
    generate_ids_num=0
    doc_ngrams_list = ngrams_cache["doc_ngrams"]
    doc_token_id_list = ngrams_cache["doc_token_id_list"]
    gtokens = []
    g_ngrams_list = [(n, []) for n in range(max_n,0,-1)]

    first = True
    copy_mode = False 
    n_mode= False
    doc_token_num = [min_candidate_num] * len(doc_token_id_list)
    # repetition_penalty = 1.2
    step = 0 
    number = 0
    while True:
        position_ids=None
        attention_mask=None
        all_matched_ids=[]
        candidate_lens=[]
        context_ids=[]
        ori_ids=prepend_ids
        candidate=0
        if prepend_ids.shape[-1]>1:
            used=set()
            for i,doc_token_ids in enumerate(doc_token_id_list):
                sent=doc_token_ids[:doc_token_num[i]]
                if tuple(sent) not in used: 
                    used.add(tuple(sent))
                    all_matched_ids.extend(sent)
                    context_ids.append(i)
                    candidate_lens.append(len(doc_token_ids[:doc_token_num[i]]))
            step_length=1+len(all_matched_ids)
            copied_ids = torch.tensor([all_matched_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)
            prepend_ids = torch.concat([prepend_ids,copied_ids], dim=-1)
            copy_mode=True
        else:
            step_length = 1
            copy_mode = False
        if not copy_mode:
            
            prefix_n, matches = match_prefix(g_ngrams_list, doc_ngrams_list, step)
            ori_ids=prepend_ids
            if prefix_n !=[]:
                used=set()
                all_matched_ids = []
                for i,match in enumerate(matches):
                    n=prefix_n[i]
                    for mat in match:
                        doc_idx,pos = mat
                        matched_ids = doc_token_id_list[doc_idx][pos+n:pos+n+doc_token_num[doc_idx]]
                        if tuple(matched_ids) not in used:
                            used.add(tuple(matched_ids))
                            context_ids.append(doc_idx)
                            candidate_lens.append(len(matched_ids))
                            all_matched_ids.extend(matched_ids)
                            # print("matched_ids:",matched_ids)
                step_length = 1 + len(all_matched_ids)
                copied_ids = torch.tensor([all_matched_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)
                prepend_ids = torch.concat([prepend_ids,copied_ids], dim=-1)
                n_mode=True
            else:
                step_length = 1
                n_mode = False
            
        if (copy_mode or n_mode) and len(candidate_lens) > 1:
            position_ids = position(
                    input_ids=ori_ids,#当前输入
                    candidate_lens=candidate_lens,#输入的复制文本长度列表
                    begin_idx=past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            attention_mask=build_parallel_attention_mask(
                    q_length=ori_ids.shape[-1], #query_len
                    kv_length=past_key_values.get_seq_length() if past_key_values is not None else 0,#content_len
                    copy_length=copied_ids.shape[-1],#copy_len
                    candidate_lens=candidate_lens,
                    device=prepend_ids.device,
                    dtype=model.dtype
            )

        with torch.no_grad():

            output = model(
                input_ids=prepend_ids,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
                position_ids=position_ids,
                attention_mask=attention_mask
            )
            logits = output['logits'][:, -step_length:, :]
            # if not copy_mode and not n_mode and generate_ids is not None and generate_ids.size(1) > 0:
            #     start_timer("重复惩罚")
            #     last_logits = logits[:, -1, :]
            #     existing_ids = generate_ids[0]
            #     recent_ids = existing_ids[-50:]
            #     penalty_tensor = torch.ones_like(last_logits)
            #     unique_ids, counts = torch.unique(recent_ids, return_counts=True)
            #     penalty_tensor.index_fill_(1, unique_ids, 1.0 / repetition_penalty)
            #     last_logits = last_logits * penalty_tensor
            #     logits[:, -1, :] = last_logits
            #     end_timer("重复惩罚")
            output_ids = torch.argmax(logits, dim=-1)
            accepted_step_length = step_length
            prepend_ids=prepend_ids[:, -step_length:]
            past_key_values = output.past_key_values

            if copy_mode or n_mode:
                # print("prepend_ids:",prepend_ids)
                # print("output_ids:",output_ids)
                probs = F.softmax(logits, dim=-1)
                candidate_start_positions = torch.tensor([1], device=prepend_ids.device)
                if len(candidate_lens) > 1:
                    cumulative_lens = torch.tensor(candidate_lens[:-1], device=prepend_ids.device).cumsum(0)
                    candidate_start_positions = torch.cat([candidate_start_positions, 1 + cumulative_lens])
                prepend_tensor = prepend_ids[0]
                output_tensor = output_ids[0]
                max_candidate_len = max(candidate_lens) if candidate_lens else 0
                match_matrix = torch.zeros((len(candidate_start_positions), max_candidate_len), 
                                        dtype=torch.bool, device=prepend_ids.device)
                for i, start_pos in enumerate(candidate_start_positions):
                    end_pos = start_pos + candidate_lens[i] if i < len(candidate_lens) else len(output_tensor)
                    length = min(end_pos - start_pos, max_candidate_len)
                    if length > 0:
                        for j in range(length):
                            pos = start_pos + j
                            current_idx = 0 if j == 0 else j
                            if pos < len(prepend_tensor) and current_idx < len(output_tensor):
                                current_token = prepend_tensor[pos].item()
                                current_token_prob = probs[0, current_idx, current_token].item()
                                max_prob = torch.max(probs[0, current_idx]).item()
                                prev_token_id = output_tensor[current_idx].item() if current_idx < len(output_tensor) else -1
                                match_matrix[i, j] = (current_token_prob >= max_prob * 0.8 and 
                                                    prev_token_id != tokenizer.eos_token_id)
                                if not match_matrix[i, j]:
                                    break
                candidate_match_lengths = match_matrix.sum(dim=1).cpu().tolist()
                for i ,idx in enumerate(context_ids):
                    if doc_token_num[idx]==candidate_match_lengths[i]:
                        doc_token_num[idx]=min(doc_token_num[idx]*2,block_K)
                    else:
                        doc_token_num[idx]=max(min_candidate_num,candidate_match_lengths[i])
                max_a=max(candidate_match_lengths)
                
                best_candidate_idx = candidate_match_lengths.index(max_a)
                real_copy_start = candidate_start_positions[best_candidate_idx].item()
                real_output_ids=[output_tensor[0].item()]
                for i in range(max_a):
                    real_output_ids[-1] = prepend_tensor[real_copy_start+i].item()
                    real_output_ids.append(output_tensor[real_copy_start+i].item())
                # print("real_output_ids:",real_output_ids)
                llma_num += len(real_output_ids) - 1 
                accepted_step_length = len(real_output_ids)
                past_key_values = make_past_key_values(past_key_values, step_length, accepted_step_length, real_copy_start)
                output_ids = torch.tensor([real_output_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)

            # else:
            #     output_ids=output_ids[:,-1:]
            step += accepted_step_length
            prepend_ids = output_ids[:,-1:]
            if generate_ids is None:
                generate_ids = output_ids
            else:
                generate_ids = torch.concat([generate_ids, output_ids],dim=1)
            generate_ids_num += output_ids.size(1)
            output_ids = output_ids.cpu().numpy()
            output_tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
            if not forced_decoding:
                gtokens += output_tokens
                for pos in range(len(g_ngrams_list)):
                    l = g_ngrams_list[pos][0]
                    g_ngrams_list[pos] = (l, get_ngrams(gtokens, l))
            if output_ids[0, -1] == tokenizer.eos_token_id or generate_ids_num >= 1024:
                break

    result = generate_ids.cpu()
    return result, generate_ids_num, llma_num,  number
def llma_generate(model, tokenizer, input_ids, gen_texts_ids, trigger_N, block_K, forced_decoding=False, ngrams_cache=None,past_key_values=DynamicCache()):
    past_key_values=DynamicCache()
    prepend_ids = input_ids.cuda()
    generate_ids = None
    generate_ids_num=0
    doc_ngrams_list = ngrams_cache["doc_ngrams"]
    doc_token_id_list = ngrams_cache["doc_token_id_list"]
    print(doc_token_id_list)
    gtokens = []
    g_ngrams_list = []
    g_ngrams_list = [(n, []) for n in range(5,0,-1)]
    llma_num=0
    step = 0

    while True:
        position_ids=None
        attention_mask=None
        prefix_n, matches = match_prefix(g_ngrams_list, doc_ngrams_list, step)
        ori_ids=prepend_ids
        if prefix_n !=[]:
            all_matched_ids = []
            candidate_lens = []
            for i,match in enumerate(matches):
                n=prefix_n[i]
                doc_idx,pos = match[0]
                matched_ids = doc_token_id_list[doc_idx][pos+n:pos+n+block_K-1]
                candidate_lens.append(len(matched_ids))
                all_matched_ids.extend(matched_ids)

            step_length = 1 + len(all_matched_ids)
            copied_ids = torch.tensor([all_matched_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)
            prepend_ids = torch.concat([prepend_ids,copied_ids], dim=-1)
            copy_mode=True
        else:
            step_length = 1
            copy_mode = False
        if copy_mode:
            position_ids = position(
                    input_ids=ori_ids,#当前输入
                    candidate_lens=candidate_lens,#输入的复制文本长度列表
                    begin_idx=past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            attention_mask=build_parallel_attention_mask(
                    q_length=ori_ids.shape[-1], #query_len
                    kv_length=past_key_values.get_seq_length() if past_key_values is not None else 0,#content_len
                    copy_length=copied_ids.shape[-1],#copy_len
                    candidate_lens=candidate_lens,
                    device=prepend_ids.device,
                    dtype=model.dtype
            )
        with torch.no_grad():
            output = model(
                    input_ids=prepend_ids,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    position_ids=position_ids,
                    attention_mask=attention_mask
                    )
            logits = output['logits'][:,-step_length:,:]
            output_ids = torch.argmax(logits,dim=-1)
            accepted_step_length = step_length
            past_key_values = output.past_key_values
                        
            if copy_mode:
                probs = F.softmax(logits, dim=-1)

                candidate_start_positions = torch.tensor([1], device=prepend_ids.device)
                if len(candidate_lens) > 1:
                    cumulative_lens = torch.tensor(candidate_lens[:-1], device=prepend_ids.device).cumsum(0)
                    candidate_start_positions = torch.cat([candidate_start_positions, 1 + cumulative_lens])

                prepend_tensor = prepend_ids[0]
                output_tensor = output_ids[0]

                max_candidate_len = max(candidate_lens) if candidate_lens else 0
                match_matrix = torch.zeros((len(candidate_start_positions), max_candidate_len), 
                                        dtype=torch.bool, device=prepend_ids.device)

                for i, start_pos in enumerate(candidate_start_positions):
                    end_pos = start_pos + candidate_lens[i] if i < len(candidate_lens) else len(output_tensor)
                    length = min(end_pos - start_pos, max_candidate_len)
                    
                    if length > 0:
                        for j in range(length):
                            pos = start_pos + j
                            current_idx = 0 if j == 0 else j
                            if pos < len(prepend_tensor) and current_idx < len(output_tensor):
                                current_token = prepend_tensor[pos].item()
                                current_token_prob = probs[0, current_idx, current_token].item()
                                max_prob = torch.max(probs[0, current_idx]).item()
                                
                                prev_token_id = output_tensor[current_idx].item() if current_idx < len(output_tensor) else -1
                                match_matrix[i, j] = (current_token_prob >= max_prob * 0.8 and 
                                                    prev_token_id != tokenizer.eos_token_id)
                                
                                if not match_matrix[i, j]:
                                    break
                
                candidate_match_lengths = match_matrix.sum(dim=1).cpu().tolist()
                candidate_tokens = []
                for i, start_pos in enumerate(candidate_start_positions):
                    tokens = [output_tensor[0].item()]
                    for j in range(candidate_match_lengths[i]):
                        if j == 0:
                            tokens[-1] = prepend_tensor[start_pos + j].item()
                        else:
                            tokens.append(output_tensor[j].item())
                    candidate_tokens.append(tokens)
                best_candidate_idx = candidate_match_lengths.index(max(candidate_match_lengths))
                real_output_ids = candidate_tokens[best_candidate_idx]
                real_copy_start = candidate_start_positions[best_candidate_idx].item()
                llma_num += len(real_output_ids) - 1 
                accepted_step_length = len(real_output_ids)
                
                past_key_values = make_past_key_values(past_key_values, step_length, accepted_step_length, real_copy_start)
                output_ids = torch.tensor([real_output_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)
                # print("output_ids_new:",output_ids)
            # else:
            #     output_ids=output_ids[:,-1:]
            step += accepted_step_length
            prepend_ids = output_ids[:,-1:]
            # prepend_ids = output_ids
            # print("prepend_ids:",prepend_ids)
            if generate_ids is None:
                generate_ids = output_ids
            else:
                generate_ids = torch.concat([generate_ids, output_ids],dim=1)
            generate_ids_num += output_ids.size(1)
            output_ids = output_ids.cpu().numpy()
            output_tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
            if not forced_decoding:
                gtokens += output_tokens
                for pos in range(len(g_ngrams_list)):
                    l = g_ngrams_list[pos][0]
                    g_ngrams_list[pos] = (l,get_ngrams(gtokens, l))
            if output_ids[0,-1] == tokenizer.eos_token_id or generate_ids_num >= 1024:
                break
    return generate_ids.cpu(), generate_ids_num,llma_num

def base_generate(model,tokenizer,input_ids,gen_texts_ids,forced_decoding=False, max_new_tokens=1024,past_key_values=DynamicCache() ):#
    prepend_ids = input_ids.cuda()
    past_key_values=DynamicCache()
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
def run_time_base(s_list, model, tokenizer, forced_decoding=False,ngrams_cache=None):
    print("预处理输入数据...")
    for s in s_list:
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

        start_time = time.time() 
        inputs = s["inputs"]
        gen_texts_ids = None
        generate_ids ,num1= base_generate(model, tokenizer, inputs.input_ids, gen_texts_ids,
                                            forced_decoding=forced_decoding)
                
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
            'speed':num1/s_time
        }
        results.append(result)
        print("speed:",num1/s_time)
    total_end_time = time.time()
    total_time = total_end_time-total_start_time
    print(f"\n所有查询处理完成，总耗时: {total_time:.2f}秒，平均每个查询: {total_time/len(s_list):.2f}秒")
    return total_time,results

def run_time_llma_test(s_list, decoding_fn, model, tokenizer,trigger_N, block_K, forced_decoding=False,ngrams_cache=None):

    print("预处理输入数据...")
    for s in s_list:
        if s['similar_answer']!=[] :
            ngrams_cache = prepare_ngrams(s, trigger_N, tokenizer, max_n=5)
            s['ngrams_cache'] = ngrams_cache
            query = s['query'][0]
            # docs = ""
            # for q, a in zip(s['sim_query'], s['similar_answer']):
            #     docs += f"similar query: {q}\nAnswer: {a}\n"
            # prompt = f"docs:{docs}\n\nquery: {query}\nanswer:"
            prompt=query
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
            generate_ids ,num1,llma_num= llma_generate(model, tokenizer, inputs.input_ids, gen_texts_ids, trigger_N=trigger_N, block_K=block_K, 
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


def run_time_test(s_list, decoding_fn, model, tokenizer,trigger_N, block_K, forced_decoding=False, similarity_model=None):
    print("预处理输入数据...")
    for s in s_list:
        if s['similar_answer']!=[] and s['high_similarity'] is not True:
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
    total_length = 0 
    total_start_time = time.time() 
    for idx, s in enumerate(tqdm(s_list)):
        if 'ngrams_cache' in s or s['high_similarity'] is True:
            start_time = time.time() 
            if s['high_similarity'] is True:
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
                gen_texts_ids = None
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
                print("speed:",num1/s_time)
            results.append(result)
                
            
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
    # print(f"\nBase解码开始")
    # base_time,results_base=run_time_base(
    #     s_list=s_list,
    #     model=model,
    #     tokenizer=tokenizer,
    #     forced_decoding=args.forced_decoding,
    #     ngrams_cache=None
    #     )    
    # with open("sample_base_no.json", "w") as f:
    #     json.dump(results_base, f, ensure_ascii=False, indent=4)
    # print(f"BASE总耗时: {base_time:.2f}秒")
    print("\n开始COPY解码")
    total_time,results = run_time_test(
            s_list, 
            copy_generate, 
            model, 
            tokenizer,            
            trigger_N, 
            block_K,
            forced_decoding=args.forced_decoding,
            similarity_model=similarity_model,
        )
    print(f"COPY总耗时: {total_time:.2f}秒")
    print("清理资源...")
    with open("sample_copy_no.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # print("\n开始LLMA解码")
    # llma_time,results_copy=run_time_llma_test(
    #         s_list, 
    #         llma_generate, 
    #         model, 
    #         tokenizer,
    #         forced_decoding=args.forced_decoding,
    #         trigger_N=trigger_N,
    #         block_K=block_K,
    #         ngrams_cache=None
    #     )
    # with open("sample_llma_no.json", "w") as f:
    #     json.dump(results_copy, f, ensure_ascii=False, indent=4)
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    # print(f"LLMA总耗时: {llma_time:.2f}秒")

    if retriever is not None:
        del retriever
    
if __name__ == "__main__":
    main()
