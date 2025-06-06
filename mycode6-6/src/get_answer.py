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

def position_my(query,draft_ids,matcheds,position_ids,begin_idx,draft_all=None):
    # draft_ids为当前草稿
    #query为正常推理时的输入token
    #position_ids为已经处理过的位置信息，根据当前草稿进行更新
    #begin_idx为kvcache长度
    # draft_all为上一步处理完的输入
    #matcheds存储已经处理过的macthed_ids和matched_pos，然后利用已处理过的matched去进行进行前缀搜索，维护三个列表，一个是总体的位置信息，一个是已经处理过的token(存在的前缀)，一个是当前草稿中未被处理过token
    prefix_pos=[]#当前草稿在input中的位置信息
    draft_new_pos=[]#当前草稿中未被处理过的token的位置信息
    query_len = query.shape[-1]#一般为1
    draft_len = len(draft_ids)#当前草稿长度
    draft_all_len = draft_all.shape[-1] if draft_all is not None else 0#上一步处理完的输入长度
    start=begin_idx+query_len
    # prefix_pos保存当前草稿中已经处理过的前缀的位置信息
    # new_pos保存当前草稿中未被处理过的token的位置信息
    for matched_ids,matched_pos,_ in matcheds:
        pre_pos=[]
        pre_i=0
        while(pre_i<draft_len):
            if draft_ids[pre_i]==matched_ids[pre_i]:
                pre_pos.append(matched_pos[pre_i])
                pre_i+=1
            else:
                break
        if len(pre_pos)>len(prefix_pos):
            prefix_pos=pre_pos    
    pre_len=len(prefix_pos)
    
    position_ids.append(torch.arange(start+pre_len, start+draft_len, device=query.device))

    position_ids=torch.cat(position_ids).unsqueeze(0)

    if pre_len < draft_len:

        if not isinstance(draft_ids, torch.Tensor):
            draft_ids_tensor = torch.tensor(draft_ids, device=query.device)
        else:
            draft_ids_tensor = draft_ids.to(query.device)

        remaining_ids = draft_ids_tensor[pre_len:].unsqueeze(0)
        if draft_all is None:
            draft_all = remaining_ids.clone()
        else:
            draft_all = torch.cat([draft_all, remaining_ids], dim=1)
        a=1+draft_all_len-pre_len
        for i in range(pre_len, draft_len):
            draft_new_pos.append(a+i)

    new_matched_ids = draft_ids.copy() if isinstance(draft_ids, list) else draft_ids.clone().cpu().tolist()
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
    total_length = q_length + copy_length
    min_dtype = torch.finfo(dtype).min
    
    causal_mask = torch.full(
        (total_length , total_length), fill_value=min_dtype, dtype=dtype, device=device
    )
    causal_indices = torch.tril(torch.ones((q_length, total_length), device=device))
    # 将前q_length行的下三角部分设置为0（可见）
    causal_mask[:q_length, :total_length] = torch.where(
        causal_indices > 0,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(min_dtype, dtype=dtype, device=device)
    )
    # 使用matched_pos和draft_new_pos进行掩码处理
    if matcheds and len(matcheds) > 0:
        # 获取最新的匹配信息
        for _, matched_pos, draft_new_pos in matcheds:
            if not draft_new_pos or not matched_pos:
                continue
            rows=torch.tensor([i for i in draft_new_pos],  device=device)
            cols=torch.tensor([i for i in matched_pos],  device=device)
            row_indices = rows.repeat_interleave(len(cols))
            col_indices = cols.repeat(len(rows))
            causal_mask[row_indices, col_indices] = 0.0
    causal_mask = torch.cat([torch.full((total_length , kv_length), fill_value=0.0, dtype=dtype, device=device),causal_mask],dim=-1)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, q_length+copy_length, total_length+kv_length)
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
    """计算匹配矩阵,判断每个候选序列的匹配程度"""
    match_matrix = torch.zeros((len(matcheds), max_candidate_len), 
                            dtype=torch.bool, device=prepend_tensor.device)

    for i, (matched_ids, matched_pos, draft_new_pos) in enumerate(matcheds):
        for pre_idx,pos in enumerate(matched_pos):
            out_pos=matched_pos[pre_idx-1] if pre_idx>0 else 0
            current_token = prepend_tensor[pos].item()
            current_token_prob = probs[0, out_pos, current_token].item()
            max_prob = torch.max(probs[0, out_pos]).item()
            
            prev_token_id = output_tensor[out_pos].item() if out_pos < len(output_tensor) else -1
            match_matrix[i, pre_idx] = (current_token_prob >= max_prob * 0.8 and 
                                prev_token_id != tokenizer.eos_token_id)
            if not match_matrix[i, pre_idx]:
                break
                
    return match_matrix

def _get_best_candidate(candidate_match_lengths, matcheds, prepend_tensor, output_tensor):
    """根据匹配矩阵选择最佳候选序列"""
    # candidate_tokens = []
    a=max(candidate_match_lengths)
    best_candidate_idx = candidate_match_lengths.index(a)
    real_copy_matched_pos = matcheds[best_candidate_idx][1]
    real_output_ids=[output_tensor[0].item()]
    for j in range(a):
        real_output_ids[-1]= prepend_tensor[real_copy_matched_pos[j]].item()
        real_output_ids.append(output_tensor[real_copy_matched_pos[j]].item())
    
    return real_output_ids, best_candidate_idx

def make_past_key_values(past_key_values, step_length, accepted_step_length, real_copy_matched_pos):
    # 只保留前a个位置
    real_copy_matched_pos = real_copy_matched_pos[:accepted_step_length-1]

    kv_length = past_key_values.get_seq_length()
    if accepted_step_length == 1:
        past_key_values.crop(past_key_values.get_seq_length() - step_length + accepted_step_length)
    else:
        past_key_values._seen_tokens = past_key_values.get_seq_length() - step_length + accepted_step_length
        old_end = kv_length - step_length + 1
        copy_indices = [kv_length - step_length + idx for idx in real_copy_matched_pos]
        for idx, (key_cache, value_cache) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
            if key_cache is None or key_cache.numel() == 0:
                continue
            old_key_cache = key_cache[..., :old_end, :]
            old_value_cache = value_cache[..., :old_end, :]
            gather_indices = torch.tensor(copy_indices, device=key_cache.device)
            copy_key_cache = torch.index_select(key_cache, -2, gather_indices)
            copy_value_cache = torch.index_select(value_cache, -2, gather_indices)
            past_key_values.key_cache[idx] = torch.cat([old_key_cache, copy_key_cache], dim=-2)
            past_key_values.value_cache[idx] = torch.cat([old_value_cache, copy_value_cache], dim=-2)
    return past_key_values

def copy_generate(model, tokenizer, input_ids,trigger_N=1, block_K=20,min_block_K=2,
                     forced_decoding=False,  ngrams_cache=None, max_new_tokens=1024,
                     past_key_values=DynamicCache(),min_candidate_num=5,max_n=5):
        past_key_values = DynamicCache()
        trigger_N = trigger_N
        block_K = block_K
        max_n = max_n
        min_candidate_num = min_candidate_num
        prepend_ids = input_ids.cuda()
        llma_num=0
        generate_ids = None
        generate_ids_num=0
        doc_ngrams_list = ngrams_cache["doc_ngrams"]
        doc_token_id_list = ngrams_cache["doc_token_id_list"]

        gtokens = []
        g_ngrams_list = [(n, []) for n in range(max_n,0,-1)]
        modify_mode = True 
        copy_mode = False 
        n_mode= False
        doc_token_num = [min_candidate_num] * len(doc_token_id_list)
        # repetition_penalty = 1.2
        step = 0 
        number = 0
        while True:
            position_ids=None
            attention_mask=None
            candidate_lens=[]
            context_ids=[]
            matcheds=[]
            begin_idx=past_key_values.get_seq_length() if past_key_values is not None else 0
            ori_ids=prepend_ids
            
            if prepend_ids.shape[-1]>1:
                used=set()
                number+=1
                draft_all=None
                postion_ids=[torch.arange(begin_idx, begin_idx+prepend_ids.shape[-1], device=prepend_ids.device)]
                for i,doc_token_ids in enumerate(doc_token_id_list):
                    sent=doc_token_ids[:doc_token_num[i]]
                    if tuple(sent) not in used: 
                        used.add(tuple(sent))
                        draft_all, matcheds, position_ids=position_my(
                            prepend_ids,
                            sent,
                            matcheds,
                            postion_ids,
                            begin_idx=past_key_values.get_seq_length() if past_key_values is not None else 0,
                            draft_all=draft_all
                            )
                        context_ids.append(i)
                        candidate_lens.append(len(doc_token_ids[:doc_token_num[i]]))
                step_length=1+draft_all.shape[-1]
                mybuild_parallel_attention_mask(prepend_ids.shape[-1],begin_idx,draft_all.shape[-1],matcheds=matcheds,device=prepend_ids.device,dtype=model.dtype)
                prepend_ids = torch.concat([prepend_ids,draft_all], dim=-1)
                copy_mode=True
            else:
                step_length = 1
                copy_mode = False
            if not copy_mode:
                prefix_n, matches = match_prefix(g_ngrams_list, doc_ngrams_list, step)
                if prefix_n !=[]:
                    draft_all=None
                    number+=1
                    used=set()
                    all_matched_ids = []
                    postion_ids=[torch.arange(begin_idx, begin_idx+prepend_ids.shape[-1], device=prepend_ids.device)]
                    for i,match in enumerate(matches):
                        n=prefix_n[i]
                        for mat in match:
                            doc_idx,pos=mat
                            matched_ids = doc_token_id_list[doc_idx][pos+n:pos+n+doc_token_num[doc_idx]]
                            if tuple(matched_ids) not in used:
                                used.add(tuple(matched_ids))
                                context_ids.append(doc_idx)
                                candidate_lens.append(len(matched_ids))
                                draft_all, matcheds, position_ids=position_my(prepend_ids,
                                    matched_ids,
                                    matcheds,
                                    postion_ids,
                                    begin_idx=past_key_values.get_seq_length() if past_key_values is not None else 0,
                                    draft_all=draft_all
                                )
                    step_length = 1 + draft_all.shape[-1]
                    mybuild_parallel_attention_mask(prepend_ids.shape[-1],begin_idx,draft_all.shape[-1],matcheds=matcheds,device=prepend_ids.device,dtype=model.dtype)
                    prepend_ids = torch.concat([prepend_ids,draft_all], dim=-1)

                    n_mode=True
                else:
                    step_length = 1
                    n_mode = False
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
                    probs = F.softmax(logits, dim=-1)
                    prepend_tensor = prepend_ids[0]
                    output_tensor = output_ids[0]
                    max_candidate_len = max(candidate_lens) if candidate_lens else 0
                    match_matrix = _compute_match_matrix(tokenizer,probs, prepend_tensor, output_tensor, matcheds, max_candidate_len)
                    candidate_match_lengths = match_matrix.sum(dim=1).cpu().tolist()
                    real_output_ids, best_candidate_idx = _get_best_candidate(candidate_match_lengths, matcheds,prepend_tensor, output_tensor)
                    llma_num += len(real_output_ids)-1 
                    for i,idx in enumerate(context_ids):
                        if candidate_match_lengths[i] == doc_token_num[idx]:
                            doc_token_num[idx]=min(doc_token_num[idx]*2,block_K)
                        else:
                            doc_token_num[idx]=max(min_candidate_num,candidate_match_lengths[i])
                    accepted_step_length = len(real_output_ids)
                    past_key_values = make_past_key_values(past_key_values, step_length, accepted_step_length, matcheds[best_candidate_idx][1])
                    output_ids = torch.tensor([real_output_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)
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
                if output_ids[0, -1] == tokenizer.eos_token_id or generate_ids_num >= max_new_tokens:
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
