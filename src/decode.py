import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
cache_dir = "/mnt/sevenT/debinx/huggingface_models"
import torch.cuda
import time
import json
import os
from collections import defaultdict
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# 修改spacy初始化部分
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.add_pipe("sentencizer") 
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--type", type=str, default="llma")
    parser.add_argument("--input_data_fn", type=str, default="output_sample.json")
    parser.add_argument("--forced_decoding", action="store_true")
    parser.add_argument("--retriever_metadata", type=str, help="检索器元数据路径", default="src/faiss_dataset/metadata/metadata.json")
    parser.add_argument("--retriever_index", type=str, help="检索器索引路径", default="src/faiss_dataset/metadata/vector_index_cosine.faiss")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()
    return args
def is_complete_sentence(text):
    """使用 NLP 解析句子，判断是否完整"""
    doc = nlp(text.strip())
    if not doc:
        return False
    last_token = doc[-1]
    return last_token.is_punct

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

def prepare_ngrams(s, n, tokenizer, max_n=5):
    if max_n < n:
        max_n = n
    docs = s['similar_answer']

    gtokens = None
    g_ngrams_list = None
    doc_list = [tokenizer.tokenize(x) for x in docs]
    doc_token_id_list = [tokenizer.convert_tokens_to_ids(x) for x in doc_list]
    doc_ngrams_list = []

    for l in range(n, max_n+1):
        doc_ngrams = defaultdict(list)
        for i, doc_tokens in enumerate(doc_list):
            ngram_list = get_ngrams(doc_tokens, l)
            for j, ngram in enumerate(ngram_list):
                doc_ngrams[ngram].append((i,j))
        doc_ngrams_list.append([l,doc_ngrams])
    doc_ngrams_list.reverse()

    return {"target_ngrams": g_ngrams_list, "doc_list": doc_list, "doc_ngrams": doc_ngrams_list, "target_tokens": gtokens, "doc_token_id_list": doc_token_id_list}
def get_tokenizer_and_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    

    model = AutoModelForCausalLM.from_pretrained(
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

def load_data(input_fn, tokenizer, retriever=None, model=None,answer_model=None):
    s_list = []
    gen_texts_ids=None
    forced_decoding=False
    with open(input_fn, 'r',encoding='utf-8') as file:
        data = json.load(file)
        for s in data:
            if retriever is not None:
                similar_docs = retriever.search(s['query'], s['id'], k=5)
                if similar_docs!=[]:
                    answers=[]
                    for i,similar_doc in enumerate(similar_docs):
                        similarity = similar_doc['similarity']  
                        sim_query = similar_doc['text']
                        sim_id = similar_doc['id']
                        if similarity >= 0.95:  
                            prompt = sim_query
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
                                    forced_decoding=forced_decoding,
                                        )
                            generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                            s['similarity_score'] = similarity
                            s['similar_id'] = sim_id
                            s['similar_answer'] = generated
                            s['sim_query'] = sim_query
                            s['high_similarity'] = True
                            s_list.append(s)
                            break
                        elif similarity >= 0.7:  
                            prompt = sim_query
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
                                        forced_decoding=forced_decoding,
                                        )
                            generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]   
                            s['high_similarity'] = False
                            answers.append(generated)
                    if "high_similarity" in s and s['high_similarity']==False:
                        start=time.time()
                        ranks= answer_model.rank(s['query'][0], answers, return_documents=True)
                        print("rank time:",time.time()-start)
                        s['similar_answer'] = []
                        s['similar_answer'].append(ranks[0]['text'])
                        print(s['similar_answer'])
                        s_list.append(s)                    
                else:
                    s_list.append(s)
    return s_list

def prepare_sentences(s, tokenizer, embedding_model=None):

    sim_ans_sentences = []
    sim_ans_token_id_lists = []
    sim_embeddings = []

    if 'similar_answer' in s:
        docs = s['similar_answer']

        for doc in nlp.pipe(docs):

            sentences = [sent.text for sent in doc.sents]
            sim_ans_sentences.extend(sentences)

            for sent in sentences:
                token_ids = tokenizer(
                sent,
                add_special_tokens=False,
                return_tensors='pt'
                )['input_ids'].tolist()[0]
                sim_ans_token_id_lists.append(token_ids)
            if embedding_model is not None and sentences:
                embeddings = embedding_model.encode(
                    sentences,
                    batch_size=8,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )
                embeddings = embeddings.cpu().numpy()
                sim_embeddings.append(embeddings)

        return {
            "sim_ans_sentences": sim_ans_sentences,
            "sim_ans_token_id_lists": sim_ans_token_id_lists,
            "sim_embeddings": sim_embeddings
        }
def make_kv_tupe(input_kv, step_length, accepted_step_length):
    kv_list = []
    for kv in input_kv:
        l = kv.shape[2]
        kv_list.append(kv[:,:,:l-step_length+accepted_step_length,:])
    return tuple(kv_list),l

def match_prefix(g_ngrams_list, doc_ngrams_list, step):
    for g_ngrams, doc_ngrams in zip(g_ngrams_list, doc_ngrams_list):
        n = g_ngrams[0]
        g_ngrams = g_ngrams[1]
        doc_ngrams = doc_ngrams[1]
        if step < n:
            continue
        if g_ngrams[step-n] in doc_ngrams.keys():
            return n, g_ngrams, doc_ngrams
    return 0, None, None

def make_past_key_values(past_key_values, step_length, accepted_step_length,):
    if step_length == accepted_step_length:
        return past_key_values
    pkv_list = []
    for kv in past_key_values:
        kv,l = make_kv_tupe(kv, step_length, accepted_step_length)
        pkv_list.append(kv)
        for kvs in kv:
            n= kvs.shape[2]
    return tuple(pkv_list)
def copy_generate(model, tokenizer, input_ids, gen_texts_ids, trigger_N, block_K,
                forced_decoding=False, similarity_cache=None,
                similarity_model=None,ngrams_cache=None,query=None):  

    

    timing_stats = defaultdict(list)
    
    start_events = {}
    end_events = {}
    
    def start_timer(name):
        start_events[name] = torch.cuda.Event(enable_timing=True)
        start_events[name].record()
    
    def end_timer(name):
        end_events[name] = torch.cuda.Event(enable_timing=True)
        end_events[name].record()
        torch.cuda.synchronize()  
        elapsed_time = start_events[name].elapsed_time(end_events[name]) / 1000  
        timing_stats[name].append(elapsed_time)
    

    start_timer("总时间")
    
    prepend_ids = input_ids.cuda()
    copy_num=0
    llma_num=0
    sentences_ids=[]
    generate_ids = None
    past_key_values = None
    generate_ids_num=0
    doc_ngrams_list = ngrams_cache["doc_ngrams"]
    doc_n_token_id_list = ngrams_cache["doc_token_id_list"]
    gtokens = []
    g_ngrams_list = []
    for nlist in doc_ngrams_list:
        g_ngrams_list.append((nlist[0], []))    
    doc_sentences = similarity_cache['sim_ans_sentences']
    doc_token_id_list = similarity_cache['sim_ans_token_id_lists']
    doc_embeddings = similarity_cache['sim_embeddings']


    if doc_embeddings and all(len(emb_list) > 0 for emb_list in doc_embeddings):
        start_timer("嵌入处理")
        device = prepend_ids.device  
        
        all_embeddings_list = [
            torch.as_tensor(emb_list, device=device) 
            if not isinstance(emb_list, torch.Tensor) 
            else emb_list.to(device)
            for emb_list in doc_embeddings
        ]
        
        all_embeddings = torch.cat(all_embeddings_list)
        

        lengths = (len(emb) for emb in all_embeddings_list)
        cum_lengths = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.tensor(list(lengths), device=device).cumsum(0)
        ])
        end_timer("嵌入处理")
    else:

        all_embeddings = torch.empty(0, dtype=torch.float32, device=prepend_ids.device)
        cum_lengths = torch.zeros(1, dtype=torch.long, device=prepend_ids.device)

    modify_mode = False 
    copy_mode = False 
    n_mode= False
    sim_sent_idx = 0  
    repetition_penalty = 1.2  # 惩罚
    
    # if len(all_embeddings) > 0:
    #     current_embedding = similarity_model.encode(
    #         query,
    #         convert_to_tensor=True,
    #         normalize_embeddings=True,
    #         device=prepend_ids.device
    #         )
    #     current_embedding_tensor = current_embedding.unsqueeze(0)
    #     similarities = torch.nn.functional.cosine_similarity(
    #         current_embedding_tensor,
    #         all_embeddings.unsqueeze(0),
    #         dim=2
    #         ).squeeze(0)
                        
    #     max_sim, flat_idx = torch.max(similarities, dim=0)
    #     max_sim = max_sim.item()
                                
    #     if max_sim > 0.6:
    #         doc_idx = torch.searchsorted(cum_lengths[1:], flat_idx + 1, right=True).item()
    #         sent_idx = flat_idx - cum_lengths[doc_idx].item()
    #         sim_sent_idx = sent_idx+1

    now_sent_ids = []
    step = 0 
    number = 0
    while True:  
        if sim_sent_idx < len(doc_sentences) and not modify_mode:
            start_timer("复制模式处理")
            sim_ids = doc_token_id_list[sim_sent_idx]
            copy_mode = True
            if len(sim_ids) > 15:
                sim_ids = sim_ids[:15]
            step_length = 1 + len(sim_ids)
            original_prepend_ids=prepend_ids
            copied_ids = torch.tensor([sim_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)
            prepend_ids = torch.concat([prepend_ids, copied_ids], dim=-1)
            end_timer("复制模式处理")
        else:
            step_length = 1
            copy_mode = False
        if not copy_mode:
            start_timer("n-gram匹配")
            prefix_n, g_ngrams, doc_ngrams = match_prefix(g_ngrams_list, doc_ngrams_list, step)
            if prefix_n > 0:
                n_mode = True
                trigger_ngram = g_ngrams[step-prefix_n]
                i, j = doc_ngrams[trigger_ngram][0]
                copied_ids = doc_n_token_id_list[i][j+prefix_n:j+prefix_n+block_K-1]
                step_length = 1+len(copied_ids)
                copied_ids = torch.tensor([copied_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)
                prepend_ids = torch.concat([prepend_ids,copied_ids], dim=-1)

            else:
                step_length = 1
                n_mode = False
            end_timer("n-gram匹配")

        with torch.no_grad():
            
            start_timer("模型推理")
            output = model(
                input_ids=prepend_ids,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True
            )
            
            logits = output['logits'][:, -step_length:, :]
            end_timer("模型推理")
            
            if not copy_mode and not n_mode and generate_ids is not None and generate_ids.size(1) > 0:
                start_timer("重复惩罚")
                last_logits = logits[:, -1, :]
                existing_ids = generate_ids[0]
                
                recent_ids = existing_ids[-50:]
                
                penalty_tensor = torch.ones_like(last_logits)
                
                unique_ids, counts = torch.unique(recent_ids, return_counts=True)
            
                penalty_tensor.index_fill_(1, unique_ids, 1.0 / repetition_penalty)
                

                last_logits = last_logits * penalty_tensor
                
                logits[:, -1, :] = last_logits
                end_timer("重复惩罚")

            output_ids = torch.argmax(logits, dim=-1)
            accepted_step_length = step_length
            past_key_values = output['past_key_values']
            if copy_mode:
                start_timer("复制模式token处理")
                iids = copied_ids.cpu().numpy()[0]
                oids = output_ids.cpu().numpy()[0]
                real_output_ids = []
                probs = F.softmax(logits, dim=-1)
                for pos in range(1, len(oids)):
                    expected_id = iids[pos-1]
                    token_prob = probs[0, pos-1, expected_id].item()
                    max_prob = torch.max(probs[0, pos-1]).item()  
                    if token_prob > max_prob * (1 - 0.2) and oids[pos-1] != tokenizer.eos_token_id:
                        real_output_ids.append(expected_id)
                        copy_num += 1
                        sentences_ids.append(sim_sent_idx)
                        copy_mode = False
                        modify_mode = True
                    else:
                        copy_mode = False
                        modify_mode = True
                        break
                end_timer("复制模式token处理")
                
                accepted_step_length = len(real_output_ids)
                past_key_values = make_past_key_values(past_key_values, step_length, accepted_step_length)

                if accepted_step_length < step_length:
                    output_ids = output_ids[:, :accepted_step_length]
            else:
                if n_mode:
                    start_timer("n-gram模式token处理")
                    iids = prepend_ids.cpu().numpy()[0]
                    oids = output_ids.cpu().numpy()[0]
                    real_output_ids = [oids[0]]  
                    probs = F.softmax(logits, dim=-1)
                    for pos in range(1, len(oids)):
                        current_token_prob = probs[0, pos-1, iids[pos]].item() 
                        max_prob = torch.max(probs[0, pos-1]).item()  

                        if current_token_prob > max_prob * (1 - 0.2) and oids[pos-1] != tokenizer.eos_token_id:
                            real_output_ids.append(iids[pos])
                            llma_num += 1
                        else:
                            copy_mode = False
                            modify_mode = True
                            break
                    # for pos in range(1,len(oids)):
                    #     if oids[pos-1] == iids[pos] and oids[pos-1] != tokenizer.eos_token_id:
                    #         real_output_ids.append(oids[pos])
                    #         llma_num += 1
                    #     else:
                    #         copy_mode = False
                    #         modify_mode = True
                    #         break
                    accepted_step_length = len(real_output_ids)
                    past_key_values = make_past_key_values(past_key_values, step_length, accepted_step_length)
                    
                    if accepted_step_length < step_length:
                        output_ids = output_ids[:, :accepted_step_length]
                    end_timer("n-gram模式token处理")
            step += accepted_step_length
            if output_ids.size(1) == 0:
                prepend_ids = original_prepend_ids[:, -1:]
                continue

            prepend_ids = output_ids[:,-1:]
            if modify_mode :
                current_token_id = output_ids[0, -1].item()
                now_sent_ids.append(current_token_id)
                if current_token_id in [13, 0, 30, 1773, 6313, 11319]:
                    start_timer("相似度计算")
                    now_sent = tokenizer.decode(now_sent_ids)
                    current_embedding = similarity_model.encode(
                                now_sent,
                                convert_to_tensor=True,
                                normalize_embeddings=True,
                                device=prepend_ids.device
                        )
                    
                    current_embedding_tensor = current_embedding.unsqueeze(0)
                    
                    if len(all_embeddings) > 0:
                        similarities = torch.nn.functional.cosine_similarity(
                                current_embedding_tensor,
                                all_embeddings.unsqueeze(0),
                                dim=2
                        ).squeeze(0)
                        
                        max_sim, flat_idx = torch.max(similarities, dim=0)
                        max_sim = max_sim.item()
                                
                        if max_sim > 0.5:
                            doc_idx = torch.searchsorted(cum_lengths[1:], flat_idx + 1, right=True).item()
                            sent_idx = flat_idx - cum_lengths[doc_idx].item()
                            if sent_idx+1 not in sentences_ids:
                                sim_sent_idx = sent_idx+1
                                modify_mode = False
                                copy_mode = True
                                number += 1
                    now_sent_ids=[]
                    end_timer("相似度计算")
            else:
                sim_sent_idx += 1
                sentences_ids.append(sim_sent_idx)

            start_timer("token生成")
            if generate_ids is None:
                generate_ids = output_ids
            else:
                generate_ids = torch.concat([generate_ids, output_ids],dim=1)
            generate_ids_num += output_ids.size(1)
            output_ids = output_ids.cpu().numpy()
            output_tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
            end_timer("token生成")
            
            if not forced_decoding:
                start_timer("n-gram更新")
                gtokens += output_tokens
                for pos in range(len(g_ngrams_list)):
                    l = g_ngrams_list[pos][0]
                    g_ngrams_list[pos] = (l, get_ngrams(gtokens, l))
                end_timer("n-gram更新")

            if output_ids[0, -1] == tokenizer.eos_token_id:
                break

    end_timer("总时间")
    
    result = generate_ids.cpu()
    print("copy_num:", copy_num)
    print("llma_num:", llma_num)

    timing_summary = {}
    for key, times in timing_stats.items():
        if times:
            timing_summary[key] = {
                "平均时间(秒)": sum(times) / len(times),
                "总时间(秒)": sum(times),
                "调用次数": len(times),
                "最短时间(秒)": min(times),
                "最长时间(秒)": max(times)
            }

            print(f"{key} - 平均: {timing_summary[key]['平均时间(秒)']:.6f}秒, 总计: {timing_summary[key]['总时间(秒)']:.6f}秒, 调用: {timing_summary[key]['调用次数']}次")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "timing_stats")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"gpu_timing_stats_{timestamp}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=4, ensure_ascii=False)
    
    print(f"GPU时间统计已保存到: {output_file}")
    
    return result, generate_ids_num, llma_num, copy_num, number

def llma_generate(model, tokenizer, input_ids, gen_texts_ids, trigger_N, block_K, forced_decoding=False, ngrams_cache=None):
    prepend_ids = input_ids.cuda()
    generate_ids = None
    past_key_values = None
    generate_ids_num=0
    doc_ngrams_list = ngrams_cache["doc_ngrams"]
    doc_token_id_list = ngrams_cache["doc_token_id_list"]
    if forced_decoding:
        gtokens = ngrams_cache["target_tokens"]
        g_ngrams_list = ngrams_cache["target_ngrams"]
        eos = torch.tensor([[tokenizer.eos_token_id]], dtype=gen_texts_ids.dtype, device=gen_texts_ids.device)
        gen_texts_ids = torch.cat([gen_texts_ids, eos], dim=-1)
    else:
        gtokens = []
        g_ngrams_list = []
        for nlist in doc_ngrams_list:
            g_ngrams_list.append((nlist[0], []))
    llma_num=0
    step = 0
    while True:
        prefix_n, g_ngrams, doc_ngrams = match_prefix(g_ngrams_list, doc_ngrams_list, step)
        if prefix_n > 0:
            copy_mode = True
            trigger_ngram = g_ngrams[step-prefix_n]
            i, j = doc_ngrams[trigger_ngram][0]
            copied_ids = doc_token_id_list[i][j+prefix_n:j+prefix_n+block_K-1]
            step_length = 1+len(copied_ids)
            copied_ids = torch.tensor([copied_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)
            prepend_ids = torch.concat([prepend_ids,copied_ids], dim=-1)
        else:
            step_length = 1
            copy_mode = False
        with torch.no_grad():
            output = model(input_ids=prepend_ids,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True)
            logits = output['logits'][:,-step_length:,:]
            output_ids = torch.argmax(logits,dim=-1)
            accepted_step_length = step_length
            past_key_values = output['past_key_values']
            if forced_decoding:
                output_ids = gen_texts_ids[:,step:step+step_length].to(output_ids.device)
            if copy_mode:
                iids = prepend_ids.cpu().numpy()[0]
                oids = output_ids.cpu().numpy()[0]
                real_output_ids = [oids[0]]
                probs = F.softmax(logits, dim=-1)
                for pos in range(1,len(oids)):
                    if oids[pos-1] == iids[pos] and oids[pos-1] != tokenizer.eos_token_id:
                        real_output_ids.append(oids[pos])
                        llma_num+=1
                    else:
                        break
                accepted_step_length = len(real_output_ids)
                print
                past_key_values = make_past_key_values(past_key_values, step_length, accepted_step_length)
                if accepted_step_length < step_length:
                    output_ids = output_ids[:,:accepted_step_length]
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
            if output_ids[0,-1] == tokenizer.eos_token_id:
                break
    return generate_ids.cpu(), generate_ids_num,llma_num



def base_generate(model,tokenizer,input_ids,gen_texts_ids,forced_decoding=False, max_new_tokens=512):
    prepend_ids = input_ids.cuda()

    generate_ids = None
    past_key_values = None
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
        if 'similar_answer' in s:
            ngrams_cache = prepare_ngrams(s, trigger_N, tokenizer, max_n=5)
            s['ngrams_cache'] = ngrams_cache
            query = s['query'][0]
            # prompt = f"docs:\n{docs}\nquery: {query}\nanswer:"
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
        if 'similarity_cache' in s:
            start_time = time.time() 
            inputs = s["inputs"]
            ngrams_cache = s["ngrams_cache"]
            gen_texts_ids = None
            generate_ids ,num1,llma_num= llma_generate(model, tokenizer, inputs.input_ids, gen_texts_ids, trigger_N=1, block_K=15, 
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
        if 'similar_answer' in s and s['high_similarity'] is not True:
            s_t = time.time()
            similarity_cache = prepare_sentences(s, tokenizer, embedding_model=similarity_model)
            print(f"预处理完成，耗时: {time.time() - s_t:.2f}秒")
            if time.time() - s_t > 10:
                print(similarity_cache["sim_ans_sentences"])
            s['similarity_cache'] = similarity_cache
            ngrams_cache = prepare_ngrams(s, trigger_N, tokenizer, max_n=5)
            s['ngrams_cache'] = ngrams_cache
            print(f"预处理完成，耗时: {time.time() - s_t:.2f}秒")
            query = s['query'][0]
            docs = '\n'.join(s['similar_answer'])
            # prompt = f"docs:\n{docs}\nquery: {query}\nanswer:"
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
    total_length = 0 
    total_start_time = time.time() 
    for idx, s in enumerate(tqdm(s_list)):
        if 'similarity_cache' in s:
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
                similarity_cache = s["similarity_cache"]
                ngrams_cache = s["ngrams_cache"]
                gen_texts_ids = None
                generate_ids ,num1,llma_num,copy_num,number= copy_generate(model, tokenizer, inputs.input_ids, gen_texts_ids, trigger_N=1, block_K=20, 
                                                forced_decoding=forced_decoding,
                                                query=s['query'][0],
                                                similarity_model=similarity_model, 
                                                similarity_cache=similarity_cache,
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
                    'llma_num':llma_num,
                    'copy_num':copy_num,
                   'number':number
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

def main():
    args = get_args()
    print(args)
    answer_model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2",
                                device="cuda" if torch.cuda.is_available() else "cpu")
    retriever = None
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
    s_list = load_data(input_fn, tokenizer, retriever, model,answer_model)
    print(f"加载了 {len(s_list)} 条数据")
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
    with open("sample_copy.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n开始LLMA解码")
    base_time,results_base=run_time_base_test(
            s_list, 
            llma_generate, 
            model, 
            tokenizer,
            forced_decoding=args.forced_decoding,
            trigger_N=trigger_N,
            block_K=block_K,
            ngrams_cache=None
        )
    with open("sample_llma.json", "w") as f:
        json.dump(results_base, f, ensure_ascii=False, indent=4)
    print(f"LLMA总耗时: {base_time:.2f}秒")
    if retriever is not None:
        del retriever
    
if __name__ == "__main__":
    main()
