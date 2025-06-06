import os
import torch
import time
import json
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import Qwen2ForCausalLM, AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm
from retriever import DocumentRetriever
from tree import ConjunctionExtractor
import spacy
import numpy as np
from transformers.cache_utils import DynamicCache

class CopyModel:
    def __init__(self, model_path, trigger_N,block_K,min_block_K,max_n,cache_dir="/root/autodl-tmp/huggingface_models"):
        """初始化LLMA模型类"""
        self.cache_dir = cache_dir
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trigger_N = trigger_N
        self.block_K = block_K
        self.max_n=max_n
        self.min_block_K = min_block_K
        # 初始化分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir
        )
        
        # self.nlp_model = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        # self.nlp_model.add_pipe("sentencizer")
        
        # self.similarity_model = self._init_similarity_model()
    def position_my(self,query,draft_ids,matcheds,position_ids,begin_idx,draft_all=None):
        """
        优化的位置信息处理函数，用于前缀树搜索。
        
        Args:
            query: 正常推理时的输入token
            draft_ids: 当前草稿
            matcheds: 存储已处理过的matched_ids和matched_pos
            position_ids: 已处理过的位置信息，根据当前草稿进行更新
            begin_idx: kvcache长度
            draft_all: 上一步处理完的输入
            
        Returns:
            Tuple包含更新后的draft_all, matcheds和position_ids
        """
        # 初始化变量
        device = query.device
        query_len = query.shape[-1]  # 一般为1
        draft_len = len(draft_ids)
        draft_all_len = 0 if draft_all is None else draft_all.shape[-1]
        start = begin_idx + query_len
        
        # 查找最长匹配前缀
        prefix_pos = []
        max_prefix_len = 0
        
        # 优化前缀匹配查找
        for matched_ids, matched_pos, _ in matcheds:
            # 计算当前匹配的前缀长度
            min_len = min(draft_len, len(matched_ids))
            i = 0
            
            # 使用快速比较找到匹配前缀
            while i < min_len and draft_ids[i] == matched_ids[i]:
                i += 1
                
            # 如果找到更长的前缀，则更新
            if i > max_prefix_len:
                max_prefix_len = i
                prefix_pos = matched_pos[:i]
        
        # 创建新位置ID并合并
        position_ids.append(torch.arange(start + max_prefix_len, start + draft_len, device=device))
        position_ids = torch.cat(position_ids).unsqueeze(0)
        
        # 处理未匹配部分
        draft_new_pos = []
        if max_prefix_len < draft_len:
            # 转换draft_ids为tensor（如果需要）
            if not isinstance(draft_ids, torch.Tensor):
                draft_ids_tensor = torch.tensor(draft_ids, device=device)
            else:
                draft_ids_tensor = draft_ids.to(device)
            
            # 提取剩余ID
            remaining_ids = draft_ids_tensor[max_prefix_len:].unsqueeze(0)
            
            # 更新draft_all
            if draft_all is None:
                draft_all = remaining_ids.clone()
            else:
                draft_all = torch.cat([draft_all, remaining_ids], dim=1)
            
            # 计算新位置
            offset = 1 + draft_all_len - max_prefix_len
            # 使用列表推导式替代循环
            draft_new_pos = [offset + i for i in range(max_prefix_len, draft_len)]
        
        # 创建新的匹配信息
        new_matched_ids = draft_ids.copy() if isinstance(draft_ids, list) else draft_ids.clone().cpu().tolist()
        new_matched_pos = prefix_pos + draft_new_pos
        
        # 添加到matcheds
        matcheds.append((new_matched_ids, new_matched_pos, draft_new_pos))
        
        return draft_all, matcheds, position_ids
        
    def mybuild_parallel_attention_mask(
        self,
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
    


    def position(self,input_ids : torch.Tensor,candidate_lens=None,begin_idx=0):#input_ids=query,begin_idx=kv cache长度
        input_len = input_ids.shape[-1]
        position_ids_list = [torch.arange(begin_idx, begin_idx+input_len, device=input_ids.device)]
        if candidate_lens!=[]:
            for num in candidate_lens:
                position_ids_list.append(torch.arange(begin_idx+input_len, begin_idx+input_len + num, device=input_ids.device))
        position_ids = torch.cat(position_ids_list).unsqueeze(0)
        return position_ids

    def build_parallel_attention_mask(
        self,
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
    
    def match_prefix(self,g_ngrams_list,per_doc_ngrams_list, step):

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

        
    def make_past_key_values(self, past_key_values, step_length, accepted_step_length, real_copy_matched_pos):
        """更新past_key_values，保留需要的位置
        
        Args:
            past_key_values: KV缓存对象
            step_length: 当前新的KV缓存长度
            accepted_step_length: 接受的KV缓存长度
            real_copy_matched_pos: 需要复制的匹配位置列表
            
        Returns:
            更新后的past_key_values对象
        """
        # 只保留前accepted_step_length-1个位置
        real_copy_matched_pos = real_copy_matched_pos[:accepted_step_length-1]

        # 获取当前KV缓存长度
        kv_length = past_key_values.get_seq_length()
        
        if accepted_step_length == 1:
            # 如果只接受一个位置，直接裁剪
            past_key_values.crop(kv_length - step_length + accepted_step_length)
        else:
            # 更新已见token数量
            past_key_values._seen_tokens = kv_length - step_length + accepted_step_length
            
            # 计算旧缓存结束位置
            old_end = kv_length - step_length + 1
            
            # 计算需要复制的索引位置
            copy_indices = [kv_length - step_length + idx for idx in real_copy_matched_pos]
            
            # 更新每层的KV缓存
            for idx, (key_cache, value_cache) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
                # 跳过空缓存
                if key_cache is None or key_cache.numel() == 0:
                    continue
                    
                # 提取旧缓存部分
                old_key_cache = key_cache[..., :old_end, :]
                old_value_cache = value_cache[..., :old_end, :]
                
                # 创建张量索引，确保在与key_cache相同的设备上
                gather_indices = torch.tensor(copy_indices, device=key_cache.device)
                
                # 使用索引选择需要复制的部分
                copy_key_cache = torch.index_select(key_cache, -2, gather_indices)
                copy_value_cache = torch.index_select(value_cache, -2, gather_indices)
                
                # 拼接旧缓存和复制的缓存
                past_key_values.key_cache[idx] = torch.cat([old_key_cache, copy_key_cache], dim=-2)
                past_key_values.value_cache[idx] = torch.cat([old_value_cache, copy_value_cache], dim=-2)
        return past_key_values
    
    def _compute_match_matrix(self, probs, prepend_tensor, output_tensor, matcheds, max_candidate_len):
        """计算匹配矩阵，判断每个候选序列的匹配程度
        
        Args:
            tokenizer: 分词器对象
            probs: 概率分布张量
            prepend_tensor: 前缀张量
            output_tensor: 输出张量
            matcheds: 匹配的候选序列信息列表
            max_candidate_len: 最大候选序列长度
            
        Returns:
            torch.Tensor: 匹配矩阵，表示每个候选序列的匹配情况
        """
        # 创建匹配矩阵，初始化为全0布尔张量
        match_matrix = torch.zeros(
            (len(matcheds), max_candidate_len), 
            dtype=torch.bool, 
            device=prepend_tensor.device
        )

        # 阈值常量，用于判断token概率是否足够高
        PROB_THRESHOLD = 0.8
        
        # 遍历每个候选序列
        for candidate_idx, (matched_ids, matched_positions, draft_new_pos) in enumerate(matcheds):
            # 遍历候选序列中的每个位置
            for position_idx, current_position in enumerate(matched_positions):
                # 确定输出位置：如果不是第一个位置，使用前一个匹配位置，否则使用0
                output_position = matched_positions[position_idx-1] if position_idx > 0 else 0
                
                # 获取当前token及其概率
                current_token = prepend_tensor[current_position].item()
                current_token_prob = probs[0, output_position, current_token].item()
                
                # 获取该位置的最大概率
                max_prob = torch.max(probs[0, output_position]).item()
                
                # 获取前一个token ID
                prev_token_id = output_tensor[output_position].item() if current_position < len(output_tensor) else -1
                
                # 判断当前token是否匹配：概率足够高且不是EOS
                is_match = (current_token_prob >= max_prob * PROB_THRESHOLD and 
                        prev_token_id != tokenizer.eos_token_id)
                
                # 更新匹配矩阵
                match_matrix[candidate_idx, position_idx] = is_match
                
                # 如果不匹配，终止当前候选序列的匹配
                if not is_match:
                    break
                    
        return match_matrix

    def _get_best_candidate(self, candidate_match_lengths, candidate_start_positions, prepend_tensor, output_tensor):
        """根据匹配矩阵选择最佳候选序列
        
        Args:
            candidate_match_lengths: 每个候选序列的匹配长度列表
            matcheds: 匹配的候选序列信息列表
            prepend_tensor: 前缀张量
            output_tensor: 输出张量
            
        Returns:
            tuple: (生成的输出ID列表, 最佳候选序列索引)
        """
    
        max_match_length = max(candidate_match_lengths)
        best_candidate_idx = candidate_match_lengths.index(max_match_length)

        matched_positions = matcheds[best_candidate_idx][1]

        real_output_ids = [output_tensor[0].item()]

        for j in range(max_match_length):
        
            real_output_ids[-1] = prepend_tensor[matched_positions[j]].item()
            
            real_output_ids.append(output_tensor[matched_positions[j]].item())
        
        return real_output_ids, best_candidate_idx

    
    def get_ngrams(self,tokens, n):
        n=self.trigger_N
        ngram_list = []
        for i in range(len(tokens)-n+1):
            ngram = ' '.join(tokens[i:i+n])
            ngram_list.append(ngram)
        return ngram_list

    def get_input_text(self,query_text, sim_docs=[]):
        if sim_docs is not None:
            docs = ""
            for q, a in zip(sim_docs['sim_query'], sim_docs['similar_answer']):
                docs += f"similar query: {q}\nAnswer: {a}\n"
            prompt = f"docs:{docs}\n\nquery: {query_text}\nanswer:"
        else:
            prompt = f"query: {query_text}\nanswer:"
        print("prompt:",prompt)
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(                            
            messages,
            tokenize=False,       
            add_generation_prompt=True
            )
        
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs

    def prepare_ngrams(self, s ,n, max_n=5):

        n=self.trigger_N
        if max_n < n:
            max_n = n
        docs = s
        doc_list = [self.tokenizer.tokenize(x) for x in docs]
        # if generate_ids is not None:
        #     doc_token_id_list=generate_ids
        # else:
        doc_token_id_list = [self.tokenizer.convert_tokens_to_ids(x) for x in doc_list]
        per_doc_ngrams_list = []
        for doc_idx, doc_tokens in enumerate(doc_list):
            doc_ngrams_list = []
            # 为每个长度生成n-grams
            for l in range(n, max_n+1):
                doc_ngrams = defaultdict(list)
                ngram_list = self.get_ngrams(doc_tokens, l)
                for pos, ngram in enumerate(ngram_list):
                    doc_ngrams[ngram].append((doc_idx, pos))  # 存储文档索引和位置
                doc_ngrams_list.append([l, doc_ngrams])
            doc_ngrams_list.reverse()  # 长的n-grams优先
            per_doc_ngrams_list.append(doc_ngrams_list)

        return { "doc_ngrams": per_doc_ngrams_list,"doc_token_id_list": doc_token_id_list}
    

    def base_generate(self, input_ids,  forced_decoding=False, 
                     max_new_tokens=1024, past_key_values=DynamicCache()):
        past_key_values = DynamicCache()
        prepend_ids = input_ids.cuda()

        generate_ids = None
        generated_tokens = 0
        if forced_decoding and gen_texts_ids is not None:
            eos = torch.tensor([[self.tokenizer.eos_token_id]], dtype=gen_texts_ids.dtype, device=gen_texts_ids.device)
            gen_texts_ids = torch.cat([gen_texts_ids, eos], dim=-1)

        step=0
        step_length = 1
        
        while True:  
            with torch.no_grad():
                output = self.model(
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
                if output_ids_cpu[0][-1] == self.tokenizer.eos_token_id or generated_tokens >= max_new_tokens:
                    break
        final_result = generate_ids.cpu()
        del output
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return final_result, generated_tokens

    def llma_generate(self,model, tokenizer, input_ids, gen_texts_ids, trigger_N, block_K, forced_decoding=False, ngrams_cache=None,past_key_values=DynamicCache(),n_grams=5,max_new_tokens=1024):
        model=self.model
        tokenizer=self.tokenizer
        prepend_ids = input_ids.cuda()
        generate_ids = None
        generate_ids_num=0
        doc_ngrams_list = ngrams_cache["doc_ngrams"]
        doc_token_id_list = ngrams_cache["doc_token_id_list"]
        gtokens = []
        g_ngrams_list = []
        g_ngrams_list = [(n, []) for n in range(n_grams,0,-1)]
        llma_num=0
        step = 0
        while True:
            position_ids=None
            attention_mask=None
            copied_ids=None
            all_matched_ids = []
            candidate_lens = []
            prefix_n, matches = self.match_prefix(g_ngrams_list, doc_ngrams_list, step)
            ori_ids=prepend_ids
            if prefix_n !=[]:
                used=set()
                for i,match in enumerate(matches):
                    n=prefix_n[i]
                    doc_idx,pos = match[0]
                    matched_ids = doc_token_id_list[doc_idx][pos+n:pos+n+block_K-1]
                    if matched_ids not in used:
                        used.add(matched_ids)
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
                position_ids = self.position(
                        input_ids=ori_ids,#当前输入
                        candidate_lens=candidate_lens,#输入的复制文本长度列表
                        begin_idx=past_key_values.get_seq_length() if past_key_values is not None else 0
                )
                attention_mask=self.build_parallel_attention_mask(
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
                    past_key_values = self.make_past_key_values(past_key_values, step_length, accepted_step_length, real_copy_start)
                    output_ids = torch.tensor([real_output_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)
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
                        g_ngrams_list[pos] = (l,self.get_ngrams(gtokens, l))
                if output_ids[0,-1] == tokenizer.eos_token_id or generate_ids_num >= 1024:
                    break
        return generate_ids.cpu(), generate_ids_num,llma_num

    def copy_generate(self, input_ids,  trigger_N=1, block_K=20,
                     forced_decoding=False,  ngrams_cache=None, max_new_tokens=1024,
                     past_key_values=DynamicCache(),min_candidate_num=5,max_n=5):
        past_key_values = DynamicCache()
        trigger_N = self.trigger_N
        block_K = self.block_K
        max_n = self.max_n
        min_candidate_num = self.min_block_K

        # if similarity_model is None:
        #     similarity_model = self.similarity_model
        prepend_ids = input_ids.cuda()
        llma_num=0
        generate_ids = None
        generate_ids_num=0
        doc_ngrams_list = ngrams_cache["doc_ngrams"]
        doc_token_id_list = ngrams_cache["doc_token_id_list"]
        # print("doc_token_id_list:",doc_token_id_list)
        gtokens = []
        g_ngrams_list = [(n, []) for n in range(max_n,0,-1)]
        modify_mode = True 
        copy_mode = False 
        n_mode= False
        doc_token_num = [block_K] * len(doc_token_id_list)
        # repetition_penalty = 1.2
        step = 0 
        number = 0
        print("prepend_ids:",prepend_ids.shape)
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
                        draft_all, matcheds, position_ids=self.position_my(
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
                self.mybuild_parallel_attention_mask(prepend_ids.shape[-1],begin_idx,draft_all.shape[-1],matcheds=matcheds,device=prepend_ids.device,dtype=model.dtype)
                prepend_ids = torch.concat([prepend_ids,draft_all], dim=-1)
                copy_mode=True
            else:
                step_length = 1
                copy_mode = False
            if not copy_mode:
                prefix_n, matches = self.match_prefix(g_ngrams_list, doc_ngrams_list, step)
                if prefix_n !=[]:
                    draft_all=None
                    begin_idx=past_key_values.get_seq_length() if past_key_values is not None else 0
                    number+=1
                    used=set()
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
                                draft_all, matcheds, position_ids=self.position_my(prepend_ids,
                                    matched_ids,
                                    matcheds,
                                    postion_ids,
                                    begin_idx=past_key_values.get_seq_length() if past_key_values is not None else 0,
                                    draft_all=draft_all
                                )
                    step_length = 1 + draft_all.shape[-1]
                    self.mybuild_parallel_attention_mask(prepend_ids.shape[-1],begin_idx,draft_all.shape[-1],matcheds=matcheds,device=prepend_ids.device,dtype=model.dtype)
                    prepend_ids = torch.concat([prepend_ids,draft_all], dim=-1)

                    n_mode=True
                else:
                    step_length = 1
                    n_mode = False
            with torch.no_grad():
                output = self.model(
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
                    match_matrix = self._compute_match_matrix(probs, prepend_tensor, output_tensor, matcheds, max_candidate_len)
                    candidate_match_lengths = match_matrix.sum(dim=1).cpu().tolist()
                    real_output_ids, best_candidate_idx  = self._get_best_candidate(candidate_match_lengths, candidate_start_positions, prepend_tensor, output_tensor)
                    llma_num += len(real_output_ids) - 1 
                    for i,idx in enumerate(context_ids):
                        if candidate_match_lengths[i] == doc_token_num[idx]:
                            doc_token_num[idx]=max(doc_token_num[idx]*2,block_K)
                        else:
                            doc_token_num[idx]=max(min_candidate_num,candidate_match_lengths[i])
                    accepted_step_length = len(real_output_ids)
                    past_key_values = self.make_past_key_values(past_key_values, step_length, accepted_step_length, matcheds[best_candidate_idx][1])
                    output_ids = torch.tensor([real_output_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)
                step += accepted_step_length
                prepend_ids = output_ids[:,-1:]
                if generate_ids is None:
                    generate_ids = output_ids
                else:
                    generate_ids = torch.concat([generate_ids, output_ids],dim=1)
                generate_ids_num += output_ids.size(1)
                output_ids = output_ids.cpu().numpy()
                output_tokens = self.tokenizer.convert_ids_to_tokens(output_ids[0])
                if not forced_decoding:
                    gtokens += output_tokens
                    for pos in range(len(g_ngrams_list)):
                        l = g_ngrams_list[pos][0]
                        g_ngrams_list[pos] = (l, self.get_ngrams(gtokens, l))
                if output_ids[0, -1] == self.tokenizer.eos_token_id or generate_ids_num >= max_new_tokens:
                    break

        result = generate_ids.cpu()
        del output
        del prepend_ids
        del ori_ids
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return result, generate_ids_num, llma_num,number


    def process(self, data, mode="base", **kwargs):

        start_time = time.time()
        inputs=self.get_input_text(data["query"], data["similar_docs"])
        # 根据模式选择不同的生成方法
        if mode == "base":
            output = self.base_generate(inputs.input_ids)
        elif mode == "llma":
            output = self.llma_generate(
                inputs.input_ids,
                trigger_N=kwargs.get("trigger_N", 1),
                block_K=kwargs.get("block_K", 10),
                ngrams_cache=data.get("ngrams_cache")
            )
        elif mode == "copy":
            output = self.copy_generate(
                inputs.input_ids,
                trigger_N=kwargs.get("trigger_N", 1),
                block_K=kwargs.get("block_K", 10),
                similarity_cache=data.get("similarity_cache"),
                ngrams_cache=data.get("ngrams_cache")
            )
            
        end_time = time.time()
        process_time = end_time - start_time
        total_time += process_time
        
        # 处理结果
        result = {
            "id": data["id"],
            "query": data["query"],
            "output": self.tokenizer.decode(output[0], skip_special_tokens=True),
            "time": process_time
        }

        return result, total_time

    def cleanup(self):
        """清理资源"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
