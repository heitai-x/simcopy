import os
# 设置 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 指定模型缓存目录（如果有需要）
cache_dir = "/mnt/sevenT/debinx/huggingface_models"


from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
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

from sentence_transformers import CrossEncoder

def get_tokenizer_and_model(model_path):
    """加载预训练的分词器和语言模型
    
    参数:
        model_path (str): HuggingFace模型路径或本地路径
        
    返回:
        tuple: (tokenizer, model) 分词器对象和模型对象
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 加载语言模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 使用半精度浮点数节省显存
        device_map="auto" ,          # 自动分配GPU设备
        cache_dir=cache_dir,       # 指定缓存目录

    )
    
    return tokenizer, model

def base_generate(model,tokenizer,input_ids,gen_texts_ids,forced_decoding=False, max_new_tokens=512):
    prepend_ids = input_ids.cuda()

    generate_ids = None
    past_key_values = None
    
    # 记录已生成的token数量
    generated_tokens = 0
    if forced_decoding and gen_texts_ids is not None:
        eos = torch.tensor([[tokenizer.eos_token_id]], dtype=gen_texts_ids.dtype, device=gen_texts_ids.device)
        gen_texts_ids = torch.cat([gen_texts_ids, eos], dim=-1)

    step=0
    step_length = 1
    
    while True:  # 主生成循环
        with torch.no_grad():
            # 计时模型推理
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
            
            # 检查终止条件
            if output_ids_cpu[0][-1] == tokenizer.eos_token_id or generated_tokens >= max_new_tokens:
                break

    final_result = generate_ids.cpu()
    
    return final_result, generated_tokens

def convert_numpy_types(obj):
    """转换 NumPy 类型为 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

similarity_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2",
                                device="cuda" if torch.cuda.is_available() else "cpu")
retriever = DocumentRetriever(
    metadata_path="src/faiss_dataset/metadata/metadata.json",
    index_path="src/faiss_dataset/metadata/vector_index_cosine.faiss"
    )
tokenizer, model=get_tokenizer_and_model(model_path="Qwen/Qwen2.5-32B-Instruct")
with open('input_data.json', 'r', encoding='utf-8') as f:
    data=json.load(f)
    ranks=[]
for item in data:
    query_id = item['id']
    query_text = item['query'][0]
    results = retriever.search(query_text,query_id, k=5)
    if results !=[]:
        generateds=[]
        for result in results:
            similarity = result['similarity']
            sim_query = result['text']
            print('sim_query: ',sim_query)
            sim_id=result['id']
            if similarity>=0.8 and similarity<0.95:
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
                            
                generate_ids,token_num = base_generate(model, tokenizer, inputs.input_ids, gen_texts_ids=None,
                    forced_decoding=False,
                )
                generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                generateds.append(generated)
        if generateds:
            start=time.time()
            rank = similarity_model.rank(item['query'][0], generateds, return_documents=True)
            print(time.time()-start)
            # 转换 rank 中的 NumPy 类型
            if isinstance(rank, (list, tuple)):
                rank = [convert_numpy_types(r) for r in rank]
            else:
                rank = convert_numpy_types(rank)
            
            result={
                "id":query_id,
                "query":query_text,
                "rank":rank
            }
            ranks.append(result)
# with open('output_sample_rank.json', 'w', encoding='utf-8') as f:
#     json.dump(ranks, f, ensure_ascii=False, indent=4, default=convert_numpy_types)
