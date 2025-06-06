import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
cache_dir = "/mnt/sevenT/debinx/huggingface_models"
import torch.cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=cache_dir
)
doc2="query:Explain the concept of quantum computing in simple terms that a high school student could understand.\n answer:\nOf course! Let\'s break down the concept of quantum computing into simpler terms that a high school student can understand:\n\nImagine you have a big box of colored balls, and each ball is either red or blue. In traditional computing, you would have to check each ball one by one to see which ones are red and which ones are blue. This can take a long time if you have a lot of balls.\n\nNow, think about a quantum computer. In the quantum world, each ball can be both red and blue at the same time—a concept known as \"superposition.\" This means that a quantum computer can check all the balls simultaneously, not one by one, which is incredibly fast and efficient.\n\nQuantum computers use particles called qubits. Qubits are special because they can be in multiple states at the same time. So, if you were checking for specific balls, a qubit would be capable of representing both possibilities (red and blue) at once, dramatically speeding up the process.\n\nAdditionally, there\'s a phenomenon called \"entanglement.\" Imagine you have two special balls that are connected. If you change the state of one ball, the state of the other ball will instantly change, no matter how far apart they are. This allows quantum computers to work on many problems at once, getting answers much faster than a regular computer can.\n\nQuantum computing is still a very new and exciting field, but it holds great promise for solving complex problems quickly and efficiently. It\'s like having a super-fast, super-smart helper to solve puzzles and find solutions to really big problems!"
# Create message arrays for each prompt
prompt2 =f"docs:{doc2}\n\n Explain the concept of quantum computing in simple terms that a high school student could understand."
messages2 = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt2}
]


# Apply chat template to each prompt
text2 = tokenizer.apply_chat_template(
    messages2,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text2], return_tensors="pt").to(model.device)
start=time.time()
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
end=time.time()
print("Time taken for generation:", end-start)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
print(len(generated_ids[0]))
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)