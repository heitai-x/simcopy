# import json
# import random
# with open('output.json', 'r',encoding='utf-8') as file:
#     data = json.load(file)
# sample_size = 100 # Number of random items to select
# random_items = random.sample(data, sample_size)
# results = []
# for item in random_items:
#     result = {
#         "id": item["id"],
#         "query": item["human_conversations"]
#     }
#     results.append(result)
# with open('output_sample.json', 'w',encoding='utf-8') as file:
#     json.dump(results, file, ensure_ascii=False, indent=4)
import json
import random
with open('sample_data/sample_copy.json', 'r',encoding='utf-8') as file:
    data = json.load(file)
    token_num=0
    time_num=0
    for item in data:
        if "copy_num" in item:
            token_num+=item["token_num"]
            time_num+=item["s_time"]
    print(token_num/time_num)
with open('sample_data/sample_llma.json', 'r',encoding='utf-8') as file1:
    data=json.load(file1)
    token_num=0
    time_num=0
    for item in data:
        if "llma_num" in item:
            token_num+=item["token_num"]
            time_num+=item["s_time"]
    print(token_num/time_num)