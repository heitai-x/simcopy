import json
with open ("sample_base_one.json","r",encoding="utf-8") as f:
    data=json.load(f)
    # print(data)
results = []
for i in data:
    result = {
        'id': i['id'],
        'query': i['query'],
        'output': i['output'],
        's_time': i['s_time'],
        'token_num': i['token_num'],
        'speed': i['token_num'] / i['s_time']
    }
    results.append(result)
with open("sample_base_one.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)