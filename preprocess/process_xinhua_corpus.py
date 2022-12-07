import json
import os
def parse_idioms(data_path):
    with open(os.path.join(data_path, "idiom.json"), 'r') as f:
        idiom_data = json.load(f)
    with open(os.path.join(data_path, "tokenizer_ori.json"), 'r') as f:
        vocabs = json.load(f)
    f_txt = open(os.path.join(data_path,  "vocab_ori.txt"), 'a')
    
    processed_data = []
    idiom_to_index = {}
    id = 21128
    max_exp_len = 0
    for data_item in idiom_data:
        idiom_dict = {}
        idiom_dict['idiom'] = data_item['word']
        idiom_dict['explaination'] = data_item['explanation']
        if max_exp_len<len(data_item["explanation"]):
            max_exp_len = len(data_item["explanation"])
        example_derivation_id = data_item['example'].find('★')
        idiom_dict['example'] = data_item['example'][:example_derivation_id].replace("～", '#idiom#') if example_derivation_id!=-1 else data_item['example']
        idiom_dict['derivation'] = data_item['derivation']
        if idiom_dict['example'] == "":
            idiom_dict['example'] = data_item['derivation'].replace(idiom_dict['idiom'], '#idiom#')
        if idiom_dict['explaination'] == "":
            print(data_item['word']+" doesn't have explaination!")
            continue
        idiom_dict['id'] = id
        idiom_to_index[data_item['word']] = id
        vocabs['model']['vocab'][data_item['word']] = id
        f_txt.write(data_item['word']+"\n")
        id += 1
        processed_data.append(idiom_dict)
    # with open(os.path.join(data_path, "idiom_processed.json"), 'w') as f:
    #     json.dump(processed_data, f, ensure_ascii=False, indent=2)
    # with open(os.path.join(data_path, "idiom_to_index.json"), 'w') as f:
    #     json.dump(idiom_to_index, f, ensure_ascii=False, indent=2)
    # with open(os.path.join(data_path, "my_tokenizer", "tokenizer.json"), 'w') as f:
    #     json.dump(vocabs, f, ensure_ascii=False, indent=1)
    f_txt.close()
    print("max length of explains is: ",max_exp_len)
    print('the number of idioms is %d'%len(processed_data))
    
    return
    
parse_idioms("./data")