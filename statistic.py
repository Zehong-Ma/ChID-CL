import json
from collections import defaultdict
from datasets import load_dataset
data_files = {}
data_files['train'] = "./data/train_data.json"
raw_datasets = load_dataset(
        'json',
        data_files=data_files,
)
# import pdb
# pdb.set_trace()
data =raw_datasets['train']

items = data['candidates']
print("total samples: %d"%(len(items)))
idiom_set = set()
idiom_num_dict = dict()

for item in items:
    for candidates_ in item:
        for candidate in candidates_:
            
            if idiom_num_dict.__contains__(candidate) is False:
                idiom_num_dict[candidate] = 1
            else:
                idiom_num_dict[candidate] +=1
            
            # print(candidate)
            idiom_set.add(candidate)
print(sorted(idiom_num_dict.items(), key=lambda x: x[0], reverse=True))
# print(sorted(list(idiom_set)))
print("the number of idioms is: ",len(idiom_set))
with open("./data/existing_idioms.json","w") as f:
    json.dump(list(idiom_set), f, ensure_ascii=False,indent=0)