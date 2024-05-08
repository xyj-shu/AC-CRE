import json


with open('MyModel/dataset/FewRel/id2rel.json') as f:
    id2rel = json.load(f)
rel2id = {rel: i for i, rel in enumerate(id2rel)}
# with open('MyModel/dataset/FewRel/rel2id.json', 'w', encoding='utf-8') as f:
#     json.dump(rel2id, f)

rel = ['P156', 'P84', 'P39', 'P276', 'P410', 'P241', 'P177', 'P264']
id = [rel2id[r] for r in rel]
print(id)