import json

rel = ['P156', 'P84', 'P39', 'P276', 'P410', 'P241', 'P177', 'P264']
with open('MyModel/dataset/FewRel/FewRel_id2name.json') as f:
    rel2name = json.load(f)
with open('MyModel/dataset/FewRel/data_with_marker.json') as f:
    data = json.load(f)
for r in rel:
    print(r, rel2name[r])
    for d in data[r][:10]:
        print(' '.join(d['tokens']))
    print()

