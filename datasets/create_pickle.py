import os
import pickle

cls2lbl = {}
counter = 0
root = os.getcwd() + '/tiny-imagenet-200'
#print(root)
for cls in sorted(os.listdir(root + '/train')):
    cls2lbl[cls] = counter
    counter += 1

data = {}
data['train'] = {}
data['train']['imgs'] = []
data['train']['labels'] = []

for cls in sorted(os.listdir(root + '/train')):
    for imgs in sorted(os.listdir(root + '/train/' + cls + '/images')):
        src = root + '/train/' + cls + '/images/' + imgs
        data['train']['imgs'].append(src)
        data['train']['labels'].append(cls2lbl[cls])

data['val'] = {}
data['val']['imgs'] = []
data['val']['labels'] = []

f = open(root + '/val/val_annotations.txt')
context = f.readlines()
f.close()
valimg2lbl = {}
for i in context:
    i_split = i.rstrip().split('\t')
    valimg2lbl[i_split[0]] = cls2lbl[i_split[1]]


for imgs in sorted(os.listdir(root + '/val/images')):
    src = root + '/val/images/' + imgs
    data['val']['imgs'].append(src)
    data['val']['labels'].append(valimg2lbl[imgs])

print("Train images: {}".format(len(data['train']['labels'])))
print("Val images: {}".format(len(data['val']['labels'])))

with open('tiny_imagenet.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
