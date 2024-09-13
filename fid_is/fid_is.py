import os
from tqdm import tqdm
import gc
import random
random.seed(42)
from collections import defaultdict

from PIL import Image
import numpy as np
import pandas as pd

import torchvision.transforms as transforms
import torch
torch.manual_seed(42)
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

from copy import deepcopy


lemma2ids = defaultdict(list)

main = pd.read_csv('main_no_def.csv')
for row in main.iterrows():
    lemma2ids[row[1]['core_lemma']].append(row[1]['wordnet_id'])
print(len(lemma2ids))



dedupl_ids = set()
for lemma, ids in lemma2ids.items():
    random.shuffle(ids)
    dedupl_ids.add(ids[0])


transform = transforms.Compose([
    transforms.ToTensor()  # Converts the image to float and normalizes it to [0, 1]
])


# подгружаем картинки
datasets = [
 'DeepFloyd_IF-I-XL-v1.0',
 'kandinsky-community_kandinsky-3',]
#  'PixArt-alpha_PixArt-Sigma-XL-2-512-MS',
#  'playgroundai_playground-v2.5-1024px-aesthetic',
#  'prompthero_openjourney',
#  'runwayml_stable-diffusion-v1-5',
#  'stabilityai_sdxl-turbo',
#  'stabilityai_stable-diffusion-3-medium-diffusers',
#  'Tencent-Hunyuan_HunyuanDiT-v1.2-Diffusers',
#  'stabilityai_stable-diffusion-xl-base-1.0',
# ]

data = dict()
is_mean = dict()
is_std = dict()
results = pd.DataFrame(columns=['metric', 'dataset', 'mean', 'std'])

for dataset in datasets:
    inception = InceptionScore(normalize=True)
    gen_image_paths = []
    for x in os.listdir('images/' + dataset):
        if int(x.split('.')[0]) in dedupl_ids:
            gen_image_paths.append('images/' + dataset +'/'+ x)

    for i in tqdm(range(len(gen_image_paths)//3)):
        gen_images = []
        for path in gen_image_paths[i*3:i*3+3]:
            gen_images.append(transform(Image.open(path).convert("RGB")))
        inception.update(torch.stack(gen_images, dim=0))
    inc = inception.compute()
    print('inception', dataset, inc)
    results.loc[len(results)] = ['inception', dataset, float(inc[0]), float(inc[1])]
    results.to_csv('fid_is.csv')
    gc.collect()

del inc
del inception
gc.collect()

retrieval_path = 'images/wikiCommonsOutput/'
real_images = [] # deduplicated
for path in tqdm(os.listdir(retrieval_path)):
    if int(path.split('/')[-1].split('.')[0]) in dedupl_ids:
        try:
            real_images.append(transform(Image.open(retrieval_path+path).convert("RGB")))
        except:
            print(path)
torch.save(real_images, 'real_images_no_duble_tensors.pkl')
gc.collect()

inception = InceptionScore(normalize=True)
for ri in tqdm(real_images):
    inception.update(torch.stack([ri], dim=0)) # это потому что все картинки разного размера
inc = inception.compute()
results.loc[len(results)] = ['inception', 'retrieval', float(inc[0]), float(inc[1])]
print('inception', 'retrieval', inc)
gc.collect()

del inc
del inception
gc.collect()


# БЕЗ ПОВТОРОВ 
real_images = torch.load('real_images_no_duble_tensors.pkl')
fid = FrechetInceptionDistance(normalize=True)
for ri in tqdm(real_images):
    fid.update(torch.stack([ri], dim=0), real=True)

for dataset in datasets:
    new_fid = deepcopy(fid)
    gen_image_paths = []
    for x in os.listdir('images/' + dataset):
        if int(x.split('.')[0]) in dedupl_ids:
            gen_image_paths.append('images/' + dataset +'/'+ x)

    for i in tqdm(range(len(gen_image_paths)//3)):
        gen_images = []
        for path in gen_image_paths[i*3:i*3+3]:
            gen_images.append(transform(Image.open(path).convert("RGB")))
        new_fid.update(torch.stack(gen_images, dim=0), real=False)
    new_fid_res = new_fid.compute()
    print('fid_no_double', dataset, new_fid_res)
    results.loc[len(results)] = ['fid_no_double', dataset, float(new_fid_res), None]
    results.to_csv('fid_is.csv')
    gc.collect()

del fid
del new_fid
gc.collect()

# С ПОВТОРАМИ
retrieval_path = 'images/wikiCommonsOutput/'
dupl_real_images = [] # deduplicated
for path in tqdm(os.listdir(retrieval_path)):
    try:
        dupl_real_images.append(transform(Image.open(retrieval_path+path).convert("RGB")))
    except:
        print(path)
torch.save(dupl_real_images, 'real_images_duble_tensors.pkl')

#real_images = torch.load('real_images_duble_tensors.pkl')
fid = FrechetInceptionDistance(normalize=True)
for ri in tqdm(dupl_real_images):
    fid.update(torch.stack([ri], dim=0), real=True)


for dataset in datasets:
    new_fid = deepcopy(fid)
    gen_image_paths = []
    for x in os.listdir('images/' + dataset):
        gen_image_paths.append('images/' + dataset +'/'+ x)

    for i in tqdm(range(len(gen_image_paths)//3)):
        gen_images = []
        for path in gen_image_paths[i*3:i*3+3]:
            gen_images.append(transform(Image.open(path).convert("RGB")))
        new_fid.update(torch.stack(gen_images, dim=0), real=False)
    new_fid_res = new_fid.compute()
    print('fid_double', dataset, new_fid_res)
    results.loc[len(results)] = ['fid_double', dataset, float(new_fid_res), None]
    results.to_csv('fid_is.csv')
    gc.collect()

results.to_csv('fid_is.csv')
print('the end!')