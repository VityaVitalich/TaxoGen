import ImageReward as reward
import pandas as pd
from tqdm import tqdm
import os
import torch
import pickle

def get_path(idx, model):
    global image_path

    path2image = f'{image_path}/{model}/{idx}.png'
    if not f'{idx}.png' in os.listdir(f'{image_path}/{model}/'):
        path2image = f'{image_path}/{model}/{idx}.jpg'

        if not f'{idx}.jpg' in os.listdir(f'{image_path}/{model}/'):
            path2image = f'{image_path}/{model}/{idx}.gif'
            if not f'{idx}.gif' in os.listdir(f'{image_path}/{model}/'):
                return None

    return path2image

if __name__ == '__main__':

    model = reward.load("ImageReward-v1.0")
    model = model.to('cuda:0')
    df = pd.read_csv('/home/data/v.moskvoretskii/TaxoGen/main.csv', index_col=0)

    image_path = '/home/data/v.moskvoretskii/taxo_demo_images/images_def'

    models = ['stabilityai_sdxl-turbo',
    'retrieval',
    'runwayml_stable-diffusion-v1-5',
    'Tencent-Hunyuan_HunyuanDiT-v1.2-Diffusers',
    'playgroundai_playground-v2.5-1024px-aesthetic',
    'prompthero_openjourney',
    'kandinsky-community_kandinsky-3',
    'stabilityai_stable-diffusion-xl-base-1.0',
    'PixArt-alpha_PixArt-Sigma-XL-2-512-MS',
    'DeepFloyd_IF-I-XL-v1.0',
    'stabilityai_stable-diffusion-3-medium-diffusers']

    all_rewards = {k: [] for k in models}

    for i, item in tqdm(df.iterrows()):

        idx = item['wordnet_id']
        prompt = item['core_lemma']
        
        for model_name in models:
            path = get_path(idx, model_name)

            if path:
                try:
                    with torch.no_grad():
                        score = model.score(prompt, path)
                except:
                    print(idx)
                    score = None
            else:
                print(idx)
                score = None
            
            all_rewards[model_name].append(score)

    with open('rewards_def.pickle', 'wb') as f:
        pickle.dump(all_rewards, f)
