import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import clip
from clipscore import extract_all_images, get_clip_score
import pandas as pd
import json
from tqdm import tqdm
import time
import ast
tqdm.pandas()

model, transform = clip.load("ViT-B/32", device="cuda", jit=False)
model.eval()

def compute_clipscore(wordnet_id, core_lemma, hypernym_synsets, cohyponym_synsets):
    hypernym_synsets, cohyponym_synsets = ast.literal_eval(hypernym_synsets), ast.literal_eval(cohyponym_synsets)
    all_scores = {}
    for image_path in image_paths:
        image_paths_model = [os.path.join(image_path, path) for path in os.listdir(image_path)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff')) and path.startswith(str(wordnet_id))]
        if len(image_paths_model)==0:
            assert os.path.basename(image_path)=='retrieval', image_path
            all_scores[os.path.basename(image_path)] = None
        else:
            try:
                image_feats = extract_all_images(image_paths_model, model, device, batch_size=64, num_workers=8)
                _, per_instance_image_text, _ = get_clip_score(model, image_feats, ["an image of " + core_lemma, *["an image of " + i.split('.')[0] for i in hypernym_synsets], *["an image of " + i.split('.')[0] for i in cohyponym_synsets]], device)
                all_scores[os.path.basename(image_path)] = {"CLIPscore_lemma": per_instance_image_text[0], 
                                                        "CLIPscore_hypernyms": list(per_instance_image_text[1:1+len(hypernym_synsets)]), 
                                                        "CLIPscore_cohyponyms": list(per_instance_image_text[1+len(hypernym_synsets):])}
            except Exception as e:
                print(e)
                all_scores[os.path.basename(image_path)] = None

    return all_scores        

if __name__ == "__main__":
    df = pd.read_csv('../TaxoGen/main.csv')
    image_dir = "/ltstorage/shares/projects/taxollama_images/images"
    device = "cuda"
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir) if not path.startswith("._")]

    df['clipscore'] = df[['wordnet_id', 'core_lemma', 'hypernym_synsets', 'cohyponym_synsets']].progress_apply(lambda x: compute_clipscore(*x), axis=1)
    df.to_csv('../TaxoGen/main_clipscore_imageof.csv', index=None)
