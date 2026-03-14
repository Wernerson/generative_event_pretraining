#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import clip
from PIL import Image
import glob
import argparse
from tqdm import tqdm

device = "cuda:6" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root='/data/storage/jianwen/cache/clip')

def extract(args):  
    subfolders = sorted(os.listdir(args.source_dir))
    for subfolder in subfolders:
        print(f"Processing subfolder: {subfolder}")
        image_dir = os.path.join(args.source_dir, subfolder, "images/left/wrapped")
        if os.path.exists(image_dir):
            image_names = sorted(glob.glob(os.path.join(image_dir, "*.png")))
            for image_name in tqdm(image_names):
                image = preprocess(Image.open(image_name)).unsqueeze(0).to(device)
                feature = model.encode_image(image).unsqueeze(0).to("cpu")
                torch.save(feature, image_name.replace(".png", ".pt"))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--source_dir", default = None, type = str)
    
    args = parser.parse_args()
    extract(args)         
        
    