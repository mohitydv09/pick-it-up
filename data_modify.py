import os
import subprocess
import numpy as np
import cv2
import json

subprocess.run("rm -rf data_cleared", shell=True)

os.makedirs('data_cleared', exist_ok=True)

meta_bagpack = {
    'ground_truth_object': 'backpack',
    'language_correct': 'the grey backpack',
    'language_noisy': 'the bag over there',
    'language_none': 'that object'
}
meta_bottle = {
    'ground_truth_object': 'bottle',
    'language_correct': 'the yellow bottle over there',
    'language_noisy': 'water container over there',
    'language_none': 'that thing'
}
meta_cup = {
    'ground_truth_object': 'cup',
    'language_correct': 'coffee cup over the counter',
    'language_noisy': 'the coffee mug over there',
    'language_none': 'that'
}
meta_wine_glass_top = {
    'ground_truth_object': 'wine glass on shelf',
    'language_correct': 'wine glass over the black shelf',
    'language_noisy': 'the glass over shelf',
    'language_none': 'pick it up'
}
meta_wine_glass_bottom = {
    'ground_truth_object': 'wine glass on table',
    'language_correct': 'wine glass over the table',
    'language_noisy': 'the glass over table',
    'language_none': 'pick it up'
}
meta_remote = {
    'ground_truth_object': 'remote',
    'language_correct': 'TV remote over the table',
    'language_noisy': 'the controller over there',
    'language_none': 'that'
}
meta_bowl = {
    'ground_truth_object': 'bowl',
    'language_correct': 'the bowl over the table',
    'language_noisy': 'the container over there',
    'language_none': 'that'
}
meta_chair = {
    'ground_truth_object': 'chair',
    'language_correct': 'the white chair over their',
    'language_noisy': 'the white thing',
    'language_none': 'that'
}

i = 0
for filename in os.listdir('data'):
    if 'depth' in filename:
        continue
    
    ## Copy the images and depth images to the new directory
    foldername = f"data_{i}"
    os.makedirs(os.path.join('data_cleared', foldername), exist_ok=True)

    subprocess.run(f"cp data/{filename} data_cleared/{foldername}/rgb.png", shell=True)
    subprocess.run(f"cp data/{filename.replace('color', 'depth')} data_cleared/{foldername}/depth.png", shell=True)

    gt_object = filename.split('_')[1]
    if gt_object == 'wine':
        gt_object = 'wine_glass_' + filename.split('_')[3]
        
    ## Save the metadata to a json file
    metadata = locals()[f'meta_{gt_object}']
    with open(os.path.join('data_cleared', foldername, 'meta.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    i += 1