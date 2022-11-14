import sys
sys.path.append("/home/ubuntu/ballads") # change if necessary
import json
import logging
import os
import random
import re
import shutil
import warnings
from datetime import datetime

import balladmodel
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import tensorflow as tf
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer, TFGPT2LMHeadModel

import datasets
import balladsutil.conceptnetutil as cnutil
import balladsutil.export, balladsutil.split_dictionary
from datasets import Dataset, load_dataset
from imgproc import cnn

logging.getLogger('tensorflow').disabled = True
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print(f"Tensorflow version: {tf.__version__}")
print(f"Transformers version: {transformers.__version__}")
os.chdir('/home/ubuntu')

shutil.unpack_archive("gpt2ballads_params.zip", "gpt2ballads_params", "zip") # unpack the GPT2 weights (1.3GB), it will take some time

# Load GPT2 Model
model, tokenizer = balladmodel.load_model_and_tokenizer(path_to_weights="gpt2ballads_params/gpt2-ep4")

### Get MultiMPoems Dataset for Image Sampling
dataset_url = "https://raw.githubusercontent.com/researchmm/img2poem/master/data/multim_poem.json"
response = requests.get(dataset_url)
multim_poem_dataset = json.loads(response.text)

# set variables
seed = 24425
n_images = 120

# Get image samples
samples = cnn.get_k_images(n_images, seed, multim_poem_dataset) # get n_images images

### KEYWORDS ###
classifications_path = "generated_ballads/classifications.json" # update path if necessary
if os.path.isfile(classifications_path):
    # get stored keywords
    print(f"Found {classifications_path}; loading classifications from json...")
    f = open(classifications_path, "r")
    img_to_keywords, img_to_urls = json.loads(f.read())
    f.close()
else:
    # Use CNN (InceptionV3) to get keywords
    img_to_keywords, img_to_urls = cnn.get_keywords_for_images(samples)
    # Store 

    if not os.path.exists(classifications_path.split("/")[0]):
        os.makedirs(classifications_path.split("/")[0])
    f = open(classifications_path, "w")
    print(json.dumps([img_to_keywords, img_to_urls], sort_keys=True, indent=4), file=f)
    f.close()

### RELATED TERMS ###
n_terms = 3
min_weight = 0.4
related_path = "generated_ballads/related.json" # update if necessary
if os.path.isfile(related_path):
    # get stored related terms
    print(f"Found {related_path}; loading related terms from json...")
    f = open(related_path, 'r')
    img_to_related = json.loads(f.read())
    f.close()
    
else:
    # use ConceptNet API to retrieve related terms
    img_to_related = {}
    for id, kw in img_to_keywords.items():
        related_raw = cnutil.get_n_related_terms_raw_from_word(kw[0], n_terms, min_weight)
        related = cnutil.convert_related_raw_to_words(related_raw)
        img_to_related[id] = related

    # store
    if not os.path.exists(related_path.split("/")[0]):
        os.makedirs(related_path.split("/")[0])
    f = open(related_path, "w")
    print(json.dumps(img_to_related, sort_keys=True, indent=4), file=f)
    f.close()

### SPLIT PROMPTS INTO MANAGEABLE CHUNKS ###
list_of_chunks = balladsutil.split_dictionary.split(img_to_related, 10)

### GENERATE POEMS ###
img_to_poems = {}
for i in range(len(list_of_chunks)):
    save_path = f"chunk{i}.json"
    chunk = list_of_chunks[i]
    chunk_img_to_poems = {}
    for id, related in chunk.items():
        poem = balladmodel.generate_ballad_lines(model, tokenizer, related)
        img_to_poems[id] = poem
        chunk_img_to_poems[id] = poem
        print(f"Generated poem for image {id}")
    chunk_file = open(save_path, 'w')
    print(json.dumps(chunk_img_to_poems, sort_keys=True, indent=4), file=chunk_file)
    chunk_file.close()
    print(f"Finished with {i+1}-th chunk!")

### EXPORT ##
path = "generated_ballads/generated_ballads.json"
balladsutil.export.export_ballads(path, img_to_poems, img_to_keywords, img_to_related, img_to_urls)

### ACCESS SAVED CHUNKS IF NEEDED ###
#img_to_poems = {}
#for i in range(n):
#    path = f"generated_ballads/chunk{i}.json"
#    f = open(path, 'r')
#    img_to_poems.update(json.loads(f.read()))
#    f.close()
