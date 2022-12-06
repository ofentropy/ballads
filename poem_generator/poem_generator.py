from dataclasses import dataclass
from typing import List, Optional, Tuple
import random
import time
import tensorflow as tf
import itertools
from itertools import combinations
import re
import nltk
from nltk.corpus import cmudict
from nltk.corpus import wordnet as wn, brown
from nltk.metrics.distance import edit_distance
nltk.download("brown")
nltk.download('universal_tagset')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
import pronouncing


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
from transformers import (TFGPT2LMHeadModel, 
                          GPT2Tokenizer, 
                          PhrasalConstraint)
import tensorflow as tf

from triple_cnn.finetuning import load_model
from triple_cnn.tc_util.importutil import get_labels_from_text
from triple_cnn.tc_processing import create_labels_lookup
from triple_cnn.generator import TripleCNN
from utils import get_k_labels_from_text, generate_poems # maybe needs to be "from utils import *"

# LOAD THE CNNS
objects_path = "/content/drive/MyDrive/CS230 Project/cnns/objects_precision_inceptionv3.h5"
scenes_path = "/content/drive/MyDrive/CS230 Project/cnns/scenes_precision_inceptionv3.h5"
sentiments_path = "/content/drive/MyDrive/CS230 Project/cnns/sentiments_precision_inceptionv3.h5"

common_object_labels = get_labels_from_text("/content/drive/MyDrive/CS230 Project/data/common_object_labels.txt")
common_sentiment_labels = get_labels_from_text("/content/drive/MyDrive/CS230 Project/data/common_sentiment_labels.txt")
common_scene_labels = get_labels_from_text("/content/drive/MyDrive/CS230 Project/data/common_scene_labels.txt")

num_object_labels = len(common_object_labels)
num_sentiment_labels = len(common_sentiment_labels)
num_scene_labels = len(common_scene_labels)

obj_model = load_model(objects_path, num_object_labels, metrics=[tf.keras.metrics.Precision(top_k = num_object_labels)])
sce_model = load_model(scenes_path, num_sentiment_labels, metrics=[tf.keras.metrics.Precision(top_k = num_sentiment_labels)])
sen_model = load_model(sentiments_path, num_scene_labels, metrics=[tf.keras.metrics.Precision(top_k = common_scene_labels)])

objects_reverse_lookup, _ = create_labels_lookup(common_object_labels)
sentiments_reverse_lookup, _ = create_labels_lookup(common_sentiment_labels)
scenes_reverse_lookup, _ = create_labels_lookup(common_scene_labels)


# CREATE TRIPLECNN
triple_cnn = TripleCNN(objects_reverse_lookup, scenes_reverse_lookup, sentiments_reverse_lookup, obj_model, sce_model, sen_model)

#CREATE BALLAD GPT2MODEL
MAX_TOKENS = 128
BOS_TOKEN = "<|beginoftext|>"
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"

tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2",
    bos_token=BOS_TOKEN,
    eos_token=EOS_TOKEN,
    pad_token=PAD_TOKEN,
    max_length=MAX_TOKENS,
    is_split_into_words=True,
)

model = TFGPT2LMHeadModel.from_pretrained(
        "gpt2",
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )

model.resize_token_embeddings(len(tokenizer))
model.tokenizer = tokenizer
model.compile()
model.layers[0].vocab_size = len(tokenizer) # something is wrong with TFGPT2 initialization so this is needed
model.load_weights(f"/content/drive/MyDrive/CS230 Project/models/gpt2-e5-b12000")


path = "sample_data/generated_ballads.json"

object_img_url = "/content/drive/MyDrive/CS230 Project/img_urls/test_urls_object.txt"
scene_img_url = "/content/drive/MyDrive/CS230 Project/img_urls/test_urls_scene.txt"
sentiment_img_url = "/content/drive/MyDrive/CS230 Project/img_urls/test_urls_sentiment.txt"
unseen_img_url = "/content/drive/MyDrive/CS230 Project/img_urls/unseen_urls.txt"

seed = 24425
n_terms = 3
min_weight = 0.4
n_images = 33

img_samples = get_k_labels_from_text(n_images, seed, object_img_url) # get n_images images
scene_samples = get_k_labels_from_text(n_images, seed, scene_img_url)
sentiment_samples = get_k_labels_from_text(n_images, seed, sentiment_img_url)
unseen_samples = get_k_labels_from_text(2, seed, unseen_img_url)

img_samples.extend(scene_samples)
img_samples.extend(sentiment_samples)
img_samples.extend(unseen_samples)




generate_poems(img_samples, model, tokenizer, triple_cnn, path)


