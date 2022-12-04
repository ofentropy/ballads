import sys
sys.path.append("/home/ubuntu/ballads")
import gpt2.training_prompt_generator # ?
from gpt2.training_prompt_generator import *

import os
import time
import warnings
import re
import random
import datasets
import transformers
import tensorflow as tf
import numpy as np
import seaborn as sns
import string
import math
import requests
import json
import pickle
from os.path import exists
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset, load_dataset
from tqdm import tqdm

np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")

print(f"Tensorflow version: {tf.__version__}")
print(f"Transformers version: {transformers.__version__}")
print("Loading the dataset...")
dataset_url = "https://github.com/mckurz/ballads/raw/main/ballads_data3.json"
response = requests.get(dataset_url)
corpus_data = json.loads(response.text)
#corpus_data = corpus_data[:50]
print(f"Dataset loaded. There are {len(corpus_data)} ballads.") # should be 6597
print("Loading the tokenizer...")
MAX_TOKENS = 128
BOS_TOKEN = "<|beginoftext|>"
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
SAVED_CORRECTED_WORDS_PATH = "correted_words.json"
SAVED_PROMPTS_PATH = "quatrain_prompts.json"

tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2",
    bos_token=BOS_TOKEN,
    eos_token=EOS_TOKEN,
    pad_token=PAD_TOKEN,
    max_length=MAX_TOKENS,
    is_split_into_words=True,
)
print(f"Tokenizer loaded! Size: {len(tokenizer)}.")
encoded_punctuation = [tokenizer.encode(char)[0] for char in string.punctuation]
new_line_token_id = tokenizer.encode("\n")[0]

print("Preparing the prompts...")
train_quatrains = []
train_prompts = []

if exists(SAVED_CORRECTED_WORDS_PATH):
    global CORRECTED_WORDS
    load_corrected_words(SAVED_CORRECTED_WORDS_PATH)
    print(f"Corrected word dictionary loaded. Length: {len(CORRECTED_WORDS)}")
        
if exists(SAVED_PROMPTS_PATH):
    with open(SAVED_PROMPTS_PATH, 'rb') as new_file:
        train_prompts = pickle.load(new_file)
else:    
    for ballad in tqdm(corpus_data):
        quatrains, prompts = make_quatrains_and_prompts_for_single_ballad(ballad, tokenizer, patterns=["AABB", "ABAB", "ABAC", "ABCB"])
        train_quatrains.extend(quatrains)
        train_prompts.extend(prompts)
    
if not exists(SAVED_CORRECTED_WORDS_PATH):
    save_corrected_words(SAVED_CORRECTED_WORDS_PATH)
    print("Corrected word dictionary saved.")
  
if not exists(SAVED_PROMPTS_PATH):
    with open(SAVED_PROMPTS_PATH, 'wb') as new_file:
        pickle.dump(train_prompts, new_file)

print(f"Prompts generated. Total number: {len(train_prompts)}")
print(f"A few sample prompts:")
for prompt in random.sample(train_prompts, k=3):
    print(prompt)
    print()

def tokenize(prompts, tokenizer=tokenizer):
    # Add start and end token to each ballad
    prompts_to_tokenize = []
    for index, prompt in enumerate(prompts):
      prompt = BOS_TOKEN + prompt + EOS_TOKEN
      prompts_to_tokenize.append(prompt)

    output = tokenizer(
        prompts_to_tokenize,
        add_special_tokens=True,
        max_length=MAX_TOKENS,
        truncation=True,
        pad_to_max_length=True,
    )
    output["labels"] = output["input_ids"].copy()

    return output

from dataclasses import dataclass
from typing import List, Optional, Tuple

import tensorflow as tf
from transformers import TFGPT2PreTrainedModel, TFGPT2MainLayer, BatchEncoding
from transformers.modeling_tf_outputs import TFCausalLMOutputWithCrossAttentions
from transformers.modeling_tf_utils import input_processing, TFModelInputType, TFCausalLanguageModelingLoss
from typing import Union

EPOCHS = 10
INITIAL_LEARNING_RATE = 0.0001
DECAY_STEPS = 300
DECAY_RATE = 0.7
BATCH_SIZE = 16
MODEL_TYPE = "gpt2-medium"

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    INITIAL_LEARNING_RATE,
    decay_steps=DECAY_STEPS,
    decay_rate=DECAY_RATE,
    staircase=True)

model = TFGPT2LMHeadModel.from_pretrained(
        MODEL_TYPE,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )

model.resize_token_embeddings(len(tokenizer))
model.tokenizer = tokenizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile()
model.layers[0].vocab_size = len(tokenizer) # something is wrong with TFGPT2 initialization so this is needed
model.summary()

import time
"""
valid_dataset = tokenize(valid_quatrains)
valid_dataset = tf.data.Dataset.from_tensor_slices({"input_ids": tf.convert_to_tensor(valid_dataset["input_ids"]),
                                                      "attention_mask": tf.convert_to_tensor(valid_dataset["attention_mask"])})
valid_dataset = valid_dataset.batch(BATCH_SIZE,drop_remainder=False)
"""

def generate_sample(model, tokenizer, prompt="<|beginoftext|>objects: tree crown\nscenes: coronation palace\nsentiments: happiness glory\nrhymes: victory crown beer gown\nballad:\n"):
  input_ids = tokenizer.encode(prompt, return_tensors='tf')
  sample_output = model.generate(input_ids, do_sample=True, max_length=MAX_TOKENS, top_k=50, top_p=0.95, num_return_sequences=1, no_repeat_ngram_size=2)
  generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
  return generated_text.strip()

train_dataset = tokenize(train_prompts)
train_dataset = tf.data.Dataset.from_tensor_slices({"input_ids": tf.convert_to_tensor(train_dataset["input_ids"]),
                                                    "attention_mask": tf.convert_to_tensor(train_dataset["attention_mask"])})
train_dataset = train_dataset.shuffle(20000).batch(BATCH_SIZE,drop_remainder=False)

for epoch in range(EPOCHS):
  batch_loop = tqdm(train_dataset)
  batch_loop.set_description(f"Epoch {epoch}")
  for batch_index, batch in enumerate(batch_loop):
    with tf.GradientTape() as tape:
      output = model(input_ids=batch["input_ids"], labels=batch["input_ids"], attention_mask=batch["attention_mask"])

    gradients = tape.gradient(output.loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    batch_loop.set_postfix_str(f"GPT loss: {float(output.loss)}")

    if batch_index % 100 == 0 and batch_index != 0:
      generated_ballad = "\n------------------------------\n" + generate_sample(model, tokenizer) + "\n------------------------------\n"
      print(generated_ballad)
      with open("generated_ballads.txt", "a") as f:
          f.write(generated_ballad)
    if batch_index % 400 == 0 and batch_index != 0:
      model.save_weights(f"checkpoints/gpt2-e{epoch}-b{batch_index}")

  """
  val_batch_loop = tqdm(valid_dataset)
  valid_loss = 0
  valid_batch_no = 0
  for batch_index, batch in enumerate(val_batch_loop):
    valid_batch_no += 1
    output = model(input_ids=batch["input_ids"], labels=batch["labels"], attention_mask=batch["attention_mask"], training=False)
    valid_loss += float(output.loss)
  print(f"Validation loss after epoch {epoch}: {valid_loss/valid_batch_no}")
  """
