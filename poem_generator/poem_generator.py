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


#BEGINNING OF UTILS

d = cmudict.dict()
pronouncing.init_cmu()

def get_syllables(word):
  word = word.lower()
  entries = d.get(word)
  if not entries:
    return [[]]
  return entries

def rhymes(word):
  """
  Taken from pronouncing module and modified
  Get words rhyming with a given word.
  This function may return an empty list if no rhyming words are found in
  the dictionary, or if the word you pass to the function is itself not
  found in the dictionary.
  .. doctest::
      >>> import pronouncing
      >>> pronouncing.rhymes("conditioner")
      ['commissioner', 'parishioner', 'petitioner', 'practitioner']
  :param word: a word
  :returns: a list of rhyming words
  """
  phones = get_syllables(word)
  combined_rhymes = []
  if phones:
      for element in phones:
        element = " ".join(element)
        combined_rhymes.append([w for w in pronouncing.rhyme_lookup.get(pronouncing.rhyming_part(
                                  element), []) if w != word])
      combined_rhymes = list(itertools.chain.from_iterable(combined_rhymes))
      unique_combined_rhymes = sorted(set(combined_rhymes))
      return unique_combined_rhymes
  else:
      return []

def get_rhymes(word):
  return rhymes(word)

def do_they_rhyme(word1, word2):
  """
  referenced: https://stackoverflow.com/a/25714769
  """
  if word1 == word2:
    return True
  return word1 in rhymes(word2)

def get_synonyms(words):
    """
    :param words: list of str
    :param pos: str that indicates part of speech, "NOUN" or "ADJ"
    """

    # assert pos in ["ADJ", "NOUN"]

    synonym_words = []
    if words is None:
      return synonym_words
    for word in words:
        word_synsets = wn.synsets(word)
        synonyms = []
        for syn in word_synsets:
            for i in syn.lemmas():
                syn = i.name()
                # skip terms/phrases with more than 1 word
                if "_" not in syn:
                    # make sure word can be proper part of speech
                    # if pos in wordtags[syn]:
                    synonyms.append(syn)
        if len(synonyms) > 0:
            choice = random.sample(synonyms, 1)[0] # randomly pick one synonym
            synonym_words.append(choice)
    return synonym_words


def generate_rhymes(keywords):
  # rhymes=[]
  rhymes = {}

  #extract keywords
  objects, scenes, sentiments = keywords["objects"], keywords["scenes"],  keywords["sentiments"]

  #generate synonyms to keywords
  objects_syn, scenes_syn, sentiments_syn = get_synonyms(objects), get_synonyms(scenes), get_synonyms(sentiments)

  words = [objects, scenes, sentiments, objects_syn, scenes_syn, sentiments_syn]
  
  #get rhymes to the synonyms and keywords
  for group in words:
    for word in group:
      # rhymes.append(get_rhymes(word))
      rhyming_words = get_rhymes(word)
      if len(rhyming_words) > 0:
        rhymes[word] = get_rhymes(word)

  return rhymes


def choose_rhyme_words(chosen_labels):
  rhymes = generate_rhymes(chosen_labels)
  # TO/DO: score rhymes from chosen labels + pick rhymes with highest score
  chosen_rhymes = []
  choices = random.sample(rhymes.keys(), 2) #sorted(rhymes, key=len, reverse=True)[:2]
  # for choice in choices:
  #   chosen_rhymes += random.sample(choice, 2)
  # return chosen_rhymes

  for choice in choices:
    chosen_rhymes.append(choice)
    chosen_rhymes += random.sample(rhymes[choice], 1)
  return chosen_rhymes



def generate_prompt_constraints_from_keywords(keywords):
  #@param: keywords - dictionary of k keywords for "objects", "scenes", "sentiments" 

  prompt = "<|beginoftext|>"
  objects = ["objects:"] + keywords["objects"]
  scenes = ["scenes:"] + keywords["scenes"]
  sentiments = ["sentiments:"] + keywords["sentiments"]

  #generate rhymes for keywords
  rhymes = ["rhymes:"] + choose_rhyme_words(keywords)

  constraint_str = " ".join(rhymes[1:])
  print(constraint_str)
  constraint_token_ids = tokenizer.encode(constraint_str)
  constraints = [PhrasalConstraint(token_ids=constraint_token_ids)]

  selections = [objects, scenes, sentiments, rhymes]
  for group in selections:
      for word in group:
          prompt += word + " "
      prompt += "\n"
  prompt += "ballad:\n"
  
  return prompt, constraints


def generate_poem_from_prompt(model, constraints, tokenizer, prompt):
  input_ids = tokenizer.encode(prompt, return_tensors='tf')
  sample_output = model.generate(input_ids, 
                                do_sample=True, 
                                min_length=50,
                                max_length=MAX_TOKENS, 
                                top_k=50, 
                                top_p=0.95, 
                                num_return_sequences=1, 
                                no_repeat_ngram_size=2,
                                num_beams=5,
                                constraints=constraints)
  generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
  return generated_text.strip()


# ===========================================================


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

#set of images
images = ["https://2.bp.blogspot.com/-EalkznnSmTk/XDyNEsqXP9I/AAAAAAACh4Q/jajYMBgy5U4ElGeAzO6iJHk_h85aib8IwCLcBGAs/s1600/Oscar%2BDroege.jpg"]

# LOAD THE CNNS
objects_path = "/content/drive/MyDrive/CS230 Project/cnns/objects_precision_inceptionv3.h5"
scenes_path = "/content/drive/MyDrive/CS230 Project/cnns/scenes_precision_inceptionv3.h5"
sentiments_path = "/content/drive/MyDrive/CS230 Project/cnns/sentiments_precision_inceptionv3.h5"
obj_model = load_model(objects_path, 550, metrics=[tf.keras.metrics.Precision(top_k = 550)])
sce_model = load_model(scenes_path, 565, metrics=[tf.keras.metrics.Precision(top_k = 565)])
sen_model = load_model(sentiments_path, 322, metrics=[tf.keras.metrics.Precision(top_k = 322)])

common_object_labels = get_labels_from_text("/content/drive/MyDrive/CS230 Project/data/common_object_labels.txt")
common_sentiment_labels = get_labels_from_text("/content/drive/MyDrive/CS230 Project/data/common_sentiment_labels.txt")
common_scene_labels = get_labels_from_text("/content/drive/MyDrive/CS230 Project/data/common_scene_labels.txt")

objects_reverse_lookup, _ = create_labels_lookup(common_object_labels)
sentiments_reverse_lookup, _ = create_labels_lookup(common_sentiment_labels)
scenes_reverse_lookup, _ = create_labels_lookup(common_scene_labels)


# CREATE TRIPLECNN
triple_cnn = TripleCNN(objects_reverse_lookup, scenes_reverse_lookup, sentiments_reverse_lookup, obj_model, sce_model, sen_model)

# GENERATE LABEL FOR ONE IMAGE
# prompt = triple_cnn.generate_labels("https://2.bp.blogspot.com/-EalkznnSmTk/XDyNEsqXP9I/AAAAAAACh4Q/jajYMBgy5U4ElGeAzO6iJHk_h85aib8IwCLcBGAs/s1600/Oscar%2BDroege.jpg", kind="url", verbose=True)
# print(prompt)



#BALLAD GPT2MODEL
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
model.load_weights(f"/content/drive/MyDrive/CS230 Project/models/gpt2-e2-b8800")
# model.summary()


def generate_poems(images, gpt2model, cnnmodel):
  for image_url in images:
    poem = generate_poem_from_image(image_url, gpt2model, cnnmodel)
    # print(poem)

def generate_poem_from_image(image_url, gpt2model, cnnmodel):
  
  #pass through CNN1, CNN2, CNN3 for outputs
  labels = cnnmodel.generate_labels(image_url, kind="url", verbose=True)
  
  #pass outputs into prompt generator
  prompt, constraints = generate_prompt_constraints_from_keywords(labels)
  print(prompt)

  #pass prompt into generate_poem from prompt
  poem = generate_poem_from_prompt(gpt2model, constraints, tokenizer, prompt)
  print(poem)

  
generate_poems(images, model, triple_cnn)


