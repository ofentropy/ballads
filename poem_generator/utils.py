import nltk
from nltk.corpus import wordnet as wn, brown
from nltk.corpus import cmudict
from nltk.corpus import words
from nltk.metrics.distance import edit_distance
import re
import pronouncing
import random
import itertools
from itertools import combinations
import time
import os
from transformers import PhrasalConstraint


d = cmudict.dict()
pronouncing.init_cmu()


def get_syllables(word):
  word = word.lower()
  entries = d.get(word)
  if not entries:
    return [[]]
  return entries

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
  chosen_rhymes = []
  choices = random.sample(rhymes.keys(), 2) #sorted(rhymes, key=len, reverse=True)[:2]
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
  chosen_rhymes = choose_rhyme_words(keywords)
  rhymes = ["rhymes:"] + chosen_rhymes

  constraint_str = " ".join(rhymes[1:])
  constraint_token_ids = tokenizer.encode(constraint_str)
  constraints = [PhrasalConstraint(token_ids=constraint_token_ids)]

  selections = [objects, scenes, sentiments, rhymes]
  for group in selections:
      for word in group:
          prompt += word + " "
      prompt += "\n"
  prompt += "ballad:\n"
  
  return prompt, constraints, chosen_rhymes


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
  generated_text = generated_text.split("ballad:\n")[1]
  return generated_text.strip()


import json
from tqdm.notebook import tqdm
import random


def get_k_labels_from_text(k, seed, labels_path):
    with open(labels_path, 'r') as f:
        data = f.read()
        labels = data.split("\n")
        random.seed(seed)
        k_labels = random.sample(labels, k)

    return k_labels

def generate_poem_from_image(image_url, gpt2model, cnnmodel):
  
  #pass through CNN1, CNN2, CNN3 for outputs
  labels = cnnmodel.generate_labels(image_url, kind="url", verbose=True)
  # img_to_related = {}
  # for id, kw in img_to_keywords.items():
  #   related_raw = get_n_related_terms_raw_from_word(kw[0], n_terms, min_weight)
  #   related = convert_related_raw_to_words(related_raw)
  #   img_to_related[id] = related
  
  #pass outputs into prompt generator
  prompt, constraints, rhymes = generate_prompt_constraints_from_keywords(labels)
  # print(prompt)

  #pass prompt into generate_poem from prompt
  poem = generate_poem_from_prompt(gpt2model, constraints, tokenizer, prompt)
  
  return poem, labels, rhymes

def generate_poems(images, gpt2model, cnnmodel,save_path):
  if not os.path.exists(save_path.split("/")[0]):
    os.makedirs(save_path.split("/")[0])
  json_format = []
  for id, image_url in enumerate(images):
    poem, img_labels,rhymes = generate_poem_from_image(image_url, gpt2model, cnnmodel)
    print("labels: \n", img_labels)
    print("poem: \n", poem)
    url = image_url
    # related_words = img_to_related.get(id)
    # kw = img_labels
    json_format.append({"id": id, "url": url, "img_labels": img_labels, "rhymes":rhymes, #img_labels is like "related" but this is a dictionary
                    "poem": poem})
    # json_format.append({"id": id, "url": url, "keyword": kw[0], 
    #                "related": related_words, "poem": poem})
    
  new_file = open(save_path, 'w')
  print(json.dumps(json_format, sort_keys=True, indent=4), file=new_file)
  new_file.close()