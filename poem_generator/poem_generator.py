from nltk.corpus import cmudict
import itertools
d = cmudict.dict()

import pronouncing
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

import time
import random
from itertools import combinations
from nltk.corpus import words

do_they_rhyme("game", "fame") # should be True

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
  rhymes=[]

  #extract keywords
  objects, scenes, sentiments = keywords["objects"], keywords["scenes"],  keywords["sentiments"]

  #generate synonyms to keywords
  objects_syn, scenes_syn, sentiments_syn = get_synonyms(objects), get_synonyms(scenes), get_synonyms(sentiments)

  words = [objects, scenes, sentiments, objects_syn, scenes_syn, sentiments_syn]
  
  #get rhymes to the synonyms and keywords
  for group in groups:
    for word in group:
      rhymes.append(get_rhymes(word))

  return rhymes



def generate_prompt_from_keywords(keywords):
  #@param: keywords - dictionary of k keywords for "objects", "scenes", "sentiments" 

  prompt = "<|beginoftext|>"
  objects = ["objects:"] + keywords["objects"]
  scenes = ["scenes:"] + keywords["scenes"]
  sentiments = ["sentiments:"] + keywords["sentiments"]

  #generate rhymes for keywords
  rhymes = ["rhymes:"] + generate_rhymes(keywords)

  selections = [objects, scenes, sentiments, rhymes]
  for group in selections:
      for word in group:
          prompt += word + " "
      prompt += "\n"
  prompt += "ballad:\n"
  
  return prompt


def generate_poem_from_prompt(model, tokenizer, prompt):
  input_ids = tokenizer.encode(prompt, return_tensors='tf')
  sample_output = model.generate(input_ids, do_sample=True, max_length=MAX_TOKENS, top_k=50, top_p=0.95, num_return_sequences=1, no_repeat_ngram_size=2)
  generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
  return generated_text.strip()


===========================================================


#set of images

#pre-processed images
def generate_poems(images):
  for image in images:
    poem = generate_poem_from_image(image)
    print(poem)

def generate_poem_from_image(image):
  #preproscessed image
  #pass through CNN1, CNN2, CNN3 for outputs
  #pass outputs into prompt generator
  #pass prompt into generate_poem from prompt
  poem = generate_poem_from_prompt(model, tokenizer, prompt)
