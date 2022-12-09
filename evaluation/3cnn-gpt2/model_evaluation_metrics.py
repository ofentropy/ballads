import numpy as np
import re
import nltk
from nltk.corpus import cmudict
from nltk.corpus import wordnet as wn
from unidecode import unidecode
import spacy
import pronouncing
import json
import requests
from tokenize import tokenize
import itertools
from collections import Counter

d = cmudict.dict()
pronouncing.init_cmu()

#PARSING JSON FILE
def json_parser(ballads_json_file_link):
  """
  Takes a json file link as input. Fetches the generated 
  ballads json file and returns a dictionary of ballads, 
  extracted keywords associated with ballad image and prompt rhymes.
  :param ballads_json_file_link: publicly accessible link to json file with generated poems.
  :returns: ballads, related, rhymes dictionaries
  """
  ballads = {}
  related = {}
  rhymes = {}
  # ballads_json_url = ballads_json_file_link
  response = requests.get(ballads_json_file_link)
  ballads_dataset = json.loads(response.text)
  for ballad in ballads_dataset:
    id = ballad.get("url")
    text = ballad.get("poem")
    related_words_dict = ballad.get("img_labels")
    cur_rhymes = ballad.get("rhymes")
    related_words = {}
    for key in related_words_dict.keys():
      if len(related_words)==0:
        related_words = set(related_words_dict[key])
      else:
        related_words.update(related_words_dict[key])
    ballads[id] = text
    related[id] = related_words
    rhymes[id] = cur_rhymes
  return ballads, related, rhymes


#EVALUATING SYLLABLE COUNTS

def get_syllables(word):
  """
  Takes word as input. Returns the word's syllables.
  """
  word = word.lower()
  entries = d.get(word)
  if not entries:
    # entries = [phoney.predict(word).split(" ")]
    return [[]]
    # entries = [phoney.predict(word).split(" ")]
  return entries

def get_syllable_count(word):
  """
  Takes word as input. Returns the word's syllable count.
  """
  lowercase = word.lower()
  # print(lowercase)
  if lowercase not in d:
    # entries = [phoney.predict(word).split(" ")]
    # print(lowercase)
    return -1 #problem - what to do if not in dict?
  else:
     return max([len([y for y in x if y[-1].isdigit()]) for x in d[lowercase]])


def eval_syllables(ballad):
  """
  Takes a ballad as input. 
  Returns the rate of lines following the 
  syllable scheme (10-14 syllables) to the total number of lines.
  """
  last_words = []
  n_lines = len(ballad)
  syllable_success = 0
  lines = ballad.split("\n")
  for line in lines:
    line_syllable_count = 0
    words = re.findall(r'[\w]+', line)
    # words =  line.split() #get rid of the punctuations
    for word in words:
      if get_syllable_count(word) >= 0:
        line_syllable_count += get_syllable_count(word)
      
    # print(line_syllable_count)
    if line_syllable_count <=14 and line_syllable_count >= 10: #check if line has 14 syllables
      # print(line)
      syllable_success += 1
  return syllable_success/n_lines


def eval_syllables_across_ballads(ballads):
  """
  Takes dictionary of ballads as input.
  Returns the average syllable rate across all ballads.
  """
  n = len(ballads.keys())
  print(n)
  average_syllable_score = 0
  for id in ballads.keys():
    ballad = ballads[id]
    average_syllable_score += eval_syllables(ballad)
  average_syllable_score /= n
  return average_syllable_score


#EVALUATING RHYME SCHEME
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

def lowercase(l):
  return [x.lower() for x in l]

def do_they_rhyme(word1, word2):
  """
  referenced: https://stackoverflow.com/a/25714769
  """
  return word1 in rhymes(word2)

def eval_rhymes_within_ballad(ballad): #ABAB
  last_words = []
  for line in ballad:
    words =  re.findall(r'[\w]+', line)
    last_words.append(words[-1])

  pointer = 2
  successes = 0
  if (do_they_rhyme(last_words[1], last_words[3])): successes +=1
  if (do_they_rhyme(last_words[0], last_words[2])): successes +=1
  
  return successes/2

def eval_rhymes_within_ballad2(ballad): #ABCB
  last_words = []
  for line in ballad:
    words =  re.findall(r'[\w]+', line)
    last_words.append(words[-1])

  pointer = 2
  successes = 0
  if (do_they_rhyme(last_words[1], last_words[3])): successes +=1
  
  return successes/1

def find_prompt_rhyme_scheme(rhymes):
  """
  Identifies the rhyme scheme suggested by the prompt that is
  fed to the GPT-2 model.
  """
  successes1, successes2, successes3, successes4 = 0, 0, 0, 0
  if (do_they_rhyme(rhymes[0], rhymes[2]) and do_they_rhyme(rhymes[1], rhymes[3])): #ABAB
        successes1 += 1
  if (do_they_rhyme(rhymes[1], rhymes[3]) and not do_they_rhyme(rhymes[0], rhymes[2])): #ABCB 
        successes2 += 1
  if (do_they_rhyme(rhymes[0], rhymes[2]) and not do_they_rhyme(rhymes[1], rhymes[3])):#ABAC 
        successes3 += 1
  if (do_they_rhyme(rhymes[0], rhymes[1]) and do_they_rhyme(rhymes[2], rhymes[3])):#AABB
        successes4 += 1
  return successes1, successes2, successes3, successes4

def eval_follows_rhyme_scheme(ballads, rhymes):
  """
  Takes dictionary of ballads as input and their associated rhymes.
  Returns the percentage of ballads following each rhyme scheme out of 
  possible [ABAB, ABCB, ABAC, AABB] and the percentage of ballads 
  containing any rhyme.
  """

  #percentage of ballads that follow the right rhyme scheme
  successes1 = 0
  successes2 = 0
  successes3 = 0
  successes4 = 0
  successes = 0
  rate1, rate2, rate3, rate4 = None, None, None, None

  ABAB, ABCB, ABAC, AABB = 0, 0, 0, 0

  n = len(ballads.keys())
  n1, n2, n3, n4 = 0, 0, 0, 0
  for id in ballads.keys():
    ballad = ballads[id]
    # print(rhymes[id])
    rhyme_scheme = rhymes[id]
    cur_ABAB, cur_ABCB, cur_ABAC, cur_AABB = find_prompt_rhyme_scheme(rhyme_scheme)
    ABAB += cur_ABAB 
    ABCB += cur_ABCB 
    ABAC += cur_ABAC 
    AABB += cur_AABB 

    last_words = []
    lines = ballad.split("\n")
    if len(lines) == 4:
      for line in lines:
        words =  re.findall(r'[\w]+', line)
        # print(words)
        if(len(words)!= 0):
          last_words.append(words[-1].lower())
        else:
          last_words.append("")
      if (do_they_rhyme(last_words[0], last_words[2]) and do_they_rhyme(last_words[1], last_words[3])): #ABAB
        successes1 += 1
      if (do_they_rhyme(last_words[1], last_words[3]) and not do_they_rhyme(last_words[0], last_words[2])): #ABCB 
            successes2 += 1
      if (do_they_rhyme(last_words[0], last_words[2]) and not do_they_rhyme(last_words[1], last_words[3])):#ABAC 
            successes3 += 1
      if (do_they_rhyme(last_words[0], last_words[1]) and do_they_rhyme(last_words[2], last_words[3])):#AABB
            successes4 += 1
      if (do_they_rhyme(last_words[0], last_words[1]) or 
          do_they_rhyme(last_words[0], last_words[2]) or 
          do_they_rhyme(last_words[0], last_words[3]) or 
          do_they_rhyme(last_words[1], last_words[2]) or
          do_they_rhyme(last_words[1], last_words[3]) or 
          do_they_rhyme(last_words[2], last_words[3])): 
        successes += 1

  if ABAB != 0:
    rate1 = successes1/ABAB
  if ABCB != 0:
    rate2 = successes2/ABCB
  if ABAC != 0:
    rate3 = successes3/ABAC
  if AABB != 0:
    rate4 = successes4/AABB
  
  return (successes1/n, successes2/n, successes3/n, successes4/n, successes/n)
  # return (rate1, rate2, rate3, rate4, successes/n)
  
  
#RELATEDNESS EVALUATION - CONCEPTNET SCORE
NOUN_KEY = "NOUN"
ADJ_KEY = "ADJ"
POEM_KEY = "poem"
ID_KEY = "id"
IMG_URL_KEY = "image_url"
ABS_KEY = "abstraction.n.06"

nlp = spacy.load("en_core_web_sm")

def get_adjs_and_nouns(poem, tagset="universal"):
    """
    Given a string with new line characters (poem),
    return list of adjectives and list of nouns
    that are found in the poem
    """
    adjs = []
    nouns = []

    lines = poem.split("\n")
    poem = " ".join(lines)

    doc = nlp(poem)
    for token in doc:
        if token.pos_ == NOUN_KEY:
            nouns.append(token.text)
        elif token.pos_ == ADJ_KEY:
            adjs.append(token.text)
    return [*set(adjs)], [*set(nouns)]

def get_word_from_conceptnet_path(path):
  """
  Takes a ConceptNet path and strips the prefix
  to return the corresponding word
  @param path - str
  @return str
  """
  prefix = "/c/en/"
  prefix_len = len(prefix)
  if len(path) < prefix_len or path[:prefix_len] != prefix:
    print(f"{path} is not a valid ConceptNet path!")
    return path
  else:
    return path[prefix_len:]

def get_conceptnet_path_from_word(word, verify=False):
  """
  Takes a string and appends the ConceptNet prefix
  Assumes word is in English
  If verify is false, assumes that the word exists in ConceptNet
  If verify is true, will make an API call to ConceptNet
  for the word path
  @param word - str
  @param verify - bool
  @return str
  """
  if not verify:
    prefix = "/c/en/"
    return prefix + word
  else:
    related_api = "http://api.conceptnet.io/uri"
    params = {"language":"en", "text":word}
    response_raw = requests.get(related_api, params=params)
    if response_raw.status_code == 200:
      response = json.loads(response_raw.text)
      path = response.get("@id")
      return path
    else: return None

def get_relatedness_score(path1, path2):
  """
  Makes an API call to ConceptNet to find the relatedness score
  of the words in path1 and path2
  Assumes that path1 and path2 are properly formatted ConceptNet paths
  
  @param path1 - str
  @param path2 - str
  @return int
  """
  query = "http://api.conceptnet.io/relatedness?node1=" + path1 + "&node2=" + path2
  response_raw = requests.get(query)
  if response_raw.status_code == 200:
    response = json.loads(response_raw.text)
    score = response.get("value")
    return score
  else: return None #maybe not include in the count if none
  # else: return 0


def get_ballad_relatedness_score(ballad, keywords):
  """
  Takes ballad and its keywords as input. Returns its Conceptnet
  relatedness score.
  """
  num_words = 0
  ballad_score = 0
  adjs, nouns = get_adjs_and_nouns(ballad, tagset="universal")
  adjs.extend(nouns)
  for word in adjs:
  #get scores for word and average
    word_score = 0
    word_path = get_conceptnet_path_from_word(word, verify=False)
    keyword_num = 0
    for keyword in keywords:
      keyword_path = get_conceptnet_path_from_word(keyword, verify=False)
      score = get_relatedness_score(word_path, keyword_path)
      if (score != None):
        word_score += score
        keyword_num += 1
        # num_words += 1
    if (keyword_num != 0):
      word_score /= keyword_num
    else:
      word_score = None
    
    #add the word average score
    if (word_score != None):
      ballad_score += word_score
      num_words += 1

  #divide the sum by number of words
  if (num_words != 0):
    ballad_score /= num_words
  else:
    ballad_score = None
  return ballad_score


def get_average_ballad_relatedness_score(ballads, ballads_keywords):
  """
  Averages the ballad conceptnet relatedness scores across ballads.
  """
  average_score = 0
  # n = len(ballads.keys())
  n = 0
  for id in ballads.keys():
    # print(id)
    ballad = ballads[id]
    keywords = ballads_keywords[id]
    score = get_ballad_relatedness_score(ballad, keywords)
    if (score != None):
      average_score += score
      n += 1
  average_score /= n
  return average_score
