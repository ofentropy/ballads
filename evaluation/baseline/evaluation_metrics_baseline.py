import numpy as np
import json
import requests
import re
import itertools
import pronouncing
import nltk
from nltk.corpus import cmudict
import pronouncing

d = cmudict.dict()
pronouncing.init_cmu()


def json_parser(ballads_json_file_link):
  ballads = {}
  related = {}
  response = requests.get(ballads_json_file_link)
  ballads_dataset = json.loads(response.text)
  for ballad in ballads_dataset:
    id = ballad.get("url")
    text = ballad.get("poem")
    related_words = ballad.get("related")
    ballads[id] = text
    related[id] = related_words
  return ballads, related


#################################################################
#EVALUATING SYLLABLE COUNTS

def get_syllables(word):
  word = word.lower()
  entries = d.get(word)
  if not entries:
    return [[]]
  return entries


def get_syllable_count(word):
  lowercase = word.lower()
  if lowercase not in d:  
    return -1 
  else:
     return max([len([y for y in x if y[-1].isdigit()]) for x in d[lowercase]])


def eval_syllables(ballad, exact=True):
  last_words = []
  n_lines = len(ballad)
  syllable_success = 0
  for line in ballad:
    line_syllable_count = 0
    words = re.findall(r'[\w]+', line)
    for word in words:
      if get_syllable_count(word) >= 0:
        line_syllable_count += get_syllable_count(word)
    if exact:
      if line_syllable_count == 14: #check if line has 14 syllables
        syllable_success += 1
    else:
      if line_syllable_count >= 10 and line_syllable_count <= 14: #check if line has 14 syllables
        syllable_success += 1
  return syllable_success/n_lines

def eval_syllables_across_ballads(ballads, exact=True):
  n = len(ballads.keys())
  average_syllable_score = 0
  for id in ballads.keys():
    ballad = ballads[id]
    average_syllable_score += eval_syllables(ballad, exact)
  average_syllable_score /= n
  return average_syllable_score

#################################################################
#EVALUATING RHYME SCHEME

def rhymes(word):
  """
  Taken from pronouncing module and modified
  Original: https://pypi.org/project/pronouncing/
  Github: https://github.com/aparrish/pronouncingpy
  Modified to include our get_syllabes()
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

def eval_follows_rhyme_scheme(ballads):
  #percentage of ballads that follow the right rhyme scheme
  successes1 = 0
  successes2 = 0
  successes3 = 0
  successes4 = 0
  successes = 0

  n = len(ballads.keys())
  for id in ballads.keys():
    ballad = ballads[id]
    last_words = []
    for line in ballad:
      words =  re.findall(r'[\w]+', line)
      if(len(words)!= 0):
        last_words.append(words[-1])
      else:
        last_words.append("")
    if (do_they_rhyme(last_words[0], last_words[2]) and do_they_rhyme(last_words[1], last_words[3])):
      successes1 += 1
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
  return (successes1/n, successes2/n, successes3/n, successes4/n, successes/n)

##################################################################
#ConceptNet Score

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
  num_words = 0
  ballad_score = 0
  print(len(ballad))
  for line in ballad:
    # print(line)
    words =  re.findall(r'[\w]+', line)
    # num_words += len(words)
    for word in words:
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

########################
