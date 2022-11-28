import sys
sys.path.append("/home/ubuntu/ballads") # change if necessary

import re
import nltk

nltk.download("brown")
nltk.download('universal_tagset')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import wordnet as wn, brown
from nltk.metrics.distance import edit_distance

from balladsutil.parser import *
import random

correct_words = set(brown.words())
# source: https://stackoverflow.com/a/44383998
wordtags = nltk.ConditionalFreqDist((w.lower(), t) 
        for w, t in brown.tagged_words(tagset="universal"))

def make_quatrains_and_training_prompts(ballad):
    rhymes_so_far = 0
    rhyming_lines = []
    ballad_text = ballad["text"]
    ballad_text = re.sub("[0-9]","",ballad_text)
    ballad_text = re.sub("\\b[Ii]le", "I'll", ballad_text)
    ballad_lines = ballad_text.split("\n")
    quatrains = []
    for line_index, line in enumerate(ballad_lines):
        if len(ballad_lines) >= line_index + 2:
            cleared_line_1 = re.sub(r'[^A-Za-z ]+', '', line)
            cleared_line_2 = re.sub(r'[^A-Za-z ]+', '', ballad_lines[line_index+1])
            if cleared_line_1[-2:] == cleared_line_2[-2:]:
                rhymes_so_far += 1
                rhyming_lines.extend([line.strip(), ballad_lines[line_index+1].strip()])
            if rhymes_so_far >= 2:
                quatrains.append(rhyming_lines)
                rhyming_lines = []
                rhymes_so_far = 0
    
    corrected_ballad = correct_and_normalize(ballad)
    training_prompts = []
    for i in range(len(quatrains)):
        training_prompt = generate_training_prompt(corrected_ballad)
        for line in quatrains[i]:
            training_prompt += line + "\n"
        training_prompts.append(training_prompt)
    
    return quatrains, training_prompts


def generate_training_prompt(poem, n_words = 3):
    temp_adjs, temp_objects, temp_scenes = choose_random_words(poem, n_words)
    adjs = ["sentiments:"] + get_synonyms(temp_adjs, "ADJ")
    objects = ["objects:"] + get_synonyms(temp_objects, "NOUN")
    scenes = ["scenes:"] + get_synonyms(temp_scenes, "NOUN")

    training_prompt = ""
    selections = [objects, scenes, adjs]
    for group in selections:
        for word in group:
            training_prompt += word + " "
        training_prompt += "\n"
    training_prompt += "ballad: "
    
    return training_prompt

def get_synonyms(words, pos):
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
                    if pos in wordtags[syn]:
                        synonyms.append(syn)
        choice = random.sample(synonyms, 1)[0] # randomly pick one synonym
        synonym_words.append(choice)
    return synonym_words


def choose_random_words(poem, n_words = 3):
    adjs, nouns = get_adjs_and_nouns(poem)
    objects, scenes = split_object_scene(nouns)

    temp_adjs, temp_objects, temp_scenes = None, None, None
    if len(adjs) <= n_words:
        print(f"WARNING: Less than or exactly {n_words} adjectives in given poem.")
        temp_adjs = adjs
    if len(objects) <= n_words:
        print(f"WARNING: Less than or exactly {n_words} object nouns in given poem.")
        temp_objects = objects
    if len(scenes) <= n_words:
        print(f"WARNING: Less than or exactly {n_words} scene nouns in given poem.")
        temp_scenes = scenes
    
    temp = [(temp_adjs, adjs), (temp_objects, objects), (temp_scenes, scenes)]
    chosen = []
    for choice, source in temp:
        if choice is None and len(source) >= n_words:
            choice = random.sample(source, n_words)
        chosen.append(choice)
    
    return chosen


def correct_and_normalize(s):
    """
    :param s: - string (poem)

    :return: 'corrected' poem (spellings adjusted) for random sampling
    """
    new_s = ""
    for ch in unidecode(s):
        if ch.isalpha() or ch.isspace():
            new_s += ch
    all_words = (" ".join(new_s.split("\n"))).split(" ")
    ret = []
    for word in all_words:
        if word.lower() not in correct_words:
            # source: https://www.geeksforgeeks.org/correcting-words-using-nltk-in-python/
            temp = [(edit_distance(word, w),w) for w in correct_words if w[0]==word[0]]
            ret.append(sorted(temp, key = lambda val:val[0])[0][1])
            # print(f"incorrect: {word}, correct: {sorted(temp, key = lambda val:val[0])[0][1]}")
        else:
            ret.append(word)
    
    return " ".join(ret).lower()