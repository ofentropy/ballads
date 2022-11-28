import sys
sys.path.append("/home/ubuntu/ballads") # change if necessary

import re
import nltk

import pronouncing
pronouncing.init_cmu()

nltk.download("brown")
nltk.download('universal_tagset')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import wordnet as wn, brown
from nltk.metrics.distance import edit_distance

from balladsutil.parser import *
from evaluation.evaluation_metrics import *
import random

correct_words = set(brown.words())
# source: https://stackoverflow.com/a/44383998
wordtags = nltk.ConditionalFreqDist((w.lower(), t) 
        for w, t in brown.tagged_words(tagset="universal"))


def make_quatrains_for_single_ballad(ballad, pattern):
    """
    :param ballad: dict {"text": string}
    :param pattern: string - either "AABB", "ABAB", "ABAC" or "ABCB"
    :return: list of quatrains fitting pattern
    """

    assert pattern in ["AABB", "ABAB", "ABAC", "ABCB"]

    rhymes_so_far = 0
    rhyming_lines = []
    ballad_text = ballad["text"]
    ballad_text = re.sub("[0-9]","",ballad_text)
    ballad_text = re.sub("\\b[Ii]le", "I'll", ballad_text)
    ballad_lines = ballad_text.split("\n")
    quatrains = []
    line_count = len(ballad_lines)
    for line_index, line in enumerate(ballad_lines):
        if pattern == "AABB":
            if line_count >= line_index + 2:
                cleared_line_1 = re.sub(r'[^A-Za-z ]+', '', line).split()
                cleared_line_2 = re.sub(r'[^A-Za-z ]+', '', ballad_lines[line_index+1]).split()
                if len(cleared_line_1) > 0 and len(cleared_line_2) > 0:
                    if do_they_rhyme(cleared_line_1[-1].lower(), cleared_line_2[-1].lower()):
                        rhymes_so_far += 1
                        rhyming_lines.extend([line.strip(), ballad_lines[line_index+1].strip()])
                    if rhymes_so_far >= 2:
                        quatrains.append(rhyming_lines)
                        rhyming_lines = []
                        rhymes_so_far = 0
        elif pattern == "ABAB" or pattern == "ABAC" or pattern == "ABCB":
            if line_count >= line_index + 4:
                cleared_line_1 = re.sub(r'[^A-Za-z ]+', '', line).split()
                cleared_line_2 = re.sub(r'[^A-Za-z ]+', '', ballad_lines[line_index+1]).split()
                cleared_line_3 = re.sub(r'[^A-Za-z ]+', '', ballad_lines[line_index+2]).split()
                cleared_line_4 = re.sub(r'[^A-Za-z ]+', '', ballad_lines[line_index+3]).split()
                if len(cleared_line_1) > 0 and len(cleared_line_2) > 0 \
                and len(cleared_line_3) > 0 and len(cleared_line_4) > 0:
                    AA = BB = False
                    if do_they_rhyme(cleared_line_1[-1].lower(), cleared_line_3[-1].lower()):
                        # checks if 1st and 3rd line rhyme
                        rhymes_so_far += 1
                        AA = True
                    if do_they_rhyme(cleared_line_2[-1].lower(), cleared_line_4[-1].lower()):
                        # checks if 2nd and 4th line rhyme
                        rhymes_so_far += 1
                        BB = True
                    if (pattern == "ABAB" and AA and BB and rhymes_so_far >= 2) \
                        or (pattern == "ABAC" and AA and rhymes_so_far >= 1) \
                        or (pattern == "ABCB" and BB and rhymes_so_far >= 1):
                            rhyming_lines = [l.strip() for l in ballad_lines[line_index:line_index+4]]
                            quatrains.append(rhyming_lines)
                            rhyming_lines = []
                            rhymes_so_far = 0
    return quatrains


def make_quatrains_and_prompts(ballad, patterns=["ABCB"]):
    """
    :param ballad: dict {"text": string}
    :param pattern: list of strings - must be subset of ["AABB", "ABAB", "ABAC", "ABCB"]
    :return: list of quatrains fitting pattern, list of prompts per quatrain
    """
    ballad_text = ballad["text"]
    
    quatrains = []
    for pattern in patterns:
        assert pattern in ["AABB", "ABAB", "ABAC", "ABCB"]
        quatrains += make_quatrains_for_single_ballad(ballad, pattern)
    
    corrected_ballad = correct_and_normalize(ballad_text)
    temp_adjs, temp_objects, temp_scenes = choose_random_words(corrected_ballad)
    prompts = []
    for i in range(len(quatrains)):
        prompt = generate_training_prompt_from_given(temp_adjs, temp_objects, temp_scenes)
        for line in quatrains[i]:
            prompt += line + "\n"
        prompts.append(prompt)
    
    return quatrains, prompts


def generate_training_prompt_from_given(temp_adjs, temp_objects, temp_scenes):
    adjs = ["sentiments:"] + get_synonyms(temp_adjs, "ADJ")
    objects = ["objects:"] + get_synonyms(temp_objects, "NOUN")
    scenes = ["scenes:"] + get_synonyms(temp_scenes, "NOUN")

    prompt = ""
    selections = [objects, scenes, adjs]
    for group in selections:
        for word in group:
            prompt += word + " "
        prompt += "\n"
    prompt += "ballad:\n"
    
    return prompt


def generate_training_prompt(poem, n_words = 3):
    temp_adjs, temp_objects, temp_scenes = choose_random_words(poem, n_words)
    return generate_training_prompt_from_given(temp_adjs, temp_objects, temp_scenes)


def get_synonyms(words, pos):
    """
    :param words: list of str
    :param pos: str that indicates part of speech, "NOUN" or "ADJ"
    """

    assert pos in ["ADJ", "NOUN"]

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