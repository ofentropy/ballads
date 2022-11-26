import nltk

nltk.download('universal_tagset')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import wordnet as wn
from unidecode import unidecode

NOUN_KEY = "NOUN"
ADJ_KEY = "ADJ"

POEM_KEY = "poem"
ID_KEY = "id"
IMG_URL_KEY = "image_url"

ABS_KEY = "abstraction.n.06"

# GRANULAR 
def get_adjs_and_nouns(poem, tagset="universal"):
    adjs = []
    nouns = []

    lines = poem.split("\n")
    for line in lines:
        words = line.split(" ")

        # use NLTK tagger to get pos for each word
        tagged = nltk.pos_tag(words, tagset=tagset)

        for word, pos in tagged:
            word = unidecode(word)

            if len(word) > 2: # somewhat arbitrary filter
                if pos == NOUN_KEY:
                    nouns.append(word)
                elif pos == ADJ_KEY:
                    adjs.append(word)
    
    return adjs, nouns


def split_object_scene(nouns):
    """
    """
    object_nouns = []
    scene_nouns = []

    for noun in nouns:
        if wn.synsets(noun):
            n = wn.synsets(noun)[0].name()
            split_name = n.split(".")
            pos = split_name[1]
            if pos == "n":
                if abs(len(noun) - len(split_name[0])) <= 2: # somewhat arbitrary filter
                    category = wn.synset(n).hypernym_paths()[0][1].name()
                    if category == ABS_KEY:
                        scene_nouns.append(noun)
                    else:
                        object_nouns.append(noun)

    return object_nouns, scene_nouns


# SPECIFIC DATASET FORMAT

def get_url_adjs_nouns_for_dataset(dataset):
    """
    """
    updated_dataset = []

    for elem in dataset:
        poem = elem[POEM_KEY]
        
        adjs, nouns = get_adjs_and_nouns(poem, tagset="universal")

        new_elem = {
            IMG_URL_KEY: elem[IMG_URL_KEY],
            NOUN_KEY: [*set(nouns)],
            ADJ_KEY: [*set(adjs)],
        }

        updated_dataset.append(new_elem)
    
    return updated_dataset


def split_sentiment_object_scene(nouns_adjs):
    """
    """
    url_to_sentiment_raw = {} # adjs
    url_to_object_raw = {} # physical nouns
    url_to_scene_raw = {} # abstract nouns

    for elem in nouns_adjs:
        url = elem[IMG_URL_KEY]

        # adjectives
        adjs = elem[ADJ_KEY]
        url_to_sentiment_raw[url] = [*set(url_to_sentiment_raw.get(url, []) + adjs)]

        # nouns
        nouns = elem[NOUN_KEY]
        object_nouns, scene_nouns = split_object_scene(nouns)
        url_to_object_raw[url] = [*set(url_to_object_raw.get(url, []) + object_nouns)]
        url_to_scene_raw[url] = [*set(url_to_scene_raw.get(url, []) + scene_nouns)]
    
    return { 
        "sentiment": url_to_sentiment_raw, 
        "object": url_to_object_raw,
        "scene": url_to_scene_raw
        }