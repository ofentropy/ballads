# replace the "pronouncing.init_cmu()" call with the below 
# and change pronouncing.rhyme_lookup.get on line 34 to rhyme_lookup.get

import pronouncing
import collections
from os.path import exists
import json 
from transformers import GPT2Tokenizer

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

GPT2_VOCAB = []
gpt2_vocab_path = "/content/gpt2_vocab.json" # change if necessary
if exists(gpt2_vocab_path):
  f = open(gpt2_vocab_path, "r")
  GPT2_VOCAB = json.loads(f.read())
  f.close()
else:
  GPT2_VOCAB = list(tokenizer.encoder.keys())
  f = open(gpt2_vocab_path, "w")
  print(json.dumps(GPT2_VOCAB), file=f)
  f.close()

def init_cmu(filehandle=None):
    """Initialize the module's pronunciation data.
    This function is called automatically the first time you attempt to use
    another function in the library that requires loading the pronunciation
    data from disk. You can call this function manually to control when and
    how the pronunciation data is loaded (e.g., you're using this module in
    a web application and want to load the data asynchronously).
    :param filehandle: a filehandle with CMUdict-formatted data
    :returns: None

    # source: https://github.com/aparrish/pronouncingpy/blob/master/pronouncing/__init__.py
    # modified to make sure words are found in gpt2 tokenizer
    """
    global pronunciations, lookup, rhyme_lookup
    if pronunciations is None:
        if filehandle is None:
            filehandle = cmudict.dict_stream()
        pronunciations = pronouncing.parse_cmu(filehandle)
        filehandle.close()
        lookup = collections.defaultdict(list)
        for word, phones in pronunciations:
            lookup[word].append(phones)
        rhyme_lookup = collections.defaultdict(list)
        for word, phones in pronunciations:
            rp = pronouncing.rhyming_part(phones)
            if rp is not None:
                if word in GPT2_VOCAB:
                  rhyme_lookup[rp].append(word)

init_cmu()