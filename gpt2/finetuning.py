import sys
sys.path.append("/home/ubuntu/ballads")
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
corpus_data = corpus_data[:500]
print(f"Dataset loaded. There are {len(corpus_data)} ballads.") # should be 6597
print("Loading the tokenizer...")
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
print(f"Tokenizer loaded! Size: {len(tokenizer)}.")
encoded_punctuation = [tokenizer.encode(char)[0] for char in string.punctuation]
new_line_token_id = tokenizer.encode("\n")[0]

def get_last_words(quatrain, tokenizer):
    last_words = []
    quatrain = quatrain.lower().split("\n")
    for line in quatrain:
        last_word = None
        line = [tokenizer.decode(word_id).strip() for word_id in tokenizer.encode(line)]
        for word in line:
            if re.match(r"[a-zA-Z]+", word):
                last_word = word
        if last_word is not None:
            last_words.append(last_word)
    return last_words


def make_quatrains_and_prompts_for_single_ballad(ballad, patterns=["ABCB"]):
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
    #corrected_ballad = correct_and_normalize(ballad_text)
    corrected_ballad = ballad_text
    temp_adjs, temp_objects, temp_scenes = choose_random_words(corrected_ballad)
    prompts = []
    for i in range(len(quatrains)):
        prompt = generate_training_prompt_from_given(temp_adjs, temp_objects, temp_scenes)
        prompt_rhymes = ["rhymes:"] + get_last_words(quatrains[i], tokenizer)
        for word in prompt_rhymes:
            prompt += word + " "
        prompt += "\n"
        prompt += "ballad:\n"
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
    #prompt += "ballad:\n"

    return prompt

print("Preparing the prompts...")
train_quatrains = []
train_prompts = []

for ballad in tqdm(corpus_data):
    quatrains, prompts = make_quatrains_and_prompts_for_single_ballad(ballad)
    train_quatrains.extend(quatrains)
    train_prompts.extend(prompts)

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

print(quatrains[0])
last_words = get_last_words(quatrains[0], tokenizer)
print(last_words)

from dataclasses import dataclass
from typing import List, Optional, Tuple

import tensorflow as tf
from transformers import TFGPT2PreTrainedModel, TFGPT2MainLayer, BatchEncoding
from transformers.modeling_tf_outputs import TFCausalLMOutputWithCrossAttentions
from transformers.modeling_tf_utils import input_processing, TFModelInputType, TFCausalLanguageModelingLoss
from typing import Union
import inspect

class RhymeLoss(TFCausalLMOutputWithCrossAttentions):
  def __init__(self, rhyme_loss, *inputs, **kwargs):
    super(TFCausalLMOutputWithCrossAttentions, self).__init__(*inputs, **kwargs)
    self.rhyme_loss = rhyme_loss

class TFBalladModel(TFGPT2PreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super(TFBalladModel, self).__init__(config, *inputs, **kwargs)
        self.transformer = TFGPT2MainLayer(config, name="transformer")
        self.compute_rhyme_loss = kwargs.get("compute_rhyme_loss", False)
        self.tokenizer = kwargs.get("tokenizer", None)

    def get_output_embeddings(self):
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    def prepare_inputs_for_generation(self, inputs, past=None, use_cache=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            inputs = tf.expand_dims(inputs[:, -1], -1)
            if token_type_ids is not None:
                token_type_ids = tf.expand_dims(token_type_ids[:, -1], -1)

        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            if past:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)

        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past": past,
            "use_cache": use_cache,
            "token_type_ids": token_type_ids,
        }

    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        past: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_hidden_states: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
    ) -> Union[TFCausalLMOutputWithCrossAttentions, Tuple[tf.Tensor]]:

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        hidden_states = transformer_outputs[0]
        logits = self.transformer.wte(hidden_states, mode="linear")

        loss = None
        if labels is not None:
            # shift labels to the left and cut last logit token
            shifted_logits = logits[:, :-1]
            shifted_labels = labels[:, 1:]
            loss = self.hf_compute_loss(shifted_labels, shifted_logits)

        total_loss = RhymeLoss(
          loss=loss,
          logits=logits,
          rhyme_loss = None,
          past_key_values=transformer_outputs.past_key_values,
          hidden_states=transformer_outputs.hidden_states,
          attentions=transformer_outputs.attentions,
          cross_attentions=transformer_outputs.cross_attentions,
        )
        return total_loss

        #if not return_dict:
        #    output = (logits,) + transformer_outputs[1:]
        #    return ((loss,) + output) if loss is not None else output

    def serving_output(self, output):
        pkv = tf.convert_to_tensor(output.past_key_values) if self.config.use_cache else None
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None
        cross_attns = (
            tf.convert_to_tensor(output.cross_attentions)
            if self.config.output_attentions
            and self.config.add_cross_attention
            and output.cross_attentions is not None
            else None
        )

        return TFCausalLMOutputWithCrossAttentions(
            logits=output.logits, past_key_values=pkv, hidden_states=hs, attentions=attns, cross_attentions=cross_attns
        )

EPOCHS = 10
INITIAL_LEARNING_RATE = 0.0001
DECAY_STEPS = 300
DECAY_RATE = 0.7
BATCH_SIZE = 16
LM_LOSS_WEIGHT = 0.4
RHYME_LOSS_WEIGHT=0.6

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    INITIAL_LEARNING_RATE,
    decay_steps=DECAY_STEPS,
    decay_rate=DECAY_RATE,
    staircase=True)

model = TFBalladModel.from_pretrained(
        "gpt2",
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )
model.resize_token_embeddings(len(tokenizer))
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile()
model.layers[0].vocab_size = len(tokenizer) # something is wrong with TFGPT2 initialization so this is needed
model.assign_tokenizer(tokenizer)
model.summary()
model.compute_rhyme_loss = True

import time
"""
valid_dataset = tokenize(valid_quatrains)
valid_dataset = tf.data.Dataset.from_tensor_slices({"input_ids": tf.convert_to_tensor(valid_dataset["input_ids"]),
                                                      "attention_mask": tf.convert_to_tensor(valid_dataset["attention_mask"])})
valid_dataset = valid_dataset.batch(BATCH_SIZE,drop_remainder=False)
"""

def generate_sample(model, tokenizer, prompt="<|beginoftext|>sentiments: happiness pleasure\nobjects: tree crown\nscenes: coronation palace\nrhymes: victory crown beer gown\nballad:\n"):
  input_ids = tokenizer.encode(prompt, return_tensors='tf')
  sample_output = model.generate(input_ids, do_sample=True, max_length=MAX_TOKENS, top_k=50, top_p=0.95, num_return_sequences=1, no_repeat_ngram_size=2)
  generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
  return generated_text.strip()

for epoch in range(EPOCHS):
  # randomize the dataset
  train_dataset = tokenize(train_quatrains)
  train_dataset = tf.data.Dataset.from_tensor_slices({"input_ids": tf.convert_to_tensor(train_dataset["input_ids"]),
                                                      "attention_mask": tf.convert_to_tensor(train_dataset["attention_mask"])})
  #train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE,drop_remainder=False)
  train_dataset = train_dataset.batch(BATCH_SIZE,drop_remainder=False)
  #

  batch_loop = tqdm(train_dataset)
  batch_loop.set_description(f"Epoch {epoch}")
  for batch_index, batch in enumerate(batch_loop):
    with tf.GradientTape() as tape:
      output = model(input_ids=batch["input_ids"], labels=batch["input_ids"], attention_mask=batch["attention_mask"])

    gradients = tape.gradient(output.loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    batch_loop.set_postfix_str(f"GPT loss: {float(output.loss)}")

    if batch_index % 200 == 0 and batch_index != 0:
      generated_ballad = "\n------------------------------\n" + generate_sample(model, tokenizer) + "\n------------------------------\n"
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
