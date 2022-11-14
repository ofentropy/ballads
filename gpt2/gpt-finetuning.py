import os
import warnings
import re
import re
import random
import datasets
import transformers
import tensorflow as tf
import numpy as np
import requests
import json
from datetime import datetime
from transformers import AutoTokenizer, TFGPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset, load_dataset
from tqdm.notebook import tqdm

np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_VERBOSITY"] = "info"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print(f"Tensorflow version: {tf.__version__}")
print(f"Transformers version: {transformers.__version__}")

MAX_TOKENS = 64
BOS_TOKEN = "<|beginoftext|>"
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
SEED = 10
EPOCHS = 10
INITIAL_LEARNING_RATE = 0.001
DECAY_STEPS = 300
DECAY_RATE = 0.7
MODEL_TYPE = "gpt2"
BATCH_SIZE = 32
SHUFFLE_TRAIN_BUFFER = 10000

random.seed(SEED)

def make_dataset(dataset_url="https://github.com/mckurz/ballads/raw/main/ballads_data3.json"):
    # downloads the dataset
    response = requests.get(dataset_url)
    corpus_data = json.loads(response.text)
    return corpus_data

def choose_random_words(poem, n_words = 3, min_length=4):
    # chooses n_words random words from a poem
    lines = poem.lower().split("\n")
    longer_words = []
    for line in lines:
      line = re.sub(r'[^A-Za-z ]+', '', line).split()
      longer_words.extend([word for word in line if len(word) >= min_length])
    if len(longer_words) > 0:
      how_many_random_words = n_words
      keywords = list(set(random.choices(longer_words, k = how_many_random_words)))
      return keywords
    else:
      return []

def tokenize(examples, tokenizer=tokenizer):
    # Transforms each quatrain into a prompt
    tokenized_examples = []
    for index, example in enumerate(examples):
      words = choose_random_words(example)
      prompt = BOS_TOKEN + "Keywords: " + " ".join(words) + "\nBallad: " + example + EOS_TOKEN
      tokenized_examples.append(prompt)

    output = tokenizer(
        tokenized_examples,
        add_special_tokens=True,
        max_length=MAX_TOKENS,
        truncation=True,
        pad_to_max_length=True,
    )

    output["labels"] = output["input_ids"].copy()

    return output

def generate_sample(model, tokenizer, prompt="<|beginoftext|> Keywords: victory love\nBallad: "):
  input_ids = tokenizer.encode(prompt, return_tensors='tf')
  sample_output = model.generate(input_ids, do_sample=True, max_length=64, top_k=50, top_p=0.95, num_return_sequences=1, no_repeat_ngram_size=2)
  generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True).strip()
  return generated_text

def make_quatrains(corpus_data):
    # change each ballad into quatrains with rhyming lines
    for ballad in corpus_data:
      rhymes_so_far = 0
      lines_read = 0
      rhyming_lines = []
      ballad_text = ballad["text"]
      ballad_text = re.sub("[0-9]","",ballad_text)
      ballad_text = re.sub("\\b[Ii]le", "I'll", ballad_text)
      ballad_lines = ballad_text.split("\n")
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
    quatrains = ["\n".join(quatrain) for quatrain in quatrains]
    return quatrains



tokenizer = GPT2Tokenizer.from_pretrained(
    MODEL_TYPE,
    bos_token=BOS_TOKEN,
    eos_token=EOS_TOKEN,
    pad_token=PAD_TOKEN,
    max_length=MAX_TOKENS,
    is_split_into_words=True,
)

ballads = make_dataset()
quatrains = make_quatrains(ballads)

valid_id = set(random.choices(range(0, len(quatrains)), k=int(0.05*len(quatrains))))
train_quatrains = [quatrain for id, quatrain in enumerate(quatrains) if id not in valid_id]
valid_quatrains = [quatrain for id, quatrain in enumerate(quatrains) if id in valid_id]

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    INITIAL_LEARNING_RATE,
    decay_steps=DECAY_STEPS,
    decay_rate=DECAY_RATE,
    staircase=True)

model = TFBalladModel.from_pretrained(
        MODEL_TYPE,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )
model.resize_token_embeddings(len(tokenizer))
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer)
model.layers[0].vocab_size = len(tokenizer) # something is wrong with TFGPT2 initialization so this is needed

valid_dataset = tokenize(valid_quatrains)
valid_dataset = tf.data.Dataset.from_tensor_slices({"input_ids": tf.convert_to_tensor(valid_dataset["input_ids"]),
                                                      "labels": tf.convert_to_tensor(valid_dataset["labels"]),
                                                      "attention_mask": tf.convert_to_tensor(valid_dataset["attention_mask"])})
valid_dataset = valid_dataset.batch(BATCH_SIZE,drop_remainder=False)

# training
for epoch in range(EPOCHS):
  train_dataset = tokenize(train_quatrains)
  train_dataset = tf.data.Dataset.from_tensor_slices({"input_ids": tf.convert_to_tensor(train_dataset["input_ids"]),
                                                      "labels": tf.convert_to_tensor(train_dataset["labels"]),
                                                      "attention_mask": tf.convert_to_tensor(train_dataset["attention_mask"])})
  train_dataset = train_dataset.shuffle(SHUFFLE_TRAIN_BUFFER).batch(BATCH_SIZE,drop_remainder=False)

  batch_loop = tqdm(train_dataset)
  batch_loop.set_description(f"Epoch {epoch}")
  for batch_index, batch in enumerate(batch_loop):
    with tf.GradientTape() as tape:
      output = model(input_ids=batch["input_ids"], labels=batch["labels"], attention_mask=batch["attention_mask"])
      #output = model(input_ids=batch[0], labels=batch[1], attention_mask=batch[2])
    gradients = tape.gradient(output.loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    batch_loop.set_postfix_str(f"Loss: {float(output.loss)}")

    if batch_index % 200 == 0 and batch_index != 0:
        print("**** Example ballad ****\n" + generate_sample(model, tokenizer) + "\n")

  model.save_weights(f"ballad-checkpoints/gpt2-ep{epoch}")
  
  val_batch_loop = tqdm(valid_dataset)
  valid_loss = 0
  valid_batch_no = 0
  for batch_index, batch in enumerate(val_batch_loop):
    valid_batch_no += 1
    output = model(input_ids=batch["input_ids"], labels=batch["labels"], attention_mask=batch["attention_mask"], training=False)
    valid_loss += float(output.loss)
  print(f"Validation loss after epoch {epoch}: {valid_loss/valid_batch_no}")
