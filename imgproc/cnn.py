import tensorflow as tf
import requests
import json
import numpy as np
import random
import os

# global variables
img_key = "image_url"
id_key = "id"
dataset_url = "https://raw.githubusercontent.com/researchmm/img2poem/master/data/multim_poem.json"
response = requests.get(dataset_url)
multim_poem_dataset = json.loads(response.text)

# InceptionV3 CNN
base_model = tf.keras.applications.InceptionV3()

def get_keywords_for_images(images, n=1):
  """
  Gets the top n keywords for each img in images
  Calls get_keywords_from_img_url

  @param images - list
  @param n - int
  @return dict {int:array}
  """
  img_keyword_dict = {}
  for img in images:
    img_url, img_id = get_img_info(img) # get url and id of sampled img
    keywords = get_keywords_from_img_url(img_url, img_id, base_model, n) # get top n keyword(s) from img using inceptionV3
    if keywords == None:
      print(f"Image at {img_url} is invalid.")
    else:
      print(f"Classified image at {img_url} as {keywords}.")
    img_keyword_dict[img_id] = keywords
  return img_keyword_dict

def get_keywords_from_img_url(img_url, img_id, model, n=1):
  """
  Gets the top n keywords of the given image
  using the given model and returns a list

  If the given image is invalid, returns None

  Assumes that model is a version of InceptionV3

  @param img_url
  @param img_id
  @param n
  @param model
  @return list
  """
  img_data = requests.get(img_url).content

  # verify that image link is not broken - reference: https://stackoverflow.com/a/51757322
  if not img_data[:3] == b'\xff\xd8\xff': return None
  img_path = f"{img_id}.jpg"
  open(img_path, 'wb').write(img_data)
  processed_img = tf.keras.preprocessing.image.load_img(img_path, 
                                                        target_size=(299,299))
  os.remove(img_path) # deletes downloaded image from os

  x = tf.keras.preprocessing.image.img_to_array(processed_img)
  x = np.expand_dims(x, axis=0)
  x = tf.keras.applications.inception_v3.preprocess_input(x)
  preds = model.predict(x)

  keywords_raw = tf.keras.applications.inception_v3.decode_predictions(preds, top=n)[0]
  keywords = [kw[1] for kw in keywords_raw]

  return keywords

# HELPER METHODS
def get_k_images(k, seed, dataset):
  """
  Randomly samples k images from the given dataset

  @param k - int
  @param dataset - json dict
  @return list
  """
  random.seed(seed)
  return random.sample(dataset, k)

def get_img_info(img):
  """
  Gets the url and id of the given image sample

  @return tuple (str, int)
  """
  return img.get(img_key), img.get(id_key)

