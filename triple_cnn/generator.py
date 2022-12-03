from tc_processing import decode_predictions, img_h, img_w
from finetuning import load_model

import tensorflow as tf
import cv2
import random
import numpy as np
from urllib.request import urlopen
import matplotlib.pyplot as plt

def load_precision_cnn(model_path, num_labels):
    metrics = [tf.keras.metrics.Precision(top_k = num_labels)]
    return load_model(model_path, num_labels, metrics)

class TripleCNN():
    def __init__(self, object_lookup, scene_lookup, sentiment_lookup,
                       object_model, scene_model, sentiment_model):
        self.verbose = False
        self.img_size = (img_h, img_w, 3)

        self.object_lookup = object_lookup
        self.scene_lookup = scene_lookup
        self.sentiment_lookup = sentiment_lookup

        self.object_model = object_model
        self.scene_model = scene_model
        self.sentiment_model = sentiment_model
    
    def generate_prompt(self, img_path, kind="url", n=3, top_k=10, verbose=False):
        """
        if img_path is a url, specify kind = "url", if img_path is a file path, specify kind = "file"
        n = number of labels per category to use in prompt
        """
        self.verbose = verbose
        chosen_labels = self.generate_labels(img_path, kind=kind, n=n, top_k=top_k, verbose=verbose)
        prompt = self._concatenate_prompt(chosen_labels)
        return prompt

    def generate_labels(self, img_path, kind="url", n=3, top_k=10, verbose=False):
        self.verbose = verbose
        crop = self._crop_image(img_path, kind=kind)
        preds_dict = self._predict_all(crop)
        chosen_labels = self._label_all(preds_dict, n=n, top_k=top_k)
        return chosen_labels

    def _crop_image(self, img_path, kind="url"):
        # reference: https://stackoverflow.com/a/60431763
        if kind == "url":
            with urlopen(img_path) as request:
                img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(img_path)

        # crop img; reference: https://stackoverflow.com/a/61942452
        crop_h, crop_w, _= self.img_size
        center = img.shape
        width = center[1]
        height = center[0]
        
        if height < width:
            scale_percent = (crop_h+2)/height
        else:
            scale_percent = (crop_w+2)/width

        resized = cv2.resize(img, (int(width*scale_percent), int(height*scale_percent)), 
                        interpolation = cv2.INTER_AREA)    
        
        new_center = resized.shape
        x = new_center[1]/2 - crop_w/2
        y = new_center[0]/2 - crop_h/2

        crop = resized[int(y):int(y+crop_h), int(x):int(x+crop_w)]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if self.verbose:
          plt.imshow(crop, interpolation='nearest')
          plt.show()
        return crop

    def _predict_all(self, img):
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        object_preds = self.object_model.predict(x)
        scene_preds = self.scene_model.predict(x)
        sentiment_preds = self.sentiment_model.predict(x)
        return {
            "object_preds": object_preds,
            "scene_preds": scene_preds,
            "sentiment_preds": sentiment_preds,
        }
    
    def _decode(self, preds, lookup, top_k):
        return decode_predictions(preds, lookup, k=top_k)

    def _label_all(self, preds_dict, n=3, top_k=10):
        objects = self._decode(preds_dict["object_preds"], self.object_lookup, top_k)
        scenes = self._decode(preds_dict["scene_preds"], self.scene_lookup, top_k)
        sentiments = self._decode(preds_dict["sentiment_preds"], self.sentiment_lookup, top_k)

        chosen_objects = random.sample(objects, n)
        chosen_scenes = random.sample(scenes, n)
        chosen_sentiments = random.sample(sentiments, n)
        
        if self.verbose:
            print(f"All object labels: {objects} / Chosen: {chosen_objects}")
            print(f"All scene labels: {scenes} / Chosen: {chosen_scenes}")
            print(f"All sentiment labels: {sentiments} / Chosen: {chosen_sentiments}")
            
        return {
            "objects": chosen_objects,
            "scenes": chosen_scenes,
            "sentiments": chosen_sentiments,
        }


    def _concatenate_prompt(self, chosen_labels):
        objects = ["objects:"] + chosen_labels["objects"]
        scenes = ["scenes:"] + chosen_labels["scenes"]
        sentiments = ["sentiments:"] + chosen_labels["sentiments"]

        prompt = ""
        selections = [objects, scenes, sentiments]
        for group in selections:
            for word in group:
                prompt += word + " "
            prompt += "\n"

        if self.verbose:
          print("===PROMPT===")
          print(prompt)

        return prompt




