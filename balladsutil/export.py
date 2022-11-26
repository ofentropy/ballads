import json
import os
import csv

def export_ballads(save_path, img_to_poems, img_to_keywords, img_to_related, img_to_urls):
  """
  Assumes img_to_poems, img_to_keywords, img_to_related, and img_to_urls are the same length
  """
  if not os.path.exists(save_path.split("/")[0]):
    os.makedirs(save_path.split("/")[0])
  json_format = []
  for id, poem in img_to_poems.items():
    url = img_to_urls.get(id)
    related_words = img_to_related.get(id)
    kw = img_to_keywords.get(id)
    json_format.append({"id": id, "url": url, "keyword": kw[0], 
                   "related": related_words, "poem": poem})
  new_file = open(save_path, 'w')
  print(json.dumps(json_format, sort_keys=True, indent=4), file=new_file)
  new_file.close()


def export_url_to_labels(url_to_labels, label_type, save_path):
  header_list = ["image_url", label_type]
  with open(save_path, 'w') as f:
    w = csv.writer(f)
    w.writerow(header_list)
    for url, labels in url_to_labels.items():
      l = ";".join(labels)
      w.writerow([url, l])


def get_labels(dataset_dict):
  labels = []
  for _, raw_labels in dataset_dict.items():
    for label in raw_labels:
      labels.append(label)
  labels = [*set(labels)]
  return labels


def export_labels(labels, save_path):
  """
  Saves labels to given save_path
  """
  with open(save_path, 'w') as f:
    for label in labels:
      print(f"{label}", file=f)


def export_common_labels(labels, save_path, min_threshold = 10):
    """
    returns labels + length of labels
    """
    real_labels = {}

    for label, urls in labels.items():
      if len(urls) >= min_threshold:
        real_labels[label] = urls

    export_labels(real_labels.keys(), save_path)

    return real_labels, len(real_labels)