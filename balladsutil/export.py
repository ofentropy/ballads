import json
import os

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