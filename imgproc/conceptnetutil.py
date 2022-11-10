from collections import Counter
import requests
import json

def get_word_from_conceptnet_path(path):
  """
  Takes a ConceptNet path and strips the prefix
  to return the corresponding word

  @param path - str
  @return str
  """
  prefix = "/c/en/"
  prefix_len = len(prefix)
  if len(path) < prefix_len or path[:prefix_len] != prefix:
    print(f"{path} is not a valid ConceptNet path!")
    return path
  else:
    return path[prefix_len:]

def get_conceptnet_path_from_word(word, verify=False):
  """
  Takes a string and appends the ConceptNet prefix
  Assumes word is in English

  If verify is false, assumes that the word exists in ConceptNet
  If verify is true, will make an API call to ConceptNet
  for the word path

  @param word - str
  @param verify - bool
  @return str
  """
  if not verify:
    prefix = "/c/en/"
    return prefix + word
  else:
    related_api = "http://api.conceptnet.io/uri"
    params = {"language":"en", "text":word}
    response_raw = requests.get(related_api, params=params)
    if response_raw.status_code == 200:
      response = json.loads(response_raw.text)
      path = response.get("@id")
      return path
    else: return None

def get_relatedness_score(path1, path2):
  """
  Makes an API call to ConceptNet to find the relatedness score
  of the words in path1 and path2

  Assumes that path1 and path2 are properly formatted ConceptNet paths
  
  @param path1 - str
  @param path2 - str
  @return int
  """
  query = "http://api.conceptnet.io/relatedness?node1=" + path1 + "&node2=" + path2
  response_raw = requests.get(query)
  if response_raw.status_code == 200:
    response = json.loads(response_raw.text)
    score = response.get("value")
    return score
  else: return None

def get_n_related_terms(path, orig_path, n, min_weight, seen, related):
  """
  """
  prefix = "http://api.conceptnet.io/related"
  suffix = "?filter=/c/en"
  query = prefix + path + suffix

  response = requests.get(query)
  seen.append(path)

  if response.status_code != 200:
    print(f"Error: {response.status_code} encountered during attempt to retrieve {path}'s related terms.")
    return related
  else:
    terms = json.loads(response.text)["related"]
    for term in terms:
      if term.get("@id") not in seen:
        if path == orig_path:
          weight = term.get("weight")
        else:
          weight = get_relatedness_score(orig_path, term.get("@id"))
        if weight >= min_weight and len(related) < n:
          related[term.get("@id")] = weight
          seen.append(term.get("@id"))
  return related

def get_n_related_terms_raw_from_word(word, n, min_weight):
  """
  @param word - str - NOT ConceptNet Path
  @param n - int, number of related terms to retrieve
  @param min_weight - float
  """
  path = get_conceptnet_path_from_word(word)
  
  if path is None:
    print(f"Error: {word} was not found in ConceptNet.")
    return []
  seen = [path]
  related = {path:1.0}
  related = get_n_related_terms(path, path, n, min_weight, seen, related)

  if len(related) < n:
    c = Counter(related)
    most_common = c.most_common()
    for i in range(len(most_common)):
      if len(related) == n:
        break
      if most_common[i][1] < min_weight:
        print(f"Could only find {len(related)} terms with a relatedness score higher than {min_weight}.")
        return related
      else:
        related = get_n_related_terms(most_common[i][0], path, n,
                                        min_weight, seen, related)
  return related

def convert_related_raw_to_words(related_raw):
  """
  Strips ConceptNet prefix from each related term in the given array

  @param related_raw - list
  @return list
  """
  c = Counter(related_raw)
  array = []
  for term, _ in c.most_common():
    array.append(get_word_from_conceptnet_path(term))
  return array