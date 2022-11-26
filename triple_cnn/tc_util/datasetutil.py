import requests
from tqdm import tqdm
import json

id_key = "id"
img_url_key = "image_url"
poem_key = "poem"

def get_valid_links_only(dataset, export=False):
    valid_dataset = []
    for elem in tqdm(dataset):
        img_url = elem[img_url_key]
        img_data = requests.get(img_url).content
        if img_data[:3] == b'\xff\xd8\xff':
            valid_dataset.append(elem)
    
    if export:
        new_file = open("valid_multim_poem.json", 'w')
        print(json.dumps(valid_dataset, sort_keys=True, indent=4), file=new_file)
        new_file.close()

    return valid_dataset