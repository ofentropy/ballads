import csv

def get_labels_from_text(labels_path):
    with open(labels_path, 'r') as f:
        data = f.read()
        labels = data.split("\n")

    return labels


def get_img_labels_from_csv(csv_path, key):
    urls_to_label = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            temp_dict = dict(row)
            url = temp_dict["image_url"]
            labels = temp_dict[key]
            labels = labels.split(";")
            urls_to_label[url] = labels
    return urls_to_label