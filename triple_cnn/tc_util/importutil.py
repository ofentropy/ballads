import csv
import matplotlib.pyplot as plt

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

def create_graph_from_urls_to_label(urls_to_label, key):
    label_counts = {}
    for _, labels in urls_to_label.items():
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
    names = label_counts.keys()
    values = label_counts.values()
    
    total = sum(values)
    percentages = [v/total for v in values]
    print(percentages)

    plt.bar(range(len(label_counts)), values, tick_label=names)
    plt.savefig(key, format="png")
    #plt.show()