import cv2
from urllib.request import urlopen
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

img_h, img_w = (299, 299)

def view_image(dataset, idx):
    """
    Print and view image at given index of the given dataset
    """
    data = dataset[idx]
    plt.imshow(data, interpolation='nearest')
    plt.show()


def load_images_and_get_ground_truths(dataset, lookup, class_num):
    """
    Load images from given URLs + resize & crop
    Returns normalized images and the associated url order
    """
    images = []
    ground_truths = []
    for url, labels in tqdm(dataset.items()):
        # reference: https://stackoverflow.com/a/60431763
        with urlopen(url) as request:
            img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # crop img; reference: https://stackoverflow.com/a/61942452
        center = img.shape
        width = center[1]
        height = center[0]
        
        if height < width:
            scale_percent = (img_h+2)/height
        else:
            scale_percent = (img_w+2)/width

        resized = cv2.resize(img, (int(width*scale_percent), int(height*scale_percent)), 
                        interpolation = cv2.INTER_AREA)    
        
        new_center = resized.shape
        x = new_center[1]/2 - img_w/2
        y = new_center[0]/2 - img_h/2

        crop = resized[int(y):int(y+img_h), int(x):int(x+img_w)]
        
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        images.append(crop)
        
        ground_truth = create_ground_truth_vector(labels, lookup, class_num)
        ground_truths.append(ground_truth)
    # normalize
    images_normalized = np.array(images, dtype=np.float32) / 255
    
    return images_normalized, np.array(ground_truths, dtype=np.int32)


def create_labels_lookup(labels):
    """
    Enumerates labels alphabetically
    """
    labels = sorted(labels)
    enumerated_dict = dict(enumerate(labels))
    return enumerated_dict, dict((value, key) for key, value in enumerated_dict.items())


def create_ground_truth_vector(labels, lookup, class_num):
    ground_truth = np.zeros(class_num)
    for label in labels:
        idx = lookup[label]
        ground_truth[idx] = 1
    return ground_truth


def labels_from_one_hot(vec, reverse_lookup):
    labels = []
    n = vec.shape[0]
    for i in range(n):
        if vec[i] == 1:
            labels.append(reverse_lookup[i])
    return labels