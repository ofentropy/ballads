import sys
sys.path.append("/home/ubuntu/ballads") # change if necessary

import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tc_util.importutil import *
from tc_processing import *
from sklearn.model_selection import train_test_split


class MultiLabelCNN(InceptionV3):
    def __init__(self, num_labels):
        base_model = super(MultiLabelCNN, self).__init__(
            weights='imagenet', include_top='false', input_shape=(299,299,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(num_labels, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=outputs)
        
        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model


common_object_labels = get_labels_from_text("data/common_object_labels.txt")
common_sentiment_labels = get_labels_from_text("data/common_sentiment_labels.txt")
common_scene_labels = get_labels_from_text("data/common_scene_labels.txt")

objects_reverse_lookup, objects_lookup = create_labels_lookup(common_object_labels)
sentiments_reverse_lookup, sentiments_lookup = create_labels_lookup(common_sentiment_labels)
scenes_reverse_lookup, scenes_lookup = create_labels_lookup(common_scene_labels)

common_url_to_objects = get_img_labels_from_csv("data/common_url_to_objects.csv", "objects")
common_url_to_sentiments = get_img_labels_from_csv("data/common_url_to_sentiments.csv", "sentiments")
common_url_to_scenes = get_img_labels_from_csv("data/common_url_to_scenes.csv", "scenes")

X_objects, Y_objects = load_images_and_get_ground_truths(common_url_to_objects, objects_lookup, len(common_object_labels))
X_sentiments, Y_sentiments = load_images_and_get_ground_truths(common_url_to_sentiments, sentiments_lookup, len(common_sentiment_labels))
X_scenes, Y_scenes = load_images_and_get_ground_truths(common_url_to_scenes, common_url_to_scenes, len(common_scene_labels))

X_objects_train, X_objects_test, Y_objects_train, Y_objects_test = train_test_split(X_objects, Y_objects)
X_sentiments_train, X_sentiments_test, Y_sentiments_train, Y_sentiments_test = train_test_split(X_sentiments, Y_sentiments)
X_scenes_train, X_scenes_test, Y_scenes_train, Y_scenes_test = train_test_split(X_scenes, Y_scenes)