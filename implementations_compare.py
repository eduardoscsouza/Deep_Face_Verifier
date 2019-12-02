from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

from tensorflow.keras.models import model_from_json
model.load_weights("vgg_face_weights.h5")

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

epsilon = 0.40

def verifyFace(img1, img2, plot=False):
    img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0,:]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0,:]
    
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    
    #print("Cosine similarity: ",cosine_similarity)
    #print("Euclidean distance: ",euclidean_distance)
    
    if plot:
        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(image.load_img(img1))
        plt.xticks([]); plt.yticks([])
        f.add_subplot(1,2, 2)
        plt.imshow(image.load_img(img2))
        plt.xticks([]); plt.yticks([])
        plt.show(block=True)
        print("-----------------------------------------")

    if(cosine_similarity < epsilon):
        return np.expand_dims(np.stack([cosine_similarity, np.asarray(1.0)], axis=0), axis=1)
    else:
        return np.expand_dims(np.stack([cosine_similarity, np.asarray(0.0)], axis=0), axis=1)



import feature_extractor
import build_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Conv2D, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.applications import vgg16

import os
from glob import glob
import shutil
import numpy as np
import pandas as pd
my_extractor = feature_extractor.get_feature_extractor()
my_base_model = build_model.build_base_model(2622)
my_cos_dis = Model(my_base_model.input, my_base_model.get_layer(name="one_minus_input").output)
gen_args = dict(featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    zca_epsilon=1e-06,
                    rotation_range=0.0,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    brightness_range=None,
                    shear_range=0.0,
                    zoom_range=0.0,
                    channel_shift_range=0.0,
                    fill_mode='nearest',
                    cval=0.0,
                    horizontal_flip=False,
                    vertical_flip=False,
                    rescale=None,
                    preprocessing_function=vgg16.preprocess_input,
                    data_format='channels_last',
                    validation_split=0.0,
                    dtype=None)

flow_args = dict(target_size=(224, 224),
                color_mode='rgb',
                classes=None,
                class_mode='binary',
                batch_size=2,
                shuffle=False,
                seed=None,
                save_to_dir=None,
                save_prefix="",
                save_format='jpg',
                follow_links=False,
                subset=None,
                interpolation='nearest')

def my_verifyFace(img1, img2):
    aux_df = pd.DataFrame([(str(img1), "0"), (str(img2), "1")], columns=["filename", "class"])
    datagen = ImageDataGenerator(**gen_args).flow_from_dataframe(aux_df, **flow_args)
    v1, v2 = my_extractor.predict(datagen.next())[-1]
    v1, v2 = np.expand_dims(v1, axis=0), np.expand_dims(v2, axis=0)
    return np.stack([my_cos_dis.predict([v1, v2])[0], my_base_model.predict([v1, v2])[0]], axis=0)

def my_verifyFace_no_datagen(img1, img2):
    img1 = vgg16.preprocess_input(img_to_array(load_img(img1, target_size=(224, 224))))
    img2 = vgg16.preprocess_input(img_to_array(load_img(img2, target_size=(224, 224))))
    v1, v2 = my_extractor.predict(np.stack([img1, img2], axis=0))[-1]
    v1, v2 = np.expand_dims(v1, axis=0), np.expand_dims(v2, axis=0)
    return np.stack([my_cos_dis.predict([v1, v2])[0], my_base_model.predict([v1, v2])[0]], axis=0)


import os
n_tests = 2*10
dataset_dir = "dataset"
dirs = [os.path.join(dataset_dir, cur_dir) for cur_dir in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, cur_dir))]
for _ in range(n_tests//2):
    indv = np.random.randint(len(dirs))
    imgs = [os.path.join(dirs[indv], img) for img in os.listdir(dirs[indv])]
    img_1, img_2 = np.random.choice(imgs, 2, replace=False)

    r1 = verifyFace(img_1, img_2, True)
    r2 = my_verifyFace(img_1, img_2)
    r3 = my_verifyFace_no_datagen(img_1, img_2)
    print(r1, r2, r3, sep='\n')
    assert np.allclose(r1, r2)
    assert np.allclose(r1, r3)

for _ in range(n_tests//2):
    indv_1, indv_2 = np.random.choice(dirs, 2, replace=False)
    img_1 = np.random.choice([os.path.join(indv_1, img) for img in os.listdir(indv_1)], 1)[0]
    img_2 = np.random.choice([os.path.join(indv_2, img) for img in os.listdir(indv_2)], 1)[0]

    r1 = verifyFace(img_1, img_2, True)
    r2 = my_verifyFace(img_1, img_2)
    r3 = my_verifyFace_no_datagen(img_1, img_2)
    print(r1, r2, r3, sep='\n')
    assert np.allclose(r1, r2)
    assert np.allclose(r1, r3)