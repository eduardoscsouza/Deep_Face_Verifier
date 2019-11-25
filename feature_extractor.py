import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D

import numpy as np



def get_feature_extractor(vgg_weights_filepath="vgg_face_weights.h5"):
    model_in = Input(shape=(224, 224, 3))

    model = Conv2D(64, (3, 3), padding='same', activation='relu')(model_in)
    model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2))(model)

    model = Conv2D(128, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(128, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2))(model)

    model = Conv2D(256, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(256, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(256, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2))(model)

    model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2))(model)

    model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
    model_last_conv = MaxPooling2D((2, 2), strides=(2, 2))(model)

    model_dense_1 = Conv2D(4096, (7, 7), padding='valid', activation='relu')(model_last_conv)
    model_dense_2 = Conv2D(4096, (1, 1), padding='valid', activation='relu')(model_dense_1)
    model_out = Conv2D(2622, (1, 1), padding='valid', activation='softmax')(model_dense_2)

    vgg_model = Model(model_in, model_out)
    vgg_model.load_weights(vgg_weights_filepath)

    extract_last_conv = Flatten()(model_last_conv)
    extract_dense_1 = Flatten()(model_dense_1)
    extract_dense_2 = Flatten()(model_dense_2)
    extract_out = Flatten()(model_out)

    extractor_model = Model(model_in, [extract_last_conv, extract_dense_1, extract_dense_2, extract_out])

    return extractor_model

get_feature_extractor().summary()