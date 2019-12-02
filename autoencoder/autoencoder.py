from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import vgg16

def get_generator(dir, batch_size=8):
    print("Creating generator")
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
                    class_mode='input',
                    batch_size=batch_size,
                    shuffle=False,
                    seed=500,
                    save_to_dir=None,
                    save_prefix="",
                    save_format='jpg',
                    follow_links=False,
                    subset=None,
                    interpolation='bicubic')

    datagen = ImageDataGenerator(**gen_args).flow_from_directory(dir, **flow_args)
    return datagen



def create_model():
    print("Creating model")
    inChannel = 3
    x, y = 224, 224
    input_img = Input(shape = (x, y, inChannel))

    # encoder
    conv = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) # 224 x 224 x 32
    pool = MaxPooling2D(pool_size=(2, 2))(conv) # 112 x 112 x 32
    conv = Conv2D(64, (3, 3), activation='relu', padding='same')(pool) # 112 x 112 x 64
    pool = MaxPooling2D(pool_size=(2, 2))(conv) # 56 x 56 x 64
    conv = Conv2D(128, (3, 3), activation='relu', padding='same')(pool) # 56 x 56 x 128

    # decoder
    conv = Conv2D(128, (3, 3), activation='relu', padding='same')(conv) #56 x 56 x 128
    up = UpSampling2D((2, 2))(conv) # 112 x 112 x 128
    conv = Conv2D(64, (3, 3), activation='relu', padding='same')(up) # 112 x 112 x 64
    up = UpSampling2D((2, 2))(conv) # 256 x 256 x 64
    conv = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up) # 256 x 256 x 3

    model = Model(input_img, conv)
    model.compile(loss='mean_squared_error', optimizer = RMSprop())

    return model

def train_model(model, generator):
    print("Training model")
    epochs = 50
    model.fit_generator(generator, steps_per_epoch=2000, epochs = epochs)

if __name__ == '__main__':
    dir = '../autoencoder_dataset'
    generator = get_generator(dir = dir)
    model = create_model()
    
    train_model(model, generator)


