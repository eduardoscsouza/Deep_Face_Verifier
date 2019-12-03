from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_autoencoder():
    print("Creating autoencoder model")
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
    conv = Conv2D(3, (3, 3), activation='tanh', padding='same')(up) # 256 x 256 x 3

    model = Model(input_img, conv)
    model.compile(loss='mean_squared_error', optimizer = Adam())

    return model