import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, DepthwiseConv2D
from tensorflow.keras.layers import Flatten, Input, Lambda, Layer, MaxPooling2D, Multiply, Reshape, Subtract, UpSampling2D
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.metrics import TrueNegatives, FalsePositives, FalseNegatives, TruePositives



def build_eucl_model(input_size, eucl_dist_lvl, n_neurons, n_layers, batch_norm=True, 
                    loss=MeanSquaredError(), optimizer=SGD(learning_rate=0.05, momentum=0.025)):
    in_1 = Input(shape=(input_size, ), name="input_1")
    in_2 = Input(shape=(input_size, ), name="input_2")

    if eucl_dist_lvl == 0:
        model = Concatenate(name="concatenate")([in_1, in_2])
    else:
        model = Subtract(name="subtract")([in_1, in_2])
        if eucl_dist_lvl >= 2:
            model = Lambda(lambda tensor : tf.square(tensor), name="square")(model)
        if eucl_dist_lvl >= 3:
            model = Lambda(lambda tensor : tf.reduce_sum(tensor, axis=1, keepdims=True), name="sum")(model)
        if eucl_dist_lvl >= 4:
            model = Lambda(lambda tensor : tf.sqrt(tensor), name="root")(model)
    
    if batch_norm:
        model = BatchNormalization(name="input_normalization")(model)

    for i in range(n_layers):
        model = Dense(n_neurons, activation='sigmoid', name="dense_{}".format(i))(model)
    model_out = Dense(1, activation='sigmoid', name="classify")(model)
    
    model = Model([in_1, in_2], model_out)
    model.compile(loss=loss, optimizer=optimizer,
                metrics=[BinaryAccuracy(), Precision(), Recall(),
                TrueNegatives(), FalsePositives(), FalseNegatives(), TruePositives()])
    
    return model



class ValueMinusInput(Layer):
    def __init__(self, value, **kwargs):
        super(ValueMinusInput, self).__init__(**kwargs)
        self.value = value

    def build(self, input_shape):
        self.const_vals = self.add_weight(shape=input_shape[1:],
                                        initializer=tf.constant_initializer(self.value),
                                        trainable=False)

    def call(self, input):
        return tf.subtract(self.const_vals, input)
    
    def get_config(self):
        cfg = super(ValueMinusInput, self).get_config()
        cfg.update({'value': self.value})
        return cfg

def build_cos_model(input_size, cos_dist_lvl, n_neurons, n_layers, batch_norm=True,
                    loss=MeanSquaredError(), optimizer=SGD(learning_rate=0.05, momentum=0.025)):
    in_1 = Input(shape=(input_size, ), name="input_1")
    in_2 = Input(shape=(input_size, ), name="input_2")

    if cos_dist_lvl == 0:
        model = Concatenate(name="concatenate")([in_1, in_2])
    else:
        model = Multiply(name="pointwise_multiply")([in_1, in_2])
        if cos_dist_lvl >= 2:
            norm_1 = Lambda(lambda tensor : tf.norm(tensor, axis=1, keepdims=True), name="norm_input_1")(in_1)
            norm_2 = Lambda(lambda tensor : tf.norm(tensor, axis=1, keepdims=True), name="norm_input_2")(in_2)
            norm_mul = Multiply(name="multiply_norms")([norm_1, norm_2])
            model = Lambda(lambda tensors : tf.divide(tensors[0], tensors[1]), name="divide")([model, norm_mul])
        if cos_dist_lvl >= 3:
            model = Lambda(lambda tensor : tf.reduce_sum(tensor, axis=1, keepdims=True), name="sum")(model)
        if cos_dist_lvl >= 4:
            model = ValueMinusInput(1, name="one_minus_input")(model)

    if batch_norm:
        model = BatchNormalization(name="input_normalization")(model)

    for i in range(n_layers):
        model = Dense(n_neurons, activation='sigmoid', name="dense_{}".format(i))(model)
    model_out = Dense(1, activation='sigmoid', name="classify")(model)
    
    model = Model([in_1, in_2], model_out)
    model.compile(loss=loss, optimizer=optimizer,
                metrics=[BinaryAccuracy(), Precision(), Recall(),
                TrueNegatives(), FalsePositives(), FalseNegatives(), TruePositives()])
    
    return model



class LessThan(Layer):
    def __init__(self, value, **kwargs):
        super(LessThan, self).__init__(**kwargs)
        self.value = value

    def build(self, input_shape):
        self.const_vals = self.add_weight(shape=input_shape[1:],
                                        initializer=tf.constant_initializer(self.value),
                                        trainable=False)

    def call(self, input):
        return tf.less(input, self.const_vals)

    def get_config(self):
        cfg = super(LessThan, self).get_config()
        cfg.update({'value': self.value})
        return cfg

def build_base_model(input_size):
    in_1 = Input(shape=(input_size, ), name="input_1")
    in_2 = Input(shape=(input_size, ), name="input_2")

    norm_1 = Lambda(lambda tensor : tf.norm(tensor, axis=1, keepdims=True), name="norm_input_1")(in_1)
    norm_2 = Lambda(lambda tensor : tf.norm(tensor, axis=1, keepdims=True), name="norm_input_2")(in_2)
    norm_mul = Multiply(name="multiply_norms")([norm_1, norm_2])

    model = Multiply(name="pointwise_multiply")([in_1, in_2])
    model = Lambda(lambda tensor : tf.reduce_sum(tensor, axis=1, keepdims=True), name="sum")(model)
    
    model = Lambda(lambda tensors : tf.divide(tensors[0], tensors[1]), name="divide")([model, norm_mul])
    model = ValueMinusInput(1, name="one_minus_input")(model)

    model = LessThan(0.4)(model)
    model_out = Lambda(lambda tensor : tf.cast(tensor, tf.float32), name="cast")(model)
    
    model = Model([in_1, in_2], model_out)
    model.compile(loss=MeanSquaredError(), optimizer=SGD(),
                metrics=[BinaryAccuracy(), Precision(), Recall(),
                TrueNegatives(), FalsePositives(), FalseNegatives(), TruePositives()])
    return model



def build_autoencoder(input_size=32, base_n_filters=8, n_layers=1, encoding_dims=128,
                    loss=MeanSquaredError(), optimizer=Adam()):
    model_in = Input(shape=(input_size, input_size, 3), name="input")

    model = model_in
    for i in range(n_layers):
        model = DepthwiseConv2D((5, 5), padding='same', activation='relu', name="encod_block_{}_depth_conv".format(i))(model)
        model = Conv2D(base_n_filters, (1, 1), padding='same', activation='relu', name="encod_block_{}_conv".format(i))(model)
        model = MaxPooling2D((2, 2), strides=(2, 2), name="encod_block_{}_max_pool".format(i))(model)
    
    model = Flatten(name="encod_reshap")(model)
    model = Dense(encoding_dims, activation='relu', name="encod_dense")(model)
    if n_layers == 0:
        model = Dense(input_size * input_size * 3, activation='relu', name="decod_dense")(model)
        model_out = Reshape((input_size, input_size, 3), name="decod_reshap")(model)
    else:
        model = Dense((input_size//(2**n_layers) * input_size//(2**n_layers) * base_n_filters), activation='relu', name="decod_dense")(model)
        model = Reshape((input_size//(2**n_layers), input_size//(2**n_layers), base_n_filters), name="decod_reshap")(model)
    
    for i in range(n_layers):
        model = UpSampling2D((2, 2), name="decod_block_{}_up_sampl".format(i))(model)
        model = DepthwiseConv2D((5, 5), padding='same', activation='relu', name="decod_block_{}_depth_conv".format(i))(model)
        model = Conv2D(base_n_filters, (1, 1), padding='same', activation='relu', name="decod_block_{}_conv".format(i))(model)
    
    if n_layers != 0:
        model_out = Conv2D(3, (1, 1), padding='same', activation='relu', name="decod_final_conv")(model)

    model = Model(model_in, model_out)
    model.compile(loss=loss, optimizer=optimizer)

    return model



if __name__ == '__main__':
    import numpy as np

    v1 = np.random.rand(1000, 10)
    v2 = np.random.rand(1000, 10)

    m1 = build_base_model(10)
    m2 = Model(m1.input, m1.get_layer(name="one_minus_input").output)
    m3 = build_cos_model(10, 4, 20, 2)
    m3 = Model(m3.input, m3.get_layer(name="one_minus_input").output)

    a = np.sum(np.multiply(v1, v2), axis=1)
    b = np.linalg.norm(v1, axis=1)
    c = np.linalg.norm(v2, axis=1)

    r1 = 1 - (a / (b * c))
    r2 = m2.predict([v1, v2])[:, 0]
    r3 = m3.predict([v1, v2])[:, 0]
    assert np.allclose(r1, r2)
    assert np.allclose(r1, r3)

    r1 = (r1 < 0.4).astype(np.float32)
    r2 = m1.predict([v1, v2])[:, 0]
    assert np.allclose(r1, r2)
    
    m1 = build_eucl_model(10, 4, 20, 2)
    m1 = Model(m1.input, m1.get_layer(name="root").output)

    r1 = np.linalg.norm(v1-v2, axis=1)
    r2 = m1.predict([v1, v2])[:, 0]
    assert np.allclose(r1, r2)