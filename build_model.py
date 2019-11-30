import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Dense, Input, Lambda, Subtract
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.metrics import TrueNegatives, FalsePositives, FalseNegatives, TruePositives



def build_model(input_size, eucl_dist_lvl, n_neurons, n_layers):
    in_1 = Input(shape=(input_size, ))
    in_2 = Input(shape=(input_size, ))

    if eucl_dist_lvl == 0:
        model = Concatenate()([in_1, in_2])
    else:
        model = Subtract()([in_1, in_2])
        if eucl_dist_lvl >= 2:
            model = Lambda(lambda tensor : tf.square(tensor), name="pointwise_square")(model)
        if eucl_dist_lvl >= 3:
            model = Lambda(lambda tensor : tf.reduce_sum(tensor, axis=1, keepdims=True), name="sum")(model)
        if eucl_dist_lvl >= 4:
            model = Lambda(lambda tensor : tf.sqrt(tensor), name="pointwise_sqrt")(model)
    
    for _ in range(n_layers):
        model = Dense(n_neurons, activation='sigmoid')(model)
    model_out = Dense(1, activation='sigmoid', name="classifier", use_bias=True)(model)
    
    model = Model([in_1, in_2], model_out)
    model.compile(loss=MeanSquaredError(), optimizer=SGD(learning_rate=0.05),
                metrics=[BinaryAccuracy(), Precision(), Recall(),
                TrueNegatives(), FalsePositives(), FalseNegatives(), TruePositives()])
    
    return model