import pandas as pd

def get_stats(datagen, model, n_samples=1000):
    outs_df = pd.DataFrame()
    for _ in range(n_samples):
        cur_df = pd.DataFrame(model.predict(datagen.__getitem__(0)[0]))
        outs_df = pd.concat([outs_df, cur_df])

    return outs_df.describe()



if __name__ == '__main__':
    from tensorflow.keras.layers import GaussianNoise, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import Sequence
    import numpy as np
    
    class ZeroGenerator(Sequence):
        def __init__(self, in_size=1000, batch_size=32):
            self.batch_size = batch_size
            self.in_size = in_size

        def __len__(self):
            return self.batch_size * self.in_size

        def __getitem__(self, index):
            return np.zeros((self.batch_size, self.in_size)), 0

    i = Input(shape=(1000, ))
    m = GaussianNoise(1)(i, training=True)
    m = Model(i, m)
    print(get_stats(ZeroGenerator(), m))
