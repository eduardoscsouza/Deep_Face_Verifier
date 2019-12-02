from tensorflow.keras.utils import Sequence
from itertools import chain
import numpy as np
import os



#Generates face pairs for verification
class FacePairGenerator(Sequence):
    @staticmethod 
    def _load_fold(fold_file):
        fold_dir = os.sep.join(fold_file.split(os.sep)[:-1])
        with open(fold_file, 'r') as op_fold_file:
            loaded_fold = [np.load(os.path.join(fold_dir, indv.strip()+".npy")) for indv in op_fold_file]
        
        return loaded_fold

    def __init__(self, folds_files, batch_size=32):
        self.batch_size = batch_size
        self.indvs = list(chain.from_iterable([FacePairGenerator._load_fold(fold_file) for fold_file in folds_files]))

        self._aux_len = len(self.indvs)
        self._aux_range = np.arange(self._aux_len)

    def __len__(self):
        return self._aux_len
    
    def _get_pair(self, y):
        aux_indv = np.random.randint(self._aux_len, high=None, size=None)
        indv_1, indv_2 = (aux_indv, aux_indv) if y else np.random.choice(self._aux_range, size=2, replace=False, p=None)
        img_1, img_2 = np.random.choice(len(self.indvs[aux_indv]), size=2, replace=False, p=None) if y else (np.random.randint(len(self.indvs[indv_1]), high=None, size=None), np.random.randint(len(self.indvs[indv_2]), high=None, size=None))

        return self.indvs[indv_1][img_1], self.indvs[indv_2][img_2]

    def __getitem__(self, index):
        y = np.random.randint(2, high=None, size=self.batch_size)
        x = np.asarray([self._get_pair(cur_y) for cur_y in y])
        
        return [x[:, 0, :], x[:, 1, :]], y



if __name__ == '__main__':
    codium 