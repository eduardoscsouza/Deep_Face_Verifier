import numpy as np
import pandas as pd
from generator import FacePairGenerator
from glob import glob
from math import ceil
import os



def get_from_dir(dir_path, name_prefix="", n_pairs=25000):
    generator = FacePairGenerator(glob(os.path.join(dir_path, "fold_*.txt")), batch_size=n_pairs)
    (a, b), y = generator.__getitem__(0)

    con = np.concatenate([a, b], axis=1)
    sqr = np.square(a-b)
    root = np.expand_dims(np.sqrt(np.sum(sqr, axis=1)), axis=1)
    root = np.concatenate([root, root], axis=1)
    
    pd.DataFrame(y, columns=["Class"]).to_csv(name_prefix+"_meta.csv", index=False, header=False, sep='\t')
    pd.DataFrame(con).to_csv(name_prefix+"_concat.csv", index=False, header=False, sep='\t')
    pd.DataFrame(sqr).to_csv(name_prefix+"_square.csv", index=False, header=False, sep='\t')
    pd.DataFrame(root).to_csv(name_prefix+"_root.csv", index=False, header=False, sep='\t')

    mul = a*b
    cos = np.expand_dims(1 - (np.sum(mul, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))), axis=1)
    cos = np.concatenate([cos, cos], axis=1)
    pd.DataFrame(mul).to_csv(name_prefix+"_multiply.csv", index=False, header=False, sep='\t')
    pd.DataFrame(cos).to_csv(name_prefix+"_cosine.csv", index=False, header=False, sep='\t')



if __name__ == '__main__':
    #get_from_dir("extracted/layer_0", name_prefix="/media/wheatley/38882E5E882E1AC0/Deep_Face_Verifier/layer_0")
    #get_from_dir("extracted/layer_1", name_prefix="/media/wheatley/38882E5E882E1AC0/Deep_Face_Verifier/layer_1")
    #get_from_dir("extracted/layer_2", name_prefix="/media/wheatley/38882E5E882E1AC0/Deep_Face_Verifier/layer_2")
    for i in range(5):
        get_from_dir("autoencoder_extracted/fold_{}".format(i), name_prefix="autoencoder_fold_{}".format(i))