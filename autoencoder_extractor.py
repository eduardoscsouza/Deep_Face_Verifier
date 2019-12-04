from tensorflow.keras.models import load_model, Model
from feature_extractor import get_generator
from glob import glob
import numpy as np
import shutil
import os



def extract_all(in_dir, out_dir, model_filepath):
    extractor = load_model(model_filepath)
    extractor = Model(extractor.input, extractor.get_layer(name="encod_dense").output)
    generator = get_generator(in_dir, batch_size=256, image_size=(32, 32),
                            preprocessing_function=None, rescale=1.0/255.0)

    if os.path.exists(out_dir):
        print("Removing output directory. Continue? [y/n]")
        if input().lower() == 'y':
            print("Deleting...")
            shutil.rmtree(out_dir)
        else:
            print("Aborting...")
            return
    os.makedirs(out_dir)

    folds_files = [fold_file.split(os.sep)[-1] for fold_file in glob(os.path.join(in_dir, "fold_*.txt"))]
    for fold_file in folds_files:
        shutil.copyfile(os.path.join(in_dir, fold_file), os.path.join(out_dir, fold_file))

    classes_map = {val : key for key, val in generator.class_indices.items()}
    for _ in range(len(generator)):
        x, y = generator.next()
        out = extractor.predict(x)
        for cur_sample, cur_class in zip(out, y):
            cur_out_file = os.path.join(out_dir, classes_map[cur_class]+".npy")
            if os.path.isfile(cur_out_file):
                cur_arr = np.load(cur_out_file)
                cur_arr = np.concatenate((cur_arr, np.expand_dims(cur_sample, axis=0)), axis=0)
            else:
                cur_arr = np.expand_dims(cur_sample, axis=0)
            np.save(cur_out_file, cur_arr)



if __name__ == '__main__':
    n_folds = 5
    for i in range(n_folds):
        extract_all("dataset",
                    os.path.join("autoencoder_extracted", "fold_{}".format(i)),
                    os.path.join("experiments", "autoencoder", "fold_{}".format(i), "best_model.h5"))