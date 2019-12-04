from tensorflow.keras.backend import clear_session
from glob import glob
from generator import FacePairGenerator
from training import train_model, evaluate_model
import tempfile
import build_model
import os
import numpy as np
import pandas as pd



def run_autoencoder_model_experiment(model, dataset_dir, exp_name,
                batch_size=32, epochs=200, steps_per_epoch=100,
                validation_steps=20, early_stop_patience=15, evaluation_steps=500,
                experiments_dir="experiments", tensorboard_logs_dir="tensorboard_logs",
                earlystop_metric="loss", checkpoint_metric="val_binary_accuracy"):
    outfiles_dir = os.path.join(experiments_dir, exp_name)
    tensorboard_logdir = os.path.join(tensorboard_logs_dir, exp_name)

    metrics_cols = [' '.join([word.capitalize() for word in metric.split('_')]) for metric in model.metrics_names]
    metrics_df = pd.DataFrame(columns=["Fold", "Set"] + metrics_cols)
    n_folds = len(os.listdir(dataset_dir))
    with tempfile.TemporaryDirectory() as temp_dir:
        init_weights_file = os.path.join(temp_dir, "init_weights.h5")
        temp_weights_file = os.path.join(temp_dir, "temp_weights.h5")
        model.save_weights(init_weights_file)
        for i in range(n_folds):
            fold_name = "fold_{}".format(i)
            folds_files = glob(os.path.join(dataset_dir, fold_name, "fold_*.txt"))
            model.load_weights(init_weights_file)

            cur_outfiles_dir = os.path.join(outfiles_dir, fold_name)
            if not os.path.exists(cur_outfiles_dir):
                os.makedirs(cur_outfiles_dir)
            cur_tensorboard_logdir = os.path.join(tensorboard_logdir, fold_name)
            if not os.path.exists(cur_tensorboard_logdir):
                os.makedirs(cur_tensorboard_logdir)

            train_datagen = FacePairGenerator([folds_files[j] for j in range(len(folds_files)) if j != i], batch_size=batch_size)
            test_datagen = FacePairGenerator([folds_files[i]], batch_size=batch_size)

            train_model(model, train_datagen, test_datagen, 
                    epochs=epochs, steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps, early_stop_patience=early_stop_patience,
                    tensorboard_logdir=cur_tensorboard_logdir, best_model_filepath=temp_weights_file,
                    earlystop_metric=earlystop_metric, checkpoint_metric=checkpoint_metric)

            model.load_weights(temp_weights_file)
            train_metrics, val_metrics = evaluate_model(model, train_datagen, test_datagen, evaluation_steps=evaluation_steps)
            model.save(os.path.join(cur_outfiles_dir, "best_model.h5"),
                    overwrite=True, include_optimizer=False, save_format='h5')

            cur_metrics_df = pd.DataFrame([train_metrics, val_metrics], columns=metrics_cols)
            cur_metrics_df.insert(0, "Set", ["Train", "Val"])
            cur_metrics_df.to_csv(os.path.join(cur_outfiles_dir, "metrics.csv"), index=False)

            cur_metrics_df.insert(0, "Fold", [i, i])
            metrics_df = pd.concat([metrics_df, cur_metrics_df], ignore_index=True)
        
    metrics_df.to_csv(os.path.join(outfiles_dir, "metrics.csv"), index=False)



if __name__ == '__main__':
    from itertools import product
    import random

    from time import time
    t0 = time()

    expr_out_dir = "experiments"
    logs_out_dir = "tensorboard_logs"
    extracted_dir = "autoencoder_extracted"

    models_types = ["eucl", "cos"]
    combination_funcs = [0, 2, 4]
    n_neurons = [16, 64]
    n_layers = np.arange(2)
    combs = list(product(models_types, combination_funcs, n_neurons, n_layers))
    random.shuffle(combs)
    for m_type, comb_f, n_neur, n_lay in combs:
        n_neur = 0 if (n_lay == 0) else n_neur
        exp_name = "autoencoder_{}_combFun:{}_nNeur:{}_nLay:{}".format(m_type, comb_f, n_neur, n_lay)
        if not os.path.isdir(os.path.join(expr_out_dir, exp_name)):
            clear_session()
            model = build_model.build_eucl_model(128, comb_f, n_neur, n_lay, True) if m_type == "eucl" else build_model.build_cos_model(128, comb_f, n_neur, n_lay, True)
            run_autoencoder_model_experiment(model, extracted_dir, exp_name,
                        experiments_dir=expr_out_dir, tensorboard_logs_dir=logs_out_dir)
    
    print("Total Time:", time() - t0)