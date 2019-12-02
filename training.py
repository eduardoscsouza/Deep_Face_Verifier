from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.backend import clear_session
from glob import glob
import os
import pandas as pd
import tempfile

from generator import FacePairGenerator
import build_model



def train_model(model, train_datagen, test_datagen, 
                epochs=200, steps_per_epoch=100, validation_steps=20, early_stop_patience=15,
                tensorboard_logdir="logs", best_model_filepath="best_model.h5"):
    tensorboard = TensorBoard(log_dir=tensorboard_logdir,
                            histogram_freq=0,
                            write_graph=False,
                            write_images=False,
                            update_freq='epoch',
                            profile_batch=2,
                            embeddings_freq=0,
                            embeddings_metadata=None)

    early_stop = EarlyStopping(monitor='loss',
                            min_delta=0.005,
                            patience=early_stop_patience,
                            verbose=True,
                            mode='min',
                            baseline=None,
                            restore_best_weights=False)

    checkpoint = ModelCheckpoint(best_model_filepath,
                            monitor="val_binary_accuracy",
                            verbose=True,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='max',
                            save_freq='epoch')

    model.fit_generator(train_datagen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=False,
                        callbacks=[tensorboard, early_stop, checkpoint],
                        validation_data=test_datagen,
                        validation_steps=validation_steps,
                        validation_freq=1,
                        class_weight=None,
                        max_queue_size=20,
                        workers=2,
                        use_multiprocessing=False,
                        shuffle=False,
                        initial_epoch=0)

def evaluate_model(model, train_datagen, test_datagen, evaluation_steps=500):
    eval_args = dict(steps=evaluation_steps,
                    callbacks=None,
                    max_queue_size=20,
                    workers=2,
                    use_multiprocessing=False,
                    verbose=False)

    train_scores = model.evaluate_generator(train_datagen, **eval_args)
    test_scores = model.evaluate_generator(test_datagen, **eval_args)

    return train_scores, test_scores

def run_experiment(model, dataset_dir, exp_name,
                batch_size=32, epochs=200, steps_per_epoch=100,
                validation_steps=20, early_stop_patience=15, evaluation_steps=500,
                experiments_dir="experiments", tensorboard_logs_dir="tensorboard_logs"):
    outfiles_dir = os.path.join(experiments_dir, exp_name)
    tensorboard_logdir = os.path.join(tensorboard_logs_dir, exp_name)

    metrics_cols = [' '.join([word.capitalize() for word in metric.split('_')]) for metric in model.metrics_names]
    metrics_df = pd.DataFrame(columns=["Fold", "Set"] + metrics_cols)
    folds_files = glob(os.path.join(dataset_dir, "fold_*.txt"))
    with tempfile.TemporaryDirectory() as temp_dir:
        init_weights_file = os.path.join(temp_dir, "init_weights.h5")
        model.save_weights(init_weights_file)
        for i in range(len(folds_files)):
            fold_name = "fold_{}".format(i)
            model.load_weights(init_weights_file)

            cur_outfiles_dir = os.path.join(outfiles_dir, fold_name)
            if not os.path.exists(cur_outfiles_dir):
                os.makedirs(cur_outfiles_dir)
            cur_tensorboard_logdir = os.path.join(tensorboard_logdir, fold_name)
            if not os.path.exists(cur_tensorboard_logdir):
                os.makedirs(cur_tensorboard_logdir)

            train_datagen = FacePairGenerator([folds_files[j] for j in range(len(folds_files)) if j != i], batch_size=batch_size)
            test_datagen = FacePairGenerator([folds_files[i]], batch_size=batch_size)

            cur_best_model_filepath = os.path.join(cur_outfiles_dir, "best_model.h5")
            train_model(model, train_datagen, test_datagen, 
                    epochs=epochs, steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps, early_stop_patience=early_stop_patience,
                    tensorboard_logdir=cur_tensorboard_logdir, best_model_filepath=cur_best_model_filepath)

            model.load_weights(cur_best_model_filepath)
            train_metrics, val_metrics = evaluate_model(model, train_datagen, test_datagen, evaluation_steps=evaluation_steps)
            model.save(cur_best_model_filepath)

            cur_metrics_df = pd.DataFrame([train_metrics, val_metrics], columns=metrics_cols)
            cur_metrics_df.insert(0, "Set", ["Train", "Val"])
            cur_metrics_df.to_csv(os.path.join(cur_outfiles_dir, "metrics.csv"), index=False)

            cur_metrics_df.insert(0, "Fold", [i, i])
            metrics_df = pd.concat([metrics_df, cur_metrics_df], ignore_index=True)
        
    metrics_df.to_csv(os.path.join(outfiles_dir, "metrics.csv"), index=False)



if __name__ == '__main__':
    clear_session()
    model = build_model.build_eucl_model(4096, 0, 2048, 2, True)
    model.summary()
    from time import time
    t0 = time()
    run_experiment(model, "extracted/layer_0/", "exp_1",
                experiments_dir="experiments",
                tensorboard_logs_dir="tensorboard_logs")
    print(time() - t0)