import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard



def train_model(model, train_datagen, test_datagen, epochs, 
                tensorboard_logdir="logs", best_model_filepath="best_model.h5",
                steps_per_epoch=500, validation_steps=100):
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
                            patience=15,
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

def evaluate_model(model, train_datagen, test_datagen, evaluation_steps=1000):
    eval_args = dict(steps=evaluation_steps,
                    callbacks=None,
                    max_queue_size=20,
                    workers=2,
                    use_multiprocessing=False,
                    verbose=False)

    train_scores = model.evaluate_generator(train_datagen, **eval_args)
    test_scores = model.evaluate_generator(test_datagen, **eval_args)

    return train_scores, test_scores

'''
def run_experiment(exp_name, transfer_type, epochs=150):
    best_model_filepath = os.path.join(best_models_dir, exp_name + ".h5")
    tensorboard_logdir = os.path.join(logs_dir, exp_name)

    clear_session()
    model = get_mobilenet_trained_model(transfer_type=transfer_type)
    model.summary()

    train_datagen, test_datagen = get_generators(train_dir=train_dir, test_dir=test_dir)
    assert train_datagen.class_indices == test_datagen.class_indices

    train_model(model, train_datagen, test_datagen, epochs, 
    tensorboard_logdir=tensorboard_logdir,
    best_model_filepath=best_model_filepath)

    model = load_model(best_model_filepath)
    scores = evaluate_model(model, train_datagen, test_datagen)
    print("Train Loss:\t{}\t\tTrain Accuracy:\t{}".format(scores[0][0], scores[0][1]))
    print("Test  Loss:\t{}\t\tTest  Accuracy:\t{}".format(scores[1][0], scores[1][1]))
'''