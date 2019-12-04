import build_model
import training

model = build_model.build_autoencoder()
training.run_autoencoder_experiment(model, "dataset", "autoencoder_dataset", "autoencoder",
                        experiments_dir="experiments", tensorboard_logs_dir="tensorboard_logs",
                        earlystop_metric="loss", checkpoint_metric="val_loss")