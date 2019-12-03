from tensorflow.keras.backend import clear_session
import build_model
import training
import os



expr_out_dir = "experiments"
logs_out_dir = "tensorboard_logs"
extracted_dir = "extracted"

for ext_lay in range(3):
    exp_name = "base_outLay:{}".format(ext_lay)
    if not os.path.isdir(os.path.join(expr_out_dir, exp_name)):
        clear_session()
        input_size = 2622 if ext_lay == 2 else 4096
        model = build_model.build_base_model(input_size)
        training.run_experiment(model, os.path.join(extracted_dir, "layer_{}".format(ext_lay)), exp_name,
                    experiments_dir=expr_out_dir, tensorboard_logs_dir=logs_out_dir,
                    epochs=1, steps_per_epoch=1, validation_steps=1, early_stop_patience=1,
                    evaluation_steps=500)