import build_model
import training
import generator
from tensorflow.keras.preprocessing.image import save_img

model = build_model.build_autoencoder()
model.summary()

train_datagen = generator.get_autoencoder_generator(["autoencoder_dataset/fold_0.txt"], image_size=(32, 32))
test_datagen = generator.get_autoencoder_generator(["autoencoder_dataset/fold_1.txt"], image_size=(32, 32))

training.train_model(model, train_datagen, test_datagen, epochs=20,
            earlystop_metric="loss", checkpoint_metric="val_loss")

model.load_weights("best_model.h5")
out = model.predict(test_datagen.next())
for i in range(len(out)):
    save_img("coder_imgs/{}.jpg".format(i), out[i])