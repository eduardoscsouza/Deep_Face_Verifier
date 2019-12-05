from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from feature_extractor import get_feature_extractor
import matplotlib.pyplot as plt
import build_model
import numpy as np
import os



model_filepath="final_model.h5"
imgs_dir = "example_imgs"
extractor = get_feature_extractor()
model = load_model(model_filepath)
def test_model(img_a, img_b):
    img_a = load_img(img_a, target_size=(224, 224), interpolation='bicubic')
    img_b = load_img(img_b, target_size=(224, 224), interpolation='bicubic')

    in_img_a = preprocess_input(img_to_array(img_a))
    in_img_b = preprocess_input(img_to_array(img_b))

    print("-----------------------------------------")
    v1, v2 = extractor.predict(np.stack([in_img_a, in_img_b], axis=0))[0]
    v1, v2 = np.expand_dims(v1, axis=0), np.expand_dims(v2, axis=0)
    if model.predict([v1, v2])[0] >= 0.5:
        print("Same Person")
    else:
        print("Different Person")
    
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(img_a)
    plt.xticks([]); plt.yticks([])
    f.add_subplot(1,2, 2)
    plt.imshow(img_b)
    plt.xticks([]); plt.yticks([])
    plt.show(block=True)

test_model(os.path.join(imgs_dir, "katy_perry_1.jpg"), os.path.join(imgs_dir, "katy_perry_2.jpg"))
test_model(os.path.join(imgs_dir, "katy_perry_3.jpg"), os.path.join(imgs_dir, "katy_perry_4.jpg"))
test_model(os.path.join(imgs_dir, "katy_perry_1.jpg"), os.path.join(imgs_dir, "zooey_deschanel_2.jpg"))
test_model(os.path.join(imgs_dir, "zooey_deschanel_1.jpg"), os.path.join(imgs_dir, "zooey_deschanel_2.jpg"))
test_model(os.path.join(imgs_dir, "zooey_deschanel_1.jpg"), os.path.join(imgs_dir, "katy_perry_2.jpg"))

test_model(os.path.join(imgs_dir, "keira_knightley_1.jpg"), os.path.join(imgs_dir, "keira_knightley_2.jpg"))
test_model(os.path.join(imgs_dir, "keira_knightley_1.jpg"), os.path.join(imgs_dir, "natalie_portman_2.jpg"))
test_model(os.path.join(imgs_dir, "natalie_portman_1.jpg"), os.path.join(imgs_dir, "natalie_portman_2.jpg"))
test_model(os.path.join(imgs_dir, "natalie_portman_1.jpg"), os.path.join(imgs_dir, "keira_knightley_2.jpg"))

test_model(os.path.join(imgs_dir, "neymar_1.jpg"), os.path.join(imgs_dir, "neymar_2.jpg"))
test_model(os.path.join(imgs_dir, "neymar_1.jpg"), os.path.join(imgs_dir, "neymar_3.jpg"))