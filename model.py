import keras
from keras.models import Model
from keras.layers import *

MAIN_DIR_PATH = "drive/My Drive/Colab Notebooks/Car Classifier/"
LOGS = MAIN_DIR_PATH + "/logs"
IMG_SHAPE = (224, 224)


def get_model(number_of_classes):
    base_model = keras.applications.MobileNet(weights='imagenet',
                                              include_top=False,
                                              input_shape=(224, 224, 3))

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    prediction = Dense(number_of_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=prediction)

    return model
