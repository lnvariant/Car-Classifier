import scipy.io
import os
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

from model import IMG_SHAPE


def build_data_frame():
    make_model = scipy.io.loadmat("make_model_name.mat")
    white_listed_makes = ['Ferrari', 'Maserati', 'Acura', 'McLaren', 'Mustang', 'Aston Martin', 'TESLA', 'Porsche',
                          'Lamorghini ', 'Benz', 'Audi', 'BWM', 'Bentley', 'Bugatti', 'Jaguar']
    single_make = "BWM"
    files_paths = []
    for root, dirs, files in os.walk("data\\image"):
        for name in files:
            files_paths.append(os.path.join(root, name))

    training = []

    for f in files_paths:
        data_p = f.split("\\")
        make = make_model["make_names"][int(data_p[2]) - 1][0][0]
        model = make_model["model_names"][int(data_p[3]) - 1][0][0]
        year = data_p[4]

        if make != single_make:
            continue

        model = model.replace(single_make, "")
        model = single_make + " " + model.strip()
        model = model.replace("BWM", "BMW")

        f = f.replace("\\", "/")
        row = {"filename": f, "label": model + " " + str(year)}

        training.append(row)

    df = pd.DataFrame(training)
    df.to_csv("data\\bmw_data.csv")

    train, validation = train_test_split(df, test_size=0.1, stratify=df["label"])
    train, test = train_test_split(train, test_size=0.1 / 0.9)

    train.to_csv("data\\bmw_training_data.csv")
    validation.to_csv("data\\bmw_validation_data.csv")
    test.to_csv("data\\bmw_test_data.csv")

    return train, validation


def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


def preprocess_image(img, img_shape):
    img = cv2.resize(img, img_shape)
    img = np.array(img, dtype=np.float64)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img


def get_data_frames(csv_training_path, csv_validation_path, csv_test_path):
    training_df = pd.read_csv(csv_training_path)
    validation_df = pd.read_csv(csv_validation_path)
    test_df = pd.read_csv(csv_test_path)

    return training_df, validation_df, test_df


def get_generators(images_directory, training_df, validation_df, test_df):
    datagen = ImageDataGenerator(rescale=1. / 255)
    training_generator = datagen.flow_from_dataframe(dataframe=training_df,
                                                     directory=images_directory,
                                                     x_col="filename",
                                                     y_col="label",
                                                     target_size=IMG_SHAPE,
                                                     color_mode='rgb',
                                                     batch_size=64,
                                                     class_mode="categorical")

    validation_generator = datagen.flow_from_dataframe(dataframe=validation_df,
                                                       directory=images_directory,
                                                       x_col="filename",
                                                       y_col="label",
                                                       target_size=IMG_SHAPE,
                                                       color_mode='rgb',
                                                       batch_size=64,
                                                       class_mode="categorical")

    test_generator = datagen.flow_from_dataframe(dataframe=test_df,
                                                 directory=images_directory,
                                                 x_col="filename",
                                                 y_col="label",
                                                 target_size=IMG_SHAPE,
                                                 color_mode='rgb',
                                                 batch_size=64,
                                                 class_mode="categorical")

    return training_generator, validation_generator, test_generator



if __name__ == "__main__":
    training_df, validation_df = build_data_frame()
    # x_train, y_train, x_valid, y_valid = get_data("", training_df, validation_df)
    print()
