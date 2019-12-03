from keras_preprocessing.image import ImageDataGenerator

from dataloader import preprocess_image
from model import IMG_SHAPE
from yolo import get_bounding_images, get_bounding_images_p
import cv2
from sift_auth import sift_authenticator
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

make_label_map = {0: 'Acura', 1: 'Aston Martin', 2: 'Audi', 3: 'BMW', 4: 'Bentley', 5: 'Benz', 6: 'Bugatti',
                  7: 'Ferrari', 8: 'Jaguar', 9: 'Lamorghini ', 10: 'Maserati', 11: 'McLaren', 12: 'Mustang',
                  13: 'Porsche', 14: 'TESLA'}

bmw_label_map = {0: 'BMW 1 Series M 2011', 1: 'BMW 1 Series convertible 2011', 2: 'BMW 1 Series couple 2011',
                 3: 'BMW 1 Series couple 2013', 4: 'BMW 1 Series hatchback 2010', 5: 'BMW 1 Series hatchback 2012',
                 6: 'BMW 1 Series hatchback 2013', 7: 'BMW 2 Series 2014', 8: 'BMW 2 Series 2015',
                 9: 'BMW 2 Series Active Tourer 2014', 10: 'BMW 2 Series Active Tourer 2015', 11: 'BMW 3 Series 2008',
                 12: 'BMW 3 Series 2009', 13: 'BMW 3 Series 2010', 14: 'BMW 3 Series 2011', 15: 'BMW 3 Series 2012',
                 16: 'BMW 3 Series 2013', 17: 'BMW 3 Series 2014', 18: 'BMW 3 Series GT 2013',
                 19: 'BMW 3 Series GT 2014', 20: 'BMW 3 Series convertible 2008', 21: 'BMW 3 Series convertible 2010',
                 22: 'BMW 3 Series convertible 2011', 23: 'BMW 3 Series convertible 2012',
                 24: 'BMW 3 Series coupe 2007', 25: 'BMW 3 Series coupe 2009', 26: 'BMW 3 Series coupe 2011',
                 27: 'BMW 3 Series estate 2013', 28: 'BMW 3 Series estate 2014', 29: 'BMW 3 Series hybrid 2012',
                 30: 'BMW 3 Series hybrid 2013', 31: 'BMW 4 Series 2014', 32: 'BMW 4 Series convertible 2014',
                 33: 'BMW 4 Series couple 2013', 34: 'BMW 4 Series couple 2014', 35: 'BMW 5 Series 2009',
                 36: 'BMW 5 Series 2010', 37: 'BMW 5 Series 2011', 38: 'BMW 5 Series 2012', 39: 'BMW 5 Series 2013',
                 40: 'BMW 5 Series 2014', 41: 'BMW 5 Series GT 2010', 42: 'BMW 5 Series GT 2011',
                 43: 'BMW 5 Series GT 2013', 44: 'BMW 5 Series GT 2014', 45: 'BMW 5 Series GT 2015',
                 46: 'BMW 5 Series estate 2012', 47: 'BMW 5 Series estate 2013', 48: 'BMW 5 Series estate 2014',
                 49: 'BMW 5 Series hybrid 2013', 50: 'BMW 5 Series hybrid 2014', 51: 'BMW 5 Series hybrid 2015',
                 52: 'BMW 6 Series 2012', 53: 'BMW 6 Series 2013', 54: 'BMW 6 Series 2014',
                 55: 'BMW 6 Series convertible 2008', 56: 'BMW 6 Series convertible 2011',
                 57: 'BMW 6 Series couple 2008', 58: 'BMW 6 Series couple 2010', 59: 'BMW 6 Series couple 2012',
                 60: 'BMW 6 Series couple 2013', 61: 'BMW 7 Series 2009', 62: 'BMW 7 Series 2010',
                 63: 'BMW 7 Series 2011', 64: 'BMW 7 Series 2013', 65: 'BMW 7 Series 2014', 66: 'BMW 7 Series 2015',
                 67: 'BMW 7 Series hybrid 2010', 68: 'BMW 7 Series hybrid 2012', 69: 'BMW 7 Series hybrid 2013',
                 70: 'BMW Active Tourer 2012', 71: 'BMW Active Tourer 2013', 72: 'BMW Active Tourer 2015',
                 73: 'BMW ConnectedDrive 2011', 74: 'BMW EfficientDynamics 2009', 75: 'BMW GINA 2008',
                 76: 'BMW Gran Lusso 2013', 77: 'BMW Isetta 1955', 78: 'BMW M3 2008', 79: 'BMW M3 2014',
                 80: 'BMW M3 convertible 2008', 81: 'BMW M3 coupe 2008', 82: 'BMW M3 coupe 2010',
                 83: 'BMW M3 coupe 2011', 84: 'BMW M3 coupe 2013', 85: 'BMW M4 coupe 2014', 86: 'BMW M5 2005',
                 87: 'BMW M5 2011', 88: 'BMW M5 2012', 89: 'BMW M5 2013', 90: 'BMW M5 2014', 91: 'BMW M6 2013',
                 92: 'BMW M6 2014', 93: 'BMW M6 coupe 2008', 94: 'BMW M6 coupe 2013', 95: 'BMW M6 coupe 2014',
                 96: 'BMW Vision Future Luxury 2014', 97: 'BMW X1 2010', 98: 'BMW X1 2012', 99: 'BMW X1 2013',
                 100: 'BMW X1 2014', 101: 'BMW X3 2009', 102: 'BMW X3 2010', 103: 'BMW X3 2011', 104: 'BMW X3 2012',
                 105: 'BMW X3 2013', 106: 'BMW X3 2014', 107: 'BMW X4 2013', 108: 'BMW X4 2014', 109: 'BMW X4 2015',
                 110: 'BMW X5 2008', 111: 'BMW X5 2009', 112: 'BMW X5 2011', 113: 'BMW X5 2012', 114: 'BMW X5 2013',
                 115: 'BMW X5 2014', 116: 'BMW X5 M 2010', 117: 'BMW X5 M 2013', 118: 'BMW X5 M 2014',
                 119: 'BMW X5 M 2016', 120: 'BMW X6 2010', 121: 'BMW X6 2011', 122: 'BMW X6 2012', 123: 'BMW X6 2013',
                 124: 'BMW X6 2014', 125: 'BMW X6 2015', 126: 'BMW X6 M 2010', 127: 'BMW X6 M 2013',
                 128: 'BMW X6 M 2014', 129: 'BMW X6 M 2015', 130: 'BMW X6 M 2016', 131: 'BMW X6 Series hybrid 2010',
                 132: 'BMW Z4 2009', 133: 'BMW Z4 2011', 134: 'BMW Z4 2012', 135: 'BMW Z4 2013', 136: 'BMW Z4 2014',
                 137: 'BMW Zagato Coupe 2012', 138: 'BMW i3 2012', 139: 'BMW i3 2013', 140: 'BMW i3 2014',
                 141: 'BMW i8 2011', 142: 'BMW i8 2012', 143: 'BMW i8 2013', 144: 'BMW i8 2014', 145: 'BMW i8 2015'}


def get_model_pred(model, img, label_map):
    make_result = model.predict(img)
    make_preds = np.argsort(-make_result)[:1].flatten()
    make_pred_label = label_map[make_preds[0]]

    return make_pred_label


def main():
    # Initialize label maps

    # Load MobileNetV2 models
    make_model = keras.models.load_model("models/make_model.h5")
    bmw_model = keras.models.load_model("models/bmw_model.h5")

    test_df = pd.read_csv("data/bmw_test_data.csv")

    for index, row in test_df.iterrows():

        if row["label"] is not None and row["filename"] is not None:
            try:
                actual_label = row["label"]
                if "BMW" not in actual_label:
                    continue

                # Run YOLOv3 to get cropped images of cars
                img, cropped_imgs = get_bounding_images_p(row["filename"])

                if img is None or len(cropped_imgs) == 0:
                    continue

                car_img = cropped_imgs[0][0]

                # Pre-process image to be inputted into the models
                car_img_p = preprocess_image(car_img, IMG_SHAPE)

                pred_make = get_model_pred(make_model, car_img_p, make_label_map)
                pred_model = ""

                # If the predicted label is BMW then us the BMW model
                if pred_make == "BMW":
                    pred_model = get_model_pred(bmw_model, car_img_p, bmw_label_map)
                else:
                    continue

                pred_model = pred_model.replace(pred_make, "")
                pred_model = pred_make + pred_model

                # label_map = training_generator.class_indices
                # label_map = dict((v, k) for k, v in label_map.items())

                # for i in range(result.shape[1]):
                #     print(label_map[i] + " : " + str(result[0][i]))

                # accuracy = sift_authenticator(cropped_imgs[0][0], "BWM", "BWM X3 2014")
                accuracy = 95

                # Draw the bounding box, model text, and accuracy text
                x, y, w, h = cropped_imgs[0][1]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(img, pred_model, (x + 10, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(img, "ACTUAL: " + actual_label, (x + 10, y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # cv2.putText(img, "CONFIDENCE: " + str(accuracy) + "%", (x + w - 180, y + h - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                plt.imshow(img)
                plt.show()
            except Exception as inst:
                print("Failed to process an image.")
                print(inst)



main()