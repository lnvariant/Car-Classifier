import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import ConnectionPatch
import scipy.io
import os
import random
import numpy as np

from yolo import get_bounding_images


def draw_matches(img1, keypoints1, img2, keypoints2, plot_title=""):
    """
    Draws matches of keypoints between img1 and img2.

    :param img1: the first image
    :param keypoints1: the first image's keypoints
    :param img2: the second image
    :param keypoints2: the second image's keypoints
    :param plot_title: the title of the plot
    """
    figure = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    img1 = cv2.drawKeypoints(img1, keypoints1, None)
    img2 = cv2.drawKeypoints(img2, keypoints2, None)
    ax1.imshow(img1)
    ax2.imshow(img2)
    for kp1, kp2 in zip(keypoints1, keypoints2):
        con = ConnectionPatch(xyA=kp2.pt, xyB=kp1.pt,
                              coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color=np.random.rand(3, ))
        ax2.add_patch(con)

    plt.title(plot_title)
    plt.show()
    figure.savefig("data/results/" + plot_title.replace(" ", "-").replace(".", "-") + '.png', dpi=100,
                   bbox_inches='tight')


def sift_keypt_extractor(img1, img2, ratio=0.7, max_matches=-1, visualize=False, max_features=-1):
    """
    Detects and shows SIFT image features in the two images, then shows
    the matching features between the images.

    The algorithm used in this method is from the OpenCV documentation:

    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html

    :param img1: the first grayscale image
    :param img2: the second grayscale image
    :param ratio: Lowe's Ratio for comparing matches
    :param max_matches: max number of matches to get
    :param visualize: whether or not to visualize the matches
    :param max_features: max number of features SIFT is allowed to detect
    """
    sift = cv2.xfeatures2d.SIFT_create(max_features) if max_features > 0 else cv2.xfeatures2d.SIFT_create()

    img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    try:
        kp1, des1 = sift.detectAndCompute(img1_g, None)
        kp2, des2 = sift.detectAndCompute(img2_g, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        pts1 = []
        pts2 = []
        filtered_kp1 = []
        filtered_kp2 = []

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < ratio * n.distance:
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
                filtered_kp1.append(kp1[m.queryIdx])
                filtered_kp2.append(kp2[m.trainIdx])

            if max_matches > 0 and len(pts1) > max_matches - 1:
                break

        if visualize:
            draw_matches(img1, filtered_kp1, img2, filtered_kp2, plot_title="")

        return kp1, kp2, pts1, pts2
    except:
        return None, None, None, None


def get_make_model_images_path(imgs_path, make, model):
    """
    Returns the path to images of the make and model.

    :param imgs_path: leads to a folder that has folders labeled with make ids. Each of those folders has folders
                      labeled by the model ids.
    :param make: the make of the image
    :param model: the model of the image (including year)
    :return:
    """
    make_dict = scipy.io.loadmat("data\\misc\\make_model_name.mat")

    make_id = -1
    model_id = -1
    just_model = ""  # The model without the year

    # Find the index of the predicted make
    for i in range(1, make_dict["make_names"].shape[0]):
        dict_make = make_dict["make_names"][i - 1][0][0]

        if make == dict_make:
            make_id = i
            break

    # Find the index of the predicted model
    for i in range(1, make_dict["model_names"].shape[0]):
        dict_model = make_dict["model_names"][i - 1][0]

        if dict_model.shape[0] != 0:
            if model in dict_model[0] or dict_model[0] in model:
                model_id = i
                just_model = dict_model[0]
                break

    year = model.replace(just_model, "").strip()

    return imgs_path + "/" + str(make_id) + "/" + str(model_id) + "/" + year


def get_random_image_path(imgs_path):
    """
    Returns a random image from the given path.

    :param imgs_path: a path to a folder of images (> 0)
    """
    img_files = os.listdir(imgs_path)

    if len(img_files) < 1:
        raise Exception("No images found pertaining to the given make and mode.")

    img_path = imgs_path + "/" + str(img_files[random.randrange(0, len(img_files))])

    return img_path


def sift_authenticator(img, predicted_make, predicted_model):
    # Get a random image to compare to
    imgs_path = get_make_model_images_path("data/image", predicted_make, predicted_model)
    accuracies = []

    # Loop through all images in directory
    for f in os.listdir(imgs_path):
        img_path = imgs_path + "/" + f

        # Get just the cars from the image
        bounding_imgs = get_bounding_images(img_path)

        # For each car found, run SIFT
        for comparison_img in bounding_imgs:
            if img is not None and comparison_img is not None:
                keypts1, keypts2, good_keypts1, good_keypts2 = sift_keypt_extractor(img, comparison_img,
                                                                                    visualize=False)

                # Make sure we receive valid keypoints
                if keypts1 is not None and good_keypts1 is not None:
                    num_of_keypts = min(len(keypts1), len(keypts2))
                    num_of_good_keypts = min(len(good_keypts1), len(good_keypts2))
                    accuracies.append(num_of_good_keypts / num_of_keypts)

    total_accuracy = (np.sum(accuracies) / len(accuracies)) * 100
    print(total_accuracy)

    return total_accuracy
