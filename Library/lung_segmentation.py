# Requirements
#   Python 3.10
#   numpy, matplotlib, opencv-python, tensorflow, keras

import numpy as np
import cv2
from keras import backend
from keras.models import load_model

def preprocess(image_files, image_size = 128):

    images = []
    mean = 0
    std = 0
    count = 0

    for image_file in image_files:
        image = np.mean(cv2.resize(cv2.imread(image_file), (image_size, image_size)), axis = 2)

        mean += image[:, :].mean()
        std += image[:, :].std()
        count += 1

        images.append(image)

    mean = mean / count
    std = std / count
    for i in range(len(images)):
        images[i] = (np.array(images[i]) - mean) / std
    return images

def segment(images):

    def dice_coef(y_true, y_pred):
        y_true_f = backend.flatten(y_true)
        y_pred_f = backend.flatten(y_pred)
        intersection = backend.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + 1) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + 1)

    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)

    model = load_model("./Models/Segmentation - U-Net.h5", custom_objects = {'dice_coef' : dice_coef, 'dice_coef_loss' : dice_coef_loss})
    segmented_images = []
    qualities = []
    predictions = model.predict(np.array(images))

    for index, pred in enumerate(predictions):
        lung = np.squeeze(pred)
        positive = lung > 0.01
        segmented_image = images[index]
        count = segmented_image.shape[0] * segmented_image.shape[1]
        for i in range(segmented_image.shape[0]):
            for j in range(segmented_image.shape[1]):
                if not positive[i, j]:
                    segmented_image[i, j] = 0
                    count -= 1
        quality = count / (segmented_image.shape[0] * segmented_image.shape[1])
        segmented_images.append(segmented_image)
        qualities.append(quality)
    return segmented_images, qualities