# Requirements
#   Python 3.10
#   numpy, matplotlib, opencv-python, tensorflow, keras

from Library.lung_segmentation import preprocess, segment
from Library.TB_image_classification import classify
import matplotlib.pyplot as plt
import cv2

image_file = "./Test.png"
image_files = [image_file, image_file]
try:
    images = preprocess(image_files)
except cv2.error:
    print("Error: Image not found or not in a regular format")
    exit()

segmented_images, qualities = segment(images)
for i in range(len(image_files)):
    segmented_image = segmented_images[i]
    quality = qualities[i]
    if quality < 0.1:
        print("Error: Low input quality from", image_files[i])
        continue
    else:
        print(quality)
        plt.imshow(segmented_image)
        plt.show()

diagnosis, probability = classify(segmented_images)
print(diagnosis)