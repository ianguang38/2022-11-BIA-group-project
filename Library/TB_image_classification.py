import numpy as np
import tensorflow as tf
import keras
from keras import backend

def classify(images):
    """
    classify a batch of images, return a tuple (probability list, diagnosis list)
    for correct prediction of the current model, please input images ranging from 0 to 255
    """
    # Loss function used in the model
    def binary_focal_loss(gamma=2, alpha=0.25):
        alpha = tf.constant(alpha, dtype=tf.float32)
        gamma = tf.constant(gamma, dtype=tf.float32)

        def binary_focal_loss_fixed(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            alpha_t = y_true*alpha + (backend.ones_like(y_true)-y_true)*(1-alpha)
            p_t = y_true*y_pred + (backend.ones_like(y_true)-y_true) * \
                (backend.ones_like(y_true)-y_pred) + backend.epsilon()
            focal_loss = - alpha_t * \
                backend.pow((backend.ones_like(y_true)-p_t), gamma) * backend.log(p_t)
            return backend.mean(focal_loss)
        return binary_focal_loss_fixed

    # Load model
    model = keras.models.load_model("./Models/Classification - Inception V3.h5",
                                    custom_objects={'binary_focal_loss_fixed': binary_focal_loss()})

    # Prepare prediction dataset
    images_RGB = []
    for image in images:
        # Convert to RGB
        if len(image.shape) == 2:
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
        images_RGB.append(image)
    images_RGB = np.array(images_RGB).reshape(-1, 128, 128, 3)

    # Make predictions
    probability = model.predict(images_RGB)
    diagnosis_num = np.where(probability >= 0.5, 1, 0).flatten().tolist()
    diagnosis = ['Normal' if i == 0 else "Tuberculosis" for i in diagnosis_num]

    return diagnosis, probability[:, 0].tolist()
