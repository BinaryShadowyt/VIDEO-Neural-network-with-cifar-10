import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("cifar10_cnn_model.h5")

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def predict_image(img_path):

    img = image.load_img(img_path, target_size=(32, 32))

    img_array = image.img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    print(predictions)

    predicted_class = np.argmax(predictions)

    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class]}")
    plt.axis('off')
    plt.show()

    return class_names[predicted_class]


image_path = ""
predicted_class = predict_image(image_path)
print(f"Predicted class: {predicted_class}")
