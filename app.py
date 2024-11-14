import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the MobileNetV2 model with custom top layers for 26-class sign language classification
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(26, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Label dictionary for sign language alphabets
labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
          10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
          19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

def load_and_preprocess_image(file_path):
    """Load an image and preprocess it for MobileNetV2"""
    img = load_img(file_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_sign_language(file_path):
    """Predict the sign language alphabet from the image file"""
    img = load_and_preprocess_image(file_path)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    predicted_label = labels[class_index]
    return predicted_label

def main():
    # Specify the image file path (replace 'your_image_path' with your image file path)
    file_path = 'images.png'  # e.g., "test_image.jpg"
    
    if not os.path.isfile(file_path):
        print(f"Error: The file {file_path} was not found. Please provide a valid image path.")
        return
    
    # Perform prediction and display the result
    print(f'Processing {file_path}...')
    predicted_sign = predict_sign_language(file_path)
    print(f"Predicted Sign: {predicted_sign}")

    # Display the image with the predicted label
    img = load_img(file_path, target_size=(224, 224))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted Sign: {predicted_sign}')
    plt.show()

# Run the main function
if __name__ == '__main__':
    main()
