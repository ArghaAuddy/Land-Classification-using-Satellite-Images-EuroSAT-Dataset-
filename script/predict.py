import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ----- Choose your model -----
model_path = "/Users/arghaauddy/Desktop/ISL Project/models/mobilenet_model.h5"
# OR
# model_path = "/Users/arghaauddy/Desktop/ISL Project/models/mobilenet_model.h5"

model = tf.keras.models.load_model(model_path)

# ----- Class index mapping -----
class_indices = {
    0: "Forest",
    1: "Industrial",
    2: "Pasture",
    3: "Residential",
    4: "River"
}

def predict_image(img_path):
    # Load and preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Print result
    print(f"Predicted class: {class_indices[predicted_class]} (Confidence: {confidence:.2f})")

# Example usage
if __name__ == "__main__":
    test_img = "/Users/arghaauddy/Desktop/ISL Project/dataset/val/Pasture/Pasture_96.jpg"  # ✅ Put your image path
    predict_image(test_img)
