import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# ----- Load model -----
model_path = "/Users/arghaauddy/Desktop/ISL Project/models/resnet_model_with_aug.h5"
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

# ----- Prediction function -----
def predict_image(img_pil):
    img_resized = img_pil.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return class_indices[predicted_class], confidence

# ----- Streamlit UI -----
st.title("🛰️ Land Cover Image Classifier")

st.write("Upload a satellite or aerial image to classify it into one of 5 classes.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)


    if st.button("Predict"):
        label, conf = predict_image(img)
        st.success(f"### Predicted class: **{label}** \nConfidence: **{conf:.2f}**")
