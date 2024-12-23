import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("PId_Best.keras")

class_labels = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    if len(image_array.shape) == 2:  # If grayscale, add channel dimension
        image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

st.title("Flower Classification App")
st.write("Upload an image, and the model will predict whether it is a daisy, dandelion, rose, tulip, or sunflower.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Classifying...")
    input_size = model.input_shape[1:3]  # Extract input size from the model
    processed_image = preprocess_image(image, input_size)

    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    predicted_label = class_labels[predicted_class]

    st.write(f"Predicted Class: {predicted_label}")
