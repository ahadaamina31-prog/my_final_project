# Import required libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("digit_model.h5", compile=False)

# Title
st.title("Handwritten Digit Recognition System")
st.write("Upload an image of a handwritten digit (0-9) to predict the digit.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess image
    image = image.resize((28, 28))
    image_array = np.array(image)

    # Invert colors if needed
    image_array = 255 - image_array

    # Normalize
    image_array = image_array / 255.0

    # Reshape for CNN
    image_array = image_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    # Display result
    st.success(f"Predicted Digit: {predicted_digit}")

    # Confidence score
    confidence = np.max(prediction) * 100
    st.write(f"Confidence: {confidence:.2f}%")