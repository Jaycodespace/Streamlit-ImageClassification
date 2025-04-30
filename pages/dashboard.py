import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.title("Image Prediction App")

# Input for image URL
image_url = st.text_input("Enter Image URL", "")

# Display image if URL is provided
if image_url:
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Submitted Image", use_container_width=True)
    except:
        st.error("Unable to load image. Please check the URL.")

# Predict button
if st.button("Predict"):
    if image_url:
        # Replace this with actual prediction logic using your trained model
        st.success("Prediction: [????????]")
    else:
        st.warning("Please provide a valid image URL.")
