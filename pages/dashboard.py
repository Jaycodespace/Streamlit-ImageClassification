import streamlit as st
import base64
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision import transforms
import torchvision.models as models
import os

# Set page config
st.set_page_config(page_title="Car Model Classifier", layout="centered")

# Encode image to base64
def get_base64_bg(path):
    with open(path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

# Apply background image
def set_background(image_file):
    encoded_bg = get_base64_bg(image_file)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_bg}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Use it
try:
    set_background(os.path.join(os.path.dirname(__file__), "background3.jpg"))
except FileNotFoundError:
    st.warning("⚠️ Background image not found. Skipping background setup.")

# Load label names without using the problematic dataset loading
@st.cache_resource
def load_label_mapping():
    labels_path = os.path.join(os.path.dirname(__file__), "car_labels.txt")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Label file not found: {labels_path}")
    with open(labels_path, "r") as file:
        labels = [line.strip() for line in file.readlines()]
    return labels


# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 196)
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.to(device)
    return model

# Prediction transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image, model):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, predicted = torch.max(probs, 1)
    return predicted.item(), conf.item() * 100

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load label names and model

model = load_model()
car_names = load_label_mapping()

# Page content
st.title("🚗 Car Model Classifier")

# Introduction paragraph
st.write("""
Welcome to the **Car Model Classifier**! 🚗

This app allows you to classify car models from images using a **pretrained ResNet50** model. The model is trained on the **Stanford Cars dataset** available on Hugging Face, which includes 196 different car models. Whether you're a car enthusiast or just curious about a specific car, this app can help you identify the car models you encounter. 

Simply upload an image of a car or provide an image URL, and let the app do the rest! 🔍
""")

# Input method selection
option = st.radio("Select input method:", ["Upload Image", "Image URL"], horizontal=True)
img = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

elif option == "Image URL":
    image_url = st.text_input("Enter image URL:", key="url_input")
    if st.button("Enter URL"):
        if image_url:
            try:
                response = requests.get(image_url)
                if 'image' in response.headers['Content-Type']:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    st.session_state['img_from_url'] = img
                else:
                    st.error("The provided URL does not seem to be an image.")
            except Exception as e:
                st.error(f"Unable to load image. Error: {e}")
        else:
            st.warning("Please enter an image URL before pressing Enter.")
    elif 'img_from_url' in st.session_state:
        img = st.session_state['img_from_url']


# Prediction section
if img:
    st.subheader("🔍 Image Preview")
    st.image(img, caption="Input Image", use_container_width=True)

    if st.button("🧠 Predict"):
        with st.spinner("Classifying..."):
            pred, acc = predict(img, model)
        st.markdown("---")
        
        # If accuracy is below 20%, display "Not a car"
        if acc < 20:
            st.subheader("🚘 Predicted Car: Not a car")
        else:
            try:
                st.subheader(f"🚘 Predicted Car: {car_names[pred:]}")
            except IndexError:
                st.subheader(f"🚘 Predicted Car: Unknown (index {pred})")
        
        # Display confidence percentage
        st.write(f"🎯 Confidence: **{acc:.2f}%**")
        
        # Display confidence bar with green section indicating the confidence level
        st.markdown(f"""
            <div class="confidence-bar" style="width:100%; background-color: white;">
                <div style="height: 20px; width:{acc}%; background-color: #4CAF50; border-radius: 5px;"></div>
            </div>
        """, unsafe_allow_html=True)