import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision import transforms
from datasets import load_dataset
import torchvision.models as models

ds = load_dataset("naufalso/stanford_cars")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Build a complete label -> car_name mapping with 196 entries
# Build label -> name dictionary first
label_to_name = {}

# Track the highest label index
max_label = 0

for example in ds["train"]:
    label = example["label"]
    name = example["car_name"]
    label_to_name[label] = name
    if label > max_label:
        max_label = label

# Then build car_names list in correct order
car_names = [label_to_name.get(i, f"Unknown label {i}") for i in range(max_label + 1)]


# Load your model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 196)  # Adjust output layer size based on class names
    checkpoint = torch.load("pages/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=False)  # Allow size mismatch
    model.eval()
    model.to(device)
    return model

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict(image, model):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, predicted = torch.max(probs, 1)
    return predicted.item(), conf.item() * 100

# Streamlit UI
st.title("🚗 Car Model Classifier")
st.write("Upload a car image or enter an image URL to classify it.")

# Load the model
model = load_model()

# Radio input method
option = st.radio("Select input method:", ["Upload Image", "Image URL"])

img = None

# Upload method
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

# URL method
elif option == "Image URL":
    image_url = st.text_input("Enter image URL:")
    if image_url:
        try:
            response = requests.get(image_url)
            if 'image' in response.headers['Content-Type']:
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                st.error("The provided URL does not seem to be an image.")
        except Exception as e:
            st.error(f"Unable to load image. Error: {e}")

# Show image and prediction
if img:
    st.image(img, caption="Input Image", use_container_width=True)
    if st.button("Predict"):
        pred, acc = predict(img, model)
        try:
            st.success(f"Prediction: **{car_names[pred]}**")
        except IndexError:
            st.error(f"Prediction index {pred} out of range for car_names.")
        st.info(f"Confidence: **{acc:.2f}%**")
