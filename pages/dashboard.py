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

car_names = [
    'AM General Hummer SUV 2000',
    'Acura Integra Type R 2001',
    'Acura RL Sedan 2012',
    'Acura TL Sedan 2012',
    'Acura TL Type-S 2008',
    'Acura TSX Sedan 2012',
    'Acura ZDX Hatchback 2012',
    'Aston Martin V8 Vantage Convertible 2012',
    'Aston Martin V8 Vantage Coupe 2012',
    'Aston Martin Virage Convertible 2012',
    'Aston Martin Virage Coupe 2012',
    'Audi 100 Sedan 1994',
    'Audi 100 Wagon 1994',
    'Audi A5 Coupe 2012',
    'Audi R8 Coupe 2012',
    'Audi RS 4 Convertible 2008',
    'Audi S4 Sedan 2007',
    'Audi S4 Sedan 2012',
    'Audi S5 Convertible 2012',
    'Audi S5 Coupe 2012',
    'Audi S6 Sedan 2011',
    'Audi TT Hatchback 2011',
    'Audi TT RS Coupe 2012',
    'Audi TTS Coupe 2012',
    'Audi V8 Sedan 1994',
    'BMW 1 Series Convertible 2012',
    'BMW 1 Series Coupe 2012',
    'BMW 3 Series Sedan 2012',
    'BMW 3 Series Wagon 2012',
    'BMW 6 Series Convertible 2007',
    'BMW ActiveHybrid 5 Sedan 2012',
    'BMW M3 Coupe 2012',
    'BMW M5 Sedan 2010',
    'BMW M6 Convertible 2010',
    'BMW X3 SUV 2012',
    'BMW X5 SUV 2007',
    'BMW X6 SUV 2012',
    'BMW Z4 Convertible 2012',
    'Bentley Arnage Sedan 2009',
    'Bentley Continental Flying Spur Sedan 2007',
    'Bentley Continental GT Coupe 2007',
    'Bentley Continental GT Coupe 2012',
    'Bentley Continental Supersports Conv. Convertible 2012',
    'Bentley Mulsanne Sedan 2011',
    'Bugatti Veyron 16.4 Convertible 2009',
    'Bugatti Veyron 16.4 Coupe 2009',
    'Buick Enclave SUV 2012',
    'Buick Rainier SUV 2007',
    'Buick Regal GS 2012',
    'Buick Verano Sedan 2012',
    'Cadillac CTS-V Sedan 2012',
    'Cadillac Escalade EXT Crew Cab 2007',
    'Cadillac SRX SUV 2012',
    'Chevrolet Avalanche Crew Cab 2012',
    'Chevrolet Camaro Convertible 2012',
    'Chevrolet Cobalt SS 2010',
    'Chevrolet Corvette Convertible 2012',
    'Chevrolet Corvette Ron Fellows Edition Z06 2007',
    'Chevrolet Corvette ZR1 2012',
    'Chevrolet Express Cargo Van 2007',
    'Chevrolet Express Van 2007',
    'Chevrolet HHR SS 2010',
    'Chevrolet Impala Sedan 2007',
    'Chevrolet Malibu Hybrid Sedan 2010',
    'Chevrolet Malibu Sedan 2007',
    'Chevrolet Monte Carlo Coupe 2007',
    'Chevrolet Silverado 1500 Classic Extended Cab 2007',
    'Chevrolet Silverado 1500 Extended Cab 2012',
    'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',
    'Chevrolet Silverado 1500 Regular Cab 2012',
    'Chevrolet Sonic Sedan 2012',
    'Chevrolet Tahoe Hybrid SUV 2012',
    'Chevrolet TrailBlazer SS 2009',
    'Chevrolet Traverse SUV 2012',
    'Chrysler 300 SRT-8 2010',
    'Chrysler Aspen SUV 2009',
    'Chrysler Crossfire Convertible 2008',
    'Chrysler PT Cruiser Convertible 2008',
    'Chrysler Sebring Convertible 2010',
    'Chrysler Town and Country Minivan 2012',
    'Daewoo Nubira Wagon 2002',
    'Dodge Caliber Wagon 2007',
    'Dodge Caliber Wagon 2012',
    'Dodge Caravan Minivan 1997',
    'Dodge Challenger SRT8 2011',
    'Dodge Charger Sedan 2012',
    'Dodge Charger SRT-8 2009',
    'Dodge Dakota Club Cab 2007',
    'Dodge Dakota Crew Cab 2010',
    'Dodge Durango SUV 2007',
    'Dodge Durango SUV 2012',
    'Dodge Journey SUV 2012',
    'Dodge Magnum Wagon 2008',
    'Dodge Ram Pickup 3500 Crew Cab 2010',
    'Dodge Ram Pickup 3500 Quad Cab 2009',
    'Dodge Sprinter Cargo Van 2009',
    'Eagle Talon Hatchback 1998',
    'FIAT 500 Abarth 2012',
    'FIAT 500 Convertible 2012',
    'Ferrari 458 Italia Convertible 2012',
    'Ferrari 458 Italia Coupe 2012',
    'Ferrari California Convertible 2012',
    'Ferrari FF Coupe 2012',
    'Fisker Karma Sedan 2012',
    'Ford E-Series Wagon Van 2012',
    'Ford Edge SUV 2012',
    'Ford Escape SUV 2012',
    'Ford Expedition EL SUV 2009',
    'Ford F-150 Regular Cab 2007',
    'Ford F-150 Regular Cab 2012',
    'Ford F-450 Super Duty Crew Cab 2012',
    'Ford Fiesta Sedan 2012',
    'Ford Focus Sedan 2007',
    'Ford Freestar Minivan 2007',
    'Ford GT Coupe 2006',
    'Ford Mustang Convertible 2007',
    'Ford Ranger SuperCab 2011',
    'GMC Acadia SUV 2012',
    'GMC Canyon Extended Cab 2012',
    'GMC Savana Van 2012',
    'GMC Terrain SUV 2012',
    'GMC Yukon Hybrid SUV 2012',
    'Geo Metro Convertible 1993',
    'HUMMER H2 SUT Crew Cab 2009',
    'HUMMER H3T Crew Cab 2010',
    'Honda Accord Coupe 2012',
    'Honda Accord Sedan 2012',
    'Honda Civic Coupe 2012',
    'Honda Civic Sedan 2012',
    'Honda Odyssey Minivan 2012',
    'Honda Ridgeline Crew Cab 2012',
    'Hyundai Accent Sedan 2012',
    'Hyundai Azera Sedan 2012',
    'Hyundai Elantra Sedan 2007',
    'Hyundai Elantra Touring Hatchback 2012',
    'Hyundai Genesis Sedan 2012',
    'Hyundai Santa Fe SUV 2012',
    'Hyundai Sonata Hybrid Sedan 2012',
    'Hyundai Sonata Sedan 2012',
    'Hyundai Tucson SUV 2012',
    'Hyundai Veloster Hatchback 2012',
    'Hyundai Veracruz SUV 2012',
    'Infiniti G Coupe IPL 2012',
    'Infiniti QX56 SUV 2011',
    'Isuzu Ascender SUV 2008',
    'Jaguar XK XKR 2012',
    'Jeep Compass SUV 2012',
    'Jeep Grand Cherokee SUV 2012',
    'Jeep Liberty SUV 2012',
    'Jeep Patriot SUV 2012',
    'Jeep Wrangler SUV 2012',
    'Lamborghini Aventador Coupe 2012',
    'Lamborghini Diablo Coupe 2001',
    'Lamborghini Gallardo LP 570-4 Superleggera 2012',
    'Lamborghini Reventon Coupe 2008',
    'Land Rover LR2 SUV 2012',
    'Land Rover Range Rover SUV 2012',
    'Lincoln Town Car Sedan 2011',
    'MINI Cooper Roadster Convertible 2012',
    'Maybach Landaulet Convertible 2012',
    'Mazda Tribute SUV 2011',
    'McLaren MP4-12C Coupe 2012',
    'Mercedes-Benz 300-Class Convertible 1993',
    'Mercedes-Benz C-Class Sedan 2012',
    'Mercedes-Benz E-Class Sedan 2012',
    'Mercedes-Benz S-Class Sedan 2012',
    'Mercedes-Benz SL-Class Coupe 2009',
    'Mercedes-Benz Sprinter Van 2012',
    'Mitsubishi Lancer Sedan 2012',
    'Nissan 240SX Coupe 1998',
    'Nissan Juke Hatchback 2012',
    'Nissan Leaf Hatchback 2012',
    'Nissan NV Passenger Van 2012',
    'Nissan Versa Sedan 2012',
    'Plymouth Neon Coupe 1999',
    'Porsche Panamera Sedan 2012',
    'Ram C/V Cargo Van Minivan 2012',
    'Rolls-Royce Ghost Sedan 2012',
    'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
    'Rolls-Royce Phantom Sedan 2012',
    'Scion xD Hatchback 2012',
    'Spyker C8 Convertible 2009',
    'Spyker C8 Coupe 2009',
    'Suzuki Aerio Sedan 2007',
    'Suzuki Kizashi Sedan 2012',
    'Suzuki SX4 Hatchback 2012',
    'Suzuki SX4 Sedan 2012',
    'Tesla Model S Sedan 2012',
    'Toyota 4Runner SUV 2012',
    'Toyota Camry Sedan 2012',
    'Toyota Corolla Sedan 2012',
    'Toyota Sequoia SUV 2012',
    'Toyota Tacoma Regular Cab 2012',
    'Toyota Tundra CrewMax Cab 2012',
    'Toyota Yaris Hatchback 2012',
    'Volkswagen Beetle Hatchback 2012',
    'Volkswagen Golf Hatchback 1991',
    'Volkswagen Golf Hatchback 2012',
    'Volkswagen GTI Hatchback 2012',
    'Volkswagen Jetta Sedan 2012',
    'Volkswagen Passat Sedan 2012',
    'Volkswagen Routan Minivan 2012',
    'Volkswagen Tiguan SUV 2012',
    'Volkswagen Touareg SUV 2012',
    'Volvo 240 Sedan 1993',
    'Volvo C30 Hatchback 2012',
    'Volvo XC70 Wagon 2007',
    'Volvo XC90 SUV 2007'
]



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
def predict(image, model):  # Make sure to pass model to the predict function
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
model = load_model()  # Load the model once at the start

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
            
            # Check if the response is an image by checking the content type
            if 'image' in response.headers['Content-Type']:
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                st.error("The provided URL does not seem to be an image.")
                img = None
        except Exception as e:
            st.error(f"Unable to load image. Error: {e}")
            img = None

# Show image and prediction
if img:
    st.image(img, caption="Input Image", use_container_width=True)
    if st.button("Predict"):
        pred, acc = predict(img, model)  # Pass the model to the predict function
        st.success(f"Prediction: **{car_names[pred]}**")
        st.info(f"Confidence: **{acc:.2f}%**")
