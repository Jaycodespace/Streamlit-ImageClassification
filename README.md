# 🚗 Car Model Classifier

This is a web application built with **Streamlit** that classifies car models from images using a **pretrained ResNet50** model. The model is fine-tuned on the **Stanford Cars Dataset** from Hugging Face, which includes 196 different car models.

Whether you're a car enthusiast or just curious about what model you're looking at, this app makes it easy to identify cars from photos.

## 🔍 Features

- Upload a car image or enter an image URL
- Predicts car model with confidence percentage
- Shows a green confidence bar for easy visual interpretation
- Detects invalid images and reports "Not a car" if prediction confidence is low
- Responsive interface with background image support

## 🧠 Model Details

- Backbone: ResNet50
- Dataset: [Stanford Cars (Hugging Face)](https://huggingface.co/datasets/naufalso/stanford_cars)
- Number of classes: 196
- Prediction confidence threshold: < 20% returns "Not a car"

