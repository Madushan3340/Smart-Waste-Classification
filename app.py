import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Page Configuration
st.set_page_config(page_title="Smart Waste Classification", layout="centered")

# Load Class Names
@st.cache_data
def load_class_names():
    with open('class_names.json', 'r') as f:
        return json.load(f)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('waste_classifier_model.keras')

try:
    class_names = load_class_names()
    model = load_model()
except Exception as e:
    st.error("Error loading model or class names. Please ensure the files are in the same folder as app.py.")
    st.stop()

# UI Layout
st.title("♻️ Smart Waste Classification System")
st.write("Upload an image of waste to classify it into one of the predefined categories.")

# Requirement 1: Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Requirement 2: Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Analyzing image...")
    
    # Preprocess image to match MobileNetV2 training format
    img = image.resize((224, 224)) 
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Run the prediction
    predictions = model.predict(img_array)[0]
    predicted_class_index = np.argmax(predictions)
    
    # Check if the keys in json are strings or ints
    str_index = str(predicted_class_index)
    if str_index in class_names:
        predicted_class_name = class_names[str_index]
    else:
        # Fallback in case indices are somehow different
        predicted_class_name = list(class_names.values())[predicted_class_index]
        
    confidence = np.max(predictions) * 100
    
    # Requirement 3 & 4: Predict category and display confidence
    st.success(f"### Predicted Category: **{predicted_class_name.upper()}**")
    st.info(f"**Prediction Confidence:** {confidence:.2f}%")
    
    st.divider()
    
    # Requirement 5: Display probabilities for all classes
    st.subheader("Probabilities for all categories:")
    
    # Map predictions to class names safely
    probs_dict = {}
    for i in range(len(predictions)):
        idx_str = str(i)
        cat_name = class_names[idx_str] if idx_str in class_names else list(class_names.values())[i]
        probs_dict[cat_name.capitalize()] = float(predictions[i]) * 100
    
    # Sort by probability descending
    sorted_probs = dict(sorted(probs_dict.items(), key=lambda item: item[1], reverse=True))
    
    for category, prob in sorted_probs.items():
        st.write(f"- **{category}**: {prob:.2f}%")
        st.progress(int(prob))