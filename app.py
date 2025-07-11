import streamlit as st
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import os

# Load the TFLite model
@st.cache_resource
def load_model(path="model/leukocyte_model.tflite"):
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

# Make prediction
def predict_image(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.argmax(output_data)
    return prediction, output_data[0]

# Class names
CLASS_NAMES = ['Neutrophil', 'Eosinophil', 'Monocyte', 'Lymphocyte']

# UI
st.set_page_config(page_title="Leukocyte Classifier", layout="centered")
st.title("🔬 Leukocyte Classifier")
st.markdown("Upload a white blood cell image to identify the cell type.")

uploaded_file = st.file_uploader("Upload WBC Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        interpreter = load_model()
        pred_idx, scores = predict_image(img, interpreter)

    st.success(f"🧬 Prediction: **{CLASS_NAMES[pred_idx]}**")

    st.subheader("📊 Confidence Scores:")
    for i, cls in enumerate(CLASS_NAMES):
        st.write(f"{cls}: {scores[i]*100:.2f}%")
