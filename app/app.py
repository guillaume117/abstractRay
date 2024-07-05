import streamlit as st
import requests
import json

st.title("Model Evaluator")

uploaded_network = st.file_uploader("Upload your network model (.onnx or .pt/.pth)", type=["onnx", "pt", "pth"])
uploaded_image = st.file_uploader("Upload your input image", type=["jpg", "jpeg", "png"])

num_worker = st.number_input("Number of workers", min_value=1, value=1)
back_end = st.selectbox("Backend", ["cpu", "cuda"])
num_symbol = st.text_input("Number of symbols", value="full")
noise = st.number_input("Noise level", value=0.0)
RAM = st.number_input("Available RAM per worker", value=1.0)

if st.button("Evaluate Model"):
    if uploaded_network is not None and uploaded_image is not None:
        files = {
            'network': uploaded_network,
            'input_image': uploaded_image
        }
        data = {
            'num_worker': num_worker,
            'back_end': back_end,
            'num_symbol': num_symbol,
            'noise': noise,
            'RAM': RAM
        }
        response = requests.post("http://localhost:8000/evaluate_model/", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            st.write("Argmax indices:", result["argmax"])
            st.write("True values:", result["true"])
            st.write("Center values:", result["center"])
            st.write("Min values:", result["min"])
            st.write("Max values:", result["max"])
            st.write("Difference (center - true):", result["diff_center_true"])
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.error("Please upload both the network model and the input image.")
