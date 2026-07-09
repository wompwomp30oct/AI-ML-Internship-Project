import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Cancer Detection Demo", page_icon="🩺", layout="wide")

st.title("🩺 Cancer Detection Demo")
st.markdown("### Internship Project - Based on MONAI")
st.write("This is a simplified live demo of medical image analysis for cancer/tumor detection.")

st.info("Note: This is a demonstration version. Full MONAI models require GPU and more resources.")

uploaded_file = st.file_uploader("Upload a medical image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Analyzing image..."):
            # Placeholder for actual model inference
            # In a real version we would load a MONAI model here
            st.success("Analysis complete (Demo mode)")
            st.warning("This is a demo UI. Full model inference will be added next.")
            
            # Fake visualization for now
            st.subheader("Segmentation Result (Demo)")
            st.image(image, caption="Detected region highlighted (placeholder)", use_column_width=True)

st.markdown("---")
st.markdown("**Original Project**: [MONAI](https://github.com/Project-MONAI/MONAI)")
st.markdown("**Repository**: [AI-ML-Internship-Project](https://github.com/wompwomp30oct/AI-ML-Internship-Project)")
