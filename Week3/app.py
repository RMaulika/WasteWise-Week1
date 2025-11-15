import streamlit as st
import numpy as np
import os
from tensorflow.keras.preprocessing import image  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
import json


st.set_page_config(page_title="WasteWise Demo", layout="centered", initial_sidebar_state="auto")


st.sidebar.title("WasteWise — Info")
st.sidebar.markdown("""
**Project:** WasteWise (Sustainability — Organic vs Recyclable)  
**Model:** MobileNetV2 (fine-tuned)  
**Week:** 1–3 trained, Week 4: final demo & docs  
""")
st.sidebar.markdown("**How to use**\n1. Upload an image (jpg/png)\n2. See prediction + confidence\n3. Try different samples from `Week3/sample_images/`")
if st.sidebar.button("Open sample folder"):
    st.sidebar.info("Use VS Code Explorer / File Explorer to open `Week3/sample_images/` and drag files here.")


st.title("♻ WasteWise – Smart Waste Classification Demo")
st.write("""
Upload an image of waste and the model will classify it as:
- **Organic (O)**  
- **Recyclable (R)**

This demo uses the fine-tuned MobileNetV2 model trained during **Week 1–3**.
""")

# small helper: reset file uploader (try another)
if "reset_uploader" not in st.session_state:
    st.session_state.reset_uploader = False

def reset_uploader():
    st.session_state.reset_uploader = True


uploaded = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], key="uploader")
if uploaded and not st.session_state.reset_uploader:
    # show uploaded image
    st.image(uploaded, caption='Uploaded Image', use_column_width=True)

    # save temp
    temp_path = "Week3/temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # model path (local; do NOT push .h5)
    model_path = "Week2/outputs/best_model_week2.h5"
    if not os.path.exists(model_path):
        st.error("Model not found locally: " + model_path + "\nPlace your trained .h5 at this path to run inference.")
    else:
        # load and run
        model = load_model(model_path)
        img = image.load_img(temp_path, target_size=(224, 224))
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        preds = model.predict(arr)
        idx = int(preds.argmax())
        confidence = float(preds.max())

        # load class mapping if available
        mapping_path = "Week2/outputs/class_indices.json"
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                class_map = json.load(f)
            inv_map = {v: k for k, v in class_map.items()}
            label = inv_map.get(idx, str(idx))
        else:
            label = str(idx)

        # color-coded result box
        if label.lower().startswith("o"):  # organic
            st.success(f"Prediction: **{label}** — Organic")
            st.info(f"Confidence: **{confidence:.3f}**")
            st.markdown("**Note:** Organic waste is biodegradable. Consider composting.")
        else:  # recyclable
            st.success(f"Prediction: **{label}** — Recyclable")
            st.info(f"Confidence: **{confidence:.3f}**")
            st.markdown("**Note:** Recyclable waste should be cleaned and placed in recycling bins.")

    # Try another
    if st.button("🔁 Try another image"):
        # reset uploader by toggling the session state key
        st.session_state.reset_uploader = True
        st.rerun()

# If reset flag was set, clear uploader
if st.session_state.reset_uploader:
    st.session_state.reset_uploader = False
    st.rerun()

# Footer
st.markdown("---")
st.markdown("Model and demo created as part of the project. Keep `.h5` model files local; only push small artifacts to GitHub.")
