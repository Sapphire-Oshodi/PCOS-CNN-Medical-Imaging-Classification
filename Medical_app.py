import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import gdown

# Function to load the trained model, download it if necessary
@st.cache_resource
def load_trained_model():
    model_path = "Model/Pcos_Scan_model.h5"
    drive_link = "https://drive.google.com/uc?id=1UTBOUNtIzhAtCRDzI5D7TnsGMHsY43D-"

    # Download the model if it does not exist
    if not os.path.exists(model_path):
        os.makedirs("Model", exist_ok=True)
        gdown.download(drive_link, model_path, quiet=False)

    return load_model(model_path)

model = load_trained_model()

# Sidebar with logo and navigation
st.sidebar.image("Cycleec.png", use_container_width=True)
options = st.sidebar.radio(
    "Choose a section:",
    [
        "🖼️ Upload & Predict",
        "📊 About the Model",
        "🧪 Evaluation",
        "👥 Team",
        "💡 For Life"
    ]
)

# Upload & Predict section
if options == "🖼️ Upload & Predict":
    st.title("Welcome to the Medical Imaging Diagnosis PCOS Dashboard")
    st.image("pngwing.com (25).png", use_container_width=True)
    st.write("This app provides insights into the medical imaging analysis.")
    st.write("Upload an ultrasound image to classify it as **Infected** or **Noninfected**.")
    
    threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    user_name = st.text_input("Enter your name:", value="Patient")
    uploaded_file = st.file_uploader("Upload an Ultrasound Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        img = img.resize((256, 256))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        yhat = model.predict(img_array)
        confidence = round(yhat[0][0] * 100, 1)  # Confidence rounded to 1 decimal

        # Classification
        result = "Noninfected" if yhat[0][0] >= threshold else "Infected"

        # Display results
        st.write(f"### **Result:** {result}")
        st.write(f"**Prediction Confidence:** {confidence}%")

        if result == "Noninfected":
            st.success("The ultrasound image is classified as **Noninfected**.")
            st.markdown("""
            **Clinical Insights**:
            - Normal ovarian size (<10 cm³).
            - Fewer than 12 follicles, evenly distributed.
            - Homogeneous ovarian stroma.
            - No cystic patterns detected.
            """)
        else:
            st.error("The ultrasound image is classified as **Infected**.")
            st.markdown("""
            **Clinical Insights**:
            - Increased ovarian size (>10 cm³).
            - Presence of 12+ follicles (2-9 mm) arranged peripherally.
            - "String of pearls" appearance observed.
            - Increased stromal echogenicity.
            - Potential thickened endometrium.
            """)

# About the Model section
elif options == "📊 About the Model":
    st.header("📊 About the Model")
    st.write("""
    This model is a **Convolutional Neural Network (CNN)** trained to classify ultrasound images as:
    - **Infected**: Presence of polycystic ovaries.
    - **Noninfected**: Normal ovaries without signs of PCOS.
    """)

    st.markdown("#### Model Performance During Training")

    # Display graphs saved from Jupyter Notebook
    st.image("1.jpeg", caption="Training and Validation Accuracy", use_container_width=True)
    st.image("2.jpeg", caption="Training and Validation Loss", use_container_width=True)

# Evaluation section
elif options == "🧪 Evaluation":
    st.header("🧪 Model Evaluation")
    st.write("Evaluate the model's performance on the test dataset.")

    # Display graphs or other metrics if needed
    st.markdown("#### Confusion Matrix")
    st.image("3.jpeg", caption="Confusion Matrix", use_container_width=True)

# Team section
elif options == "👥 Team":
    st.header("👥 Meet the Team")
    st.write("""
    This project was developed by:
    - **Sapphire Oshodi**
    - **Samuel Odukoya**
    - **Habeebat Jinadu**
    - **Hamzat Akolade**
    - **Tiletile Toheebat**
    - **Balogun Memunat**
    - **Adeleke Joshua**
    - **Adewale Abidemi**
    
    ### Acknowledgements
    We thank our mentors, instructors, and the dataset contributors for their valuable guidance and support.
    """)

# For Life section
elif options == "💡 For Life":
    st.header("For Life: Stay Inspired")
    st.write("""
    Life is a journey filled with challenges, but every challenge is an opportunity to grow stronger.
    
    ### You Are Not Alone
    - Support and care are always within reach.
    - Surround yourself with positivity and hope.

    ### Inspirational Quote
    > *"PCOS is a part of your story, but it is not the whole story. You are so much more than a diagnosis."*

    ### 📚 Resources
    - [PCOS Awareness Association](https://www.pcosaa.org/)
    - [Support Groups](https://www.resolve.org/support/)
    - [Mindfulness Exercises](https://www.mindful.org/)
    """)
    st.image(
        "105 Uplifting Affirmations for a Healthy Body and Beautiful Mind.jpeg",
        caption="Keep Moving Forward",
        use_container_width=True,
    )
