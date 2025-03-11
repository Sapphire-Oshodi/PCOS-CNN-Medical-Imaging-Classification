import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import gdown
from fpdf import FPDF

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

# User Type Selection
selected_user_type = st.sidebar.selectbox(
    "Select User Type:",
    ["Patient", "Healthcare Professional", "Researcher"]
)

# Section Navigation
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

# Upload & Predict Section
if options == "🖼️ Upload & Predict":
    st.title("Welcome to the Medical Imaging Diagnosis PCOS Dashboard")
    st.image("pngwing.com (25).png", use_container_width=True)
    st.markdown(
        "<h4 style='color:#e75480;'>This app provides insights into the medical imaging analysis.</h4>",
        unsafe_allow_html=True
    )
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

        # Generate dynamic content based on user type and classification result
        if result == "Noninfected":
            st.success("The ultrasound image is classified as **Noninfected**.")
            clinical_insights = "..."  # Add dynamic content for Noninfected
            advice = "..."  # Add advice for Noninfected
        else:
            st.error("The ultrasound image is classified as **Infected**.")
            clinical_insights = "..."  # Add dynamic content for Infected
            advice = "..."  # Add advice for Infected

        # Display clinical insights and advice
        st.markdown(clinical_insights)
        st.markdown(advice)

        # Generate and Download PDF
        if st.button("Download Report"):
            uploaded_image_path = "uploaded_image.jpg"
            img.save(uploaded_image_path)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="PCOS Medical Diagnosis Report", ln=True, align="C")
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Patient Name: {user_name}", ln=True)
            pdf.cell(200, 10, txt=f"Classification Result: {result}", ln=True)
            pdf.cell(200, 10, txt=f"Prediction Confidence: {confidence}%", ln=True)
            pdf.ln(10)
            pdf.multi_cell(0, 10, txt=clinical_insights.replace("### ", "").replace("**", ""))
            pdf.ln(5)
            pdf.multi_cell(0, 10, txt=advice.replace("### ", "").replace("**", ""))
            pdf.ln(10)
            pdf.cell(200, 10, txt="Uploaded Ultrasound Image:", ln=True)
            pdf.image(uploaded_image_path, x=10, y=None, w=100)

            pdf_content = pdf.output(dest="S").encode("latin1")
            os.remove(uploaded_image_path)

            st.download_button(
                label="Download Report as PDF",
                data=pdf_content,
                file_name=f"{user_name.replace(' ', '_')}_PCOS_Report.pdf",
                mime="application/pdf"
            )

# About the Model Section
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

# Evaluation Section
elif options == "🧪 Evaluation":
    st.header("🧪 Model Evaluation")
    st.write("Evaluate the model's performance on the test dataset.")
    st.markdown("#### Confusion Matrix")
    st.image("3.jpeg", caption="Confusion Matrix", use_container_width=True)

# Team Section
elif options == "👥 Team":
    st.header("👥 Meet the Team")
    st.write("""
    This project was developed by:
    - **Sapphire Oshodi**
    - **Samuel Odukoya**
    - **Habeebat Jinadu**
    - **Hamzat Akolade**
    - **Tiletile Toheebat**
    
    ### Acknowledgements
    We thank our mentors, instructors, and the dataset contributors for their valuable guidance and support.
    """)

# For Life Section
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
