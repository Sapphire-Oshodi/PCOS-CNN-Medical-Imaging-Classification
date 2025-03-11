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
    st.markdown(
        "<h4 style='color:#e75480;'>This app provides insights into the medical imaging analysis.</h4>",
        unsafe_allow_html=True,
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

        # Display results
        st.write(f"### **Result:** {result}")
        st.write(f"**Prediction Confidence:** {confidence}%")

        if result == "Noninfected":
            st.success("The ultrasound image is classified as **Noninfected**.")
            clinical_insights = """
            - Normal ovarian size (<10 cm³).
            - Fewer than 12 follicles, evenly distributed.
            - Homogeneous ovarian stroma.
            - No cystic patterns detected.
            """
            advice = """
            - **Great News**: Your ovaries show no signs of PCOS.
            - **Maintain Health**: Keep up with a balanced diet and regular exercise.
            - **Regular Check-ups**: Continue routine gynecological examinations for ongoing health monitoring.
            - **Awareness**: Stay informed about women's health issues for preventive care.
            """
        else:
            st.error("The ultrasound image is classified as **Infected**.")
            clinical_insights = """
            - Increased ovarian size (>10 cm³).
            - Presence of 12+ follicles (2-9 mm) arranged peripherally.
            - "String of pearls" appearance observed.
            - Increased stromal echogenicity.
            - Potential thickened endometrium.
            """
            advice = """
            - **You Are Not Alone**: Many individuals successfully manage PCOS with the right support and care.
            - **Consultation**: Consult a gynecologist for further evaluation and management.
            - **Treatment Options**: Discuss potential treatments such as lifestyle changes, medications, or hormonal therapy.
            - **Self-Care**: Stay proactive in monitoring symptoms and following up with healthcare professionals.
            - **Support Groups**: Reach out to support groups or trusted healthcare providers for guidance.
            """

        # Display insights and advice
        st.markdown("### Clinical Insights")
        st.markdown(clinical_insights)
        st.markdown("### Encouragement and Advice")
        st.markdown(advice)

        # Generate and Download PDF
        if st.button("Download Report"):
            # Save the uploaded image temporarily
            uploaded_image_path = "uploaded_image.jpg"
            img.save(uploaded_image_path)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            # Title
            pdf.cell(200, 10, txt="PCOS Medical Diagnosis Report", ln=True, align="C")
            pdf.ln(10)

            # Patient Details
            pdf.cell(200, 10, txt=f"Patient Name: {user_name}", ln=True)
            pdf.cell(200, 10, txt=f"Classification Result: {result}", ln=True)
            pdf.cell(200, 10, txt=f"Prediction Confidence: {confidence}%", ln=True)
            pdf.ln(10)

            # Clinical Insights
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(200, 10, txt="Clinical Insights:", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=clinical_insights)
            pdf.ln(5)

            # Encouragement and Advice
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(200, 10, txt="Encouragement and Advice:", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=advice)
            pdf.ln(10)

            # Uploaded Image
            pdf.cell(200, 10, txt="Uploaded Ultrasound Image:", ln=True)
            pdf.image(uploaded_image_path, x=10, y=None, w=100)

            # Generate PDF as bytes
            pdf_content = pdf.output(dest="S").encode("latin1")

            # Clean up temporary file
            os.remove(uploaded_image_path)

            # Streamlit download button
            st.download_button(
                label="Download Report as PDF",
                data=pdf_content,
                file_name=f"{user_name.replace(' ', '_')}_PCOS_Report.pdf",
                mime="application/pdf"
            )
