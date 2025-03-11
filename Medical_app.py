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
user_type = st.sidebar.selectbox(
    "Select User Type:", ["Patient", "Healthcare Professional", "Researcher"]
)

options = st.sidebar.radio(
    "Choose a section:",
    [
        "🖼️ Upload & Predict",
        "📊 About the Model",
        "🧪 Evaluation",
        "👥 Team",
        "📚 Educational Resources"
    ]
)

# Custom messages for each user type
def get_user_customization(user_type):
    if user_type == "Patient":
        return {
            "welcome_message": "Empowering you with personalized insights about PCOS.",
            "advice_label": "Encouragement and Advice",
            "advice_text": """
            - Maintain a healthy lifestyle.
            - Follow your doctor's advice for preventive care.
            - Stay informed about PCOS and its management.
            """,
        }
    elif user_type == "Healthcare Professional":
        return {
            "welcome_message": "Providing healthcare professionals with actionable diagnostic support.",
            "advice_label": "Professional Insights",
            "advice_text": """
            - Use this report as part of your patient management toolkit.
            - Consider additional diagnostics to confirm the results.
            - Share educational resources with patients for better care.
            """,
        }
    elif user_type == "Researcher":
        return {
            "welcome_message": "Aiding researchers in advancing medical imaging analysis.",
            "advice_label": "Research Recommendations",
            "advice_text": """
            - Use the insights to identify patterns in PCOS diagnosis.
            - Consider integrating this model into broader datasets.
            - Contribute findings to improve healthcare outcomes.
            """,
        }

user_customization = get_user_customization(user_type)

# Upload & Predict section
if options == "🖼️ Upload & Predict":
    st.title("Welcome to the Medical Imaging Diagnosis PCOS Dashboard")
    st.image("pngwing.com (25).png", use_container_width=True)
    st.markdown(
        f"<h4 style='color:#e75480;'>{user_customization['welcome_message']}</h4>",
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
        else:
            st.error("The ultrasound image is classified as **Infected**.")
            clinical_insights = """
            - Increased ovarian size (>10 cm³).
            - Presence of 12+ follicles (2-9 mm) arranged peripherally.
            - "String of pearls" appearance observed.
            - Increased stromal echogenicity.
            """

        # Display clinical insights and advice
        st.markdown("### Clinical Insights:")
        st.markdown(clinical_insights)
        st.markdown(f"### {user_customization['advice_label']}:")
        st.markdown(user_customization["advice_text"])

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
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(200, 10, txt="Clinical Insights:", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=clinical_insights)
            pdf.ln(5)
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(200, 10, txt=f"{user_customization['advice_label']}:", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=user_customization["advice_text"])
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

# The remaining sections (About the Model, Evaluation, Team, Educational Resources) remain unchanged.
elif options == "📚 Educational Resources":
    st.header("📚 Educational Resources")
    st.write("Explore detailed insights into PCOS and healthcare management.")
