# app.py
import streamlit as st
from PIL import Image
import io
from model import generate_bleu_score_and_report  # Import function from model.py

# Function to display the Streamlit app
def display_app():
    st.set_page_config(page_title="IU X-Ray Analysis", layout="wide")

    # Header section
    st.markdown("""
        <div style="text-align: center; background-color: #023e8a; padding: 15px; border-radius: 10px;">
            <h1 style="color: white; font-family: Arial, sans-serif;">IU X-Ray Analysis</h1>
            <p style="color: #f8f9fa;">Upload an X-ray image to analyze BLEU score and generate a detailed report.</p>
        </div>
    """, unsafe_allow_html=True)

    # Upload image section
    uploaded_file = st.file_uploader("Upload an X-ray image (JPG/PNG)", type=["jpg", "png"])

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)
        
        # Placeholder for results
        with st.spinner("Processing image..."):
            # Set reference reports for BLEU calculation (dummy reference used here)
            reference_reports = [["The heart size and pulmonary vascularity appear within normal limits."]]
            
            # Fetch the result from the model
            bleu_score, report = generate_bleu_score_and_report(uploaded_file, reference_reports)
        
        # Display BLEU score
        st.markdown(f"""
            <div style="text-align: center; margin-top: 20px; padding: 10px; background-color: #48cae4; border-radius: 10px;">
                <h3 style="color: #03045e;">BLEU Score</h3>
                <p style="font-size: 24px; color: white;">{bleu_score}</p>
            </div>
        """, unsafe_allow_html=True)

        # Display the report
        st.markdown("<h3 style='text-align: center;'>Generated Report</h3>", unsafe_allow_html=True)
        st.text(report)

        # Allow downloading of the report
        report_bytes = io.BytesIO(report.encode())
        st.download_button(
            label="Download Report",
            data=report_bytes,
            file_name="xray_report.txt",
            mime="text/plain"
        )
    else:
        st.markdown("""
            <div style="text-align: center; padding: 20px; background-color: #ade8f4; border-radius: 10px;">
                <h4 style="color: #023e8a;">Please upload an X-ray image to proceed.</h4>
            </div>
        """, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    display_app()
