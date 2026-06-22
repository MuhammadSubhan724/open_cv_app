import streamlit as st
import cv2
import numpy as np
from PIL import Image
from google import genai
import os

# Page Configuration
st.set_page_config(page_title="Smart Urdu Handwriting OCR", page_icon="📝", layout="wide")

st.title("Smart Urdu & Mixed Handwriting OCR Fixer")
st.write("Upload handwritten images (Urdu + English + Numbers) to extract text with high precision.")

# Sidebar: API Key Input
st.sidebar.header("🔑 Configuration")
api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

# OpenCV Preprocessing Function
def enhance_handwriting_image(uploaded_file):
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 1. Grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Adaptive Thresholding (Shadows remove karne aur text ko intense black karne ke liye)
    # Yeh step haath se likhay lafzon ke nuqtay (dots) ko saaf dikhata hai
    enhanced = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 11
    )
    
    # Convert back to PIL Image for Streamlit and Gemini displaying
    return Image.fromarray(enhanced), Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Main UI Layout
uploaded_file = st.file_uploader("Choose a handwritten image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Processing Image Data")
        # Process the image through OpenCV
        enhanced_img, original_img = enhance_handwriting_image(uploaded_file)
        
        # Display both for comparison
        tab1, tab2 = st.tabs(["Enhanced (Cleaned for AI)", "Original Upload"])
        with tab1:
            st.image(enhanced_img, use_container_width=True, caption="Binarized & Cleaned Handwriting")
        with tab2:
            st.image(original_img, use_container_width=True, caption="Original Image")

    with col2:
        st.subheader(" Text Extraction Engine")
        
        if st.button(" Extract & Refine Text"):
            if not api_key:
                st.error("Bhai, pehle side panel mein Gemini API key toh enter karo! 🔑")
            else:
                with st.spinner("Processing visual strokes and correcting text..."):
                    try:
                        # Initialize the Google GenAI Client dynamically with user's key
                        client = genai.Client(api_key=api_key)
                        
                        # High-level specialized prompt designed to stop AI from guessing/hallucinating words
                        ocr_prompt = """
                        You are an expert, highly precise character-by-character OCR engine fine-tuned for handwritten manuscripts.
                        The uploaded image contains handwritten text in mixed Urdu (Nastaliq script), English, and numerical digits.
                        
                        Analyze the visual strokes and diacritics (specifically Urdu nuqtay/dots) with extreme care.
                        
                        Strict Rules to follow:
                        1. Extract the text EXACTLY as written in the image. 
                        2. DO NOT assume or autocomplete words based on context. For example, if a word is clearly written as 'اختر' (Akhtar), pay close attention to the dots so you do not misinterpret it as 'اخبر' or 'اصغر'.
                        3. Keep the layout and mixed language context intact (Urdu text, English words, and numbers should maintain their original relative sequence).
                        4. Do not fix grammatical errors or spelling mistakes made by the writer. Output the raw true text.
                        5. If a specific word is completely unreadable, output '[Unreadable]' for that part.
                        
                        Directly output the extracted text. No conversational introductions or notes.
                        """
                        
                        # Fire the multimodal request using the enhanced black & white image
                        response = client.models.generate_content(
                            model="gemini-3-flash-preview",
                            contents=[enhanced_img, ocr_prompt]
                        )
                        
                        # Output Result
                        st.success("Extraction Completed!")
                        st.text_area("Extracted Text Output:", value=response.text, height=300)
                        
                        # Download Button for the student/brother
                        st.download_button(
                            label=" Download Text File",
                            data=response.text,
                            file_name="extracted_handwriting.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Error encountered: {str(e)}")