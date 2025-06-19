import streamlit as st
import os
from PIL import Image
import tempfile
import json
from qwen_ocr import QwenOCR
import plotly.express as px
import pandas as pd
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Invoice OCR Processor",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-section {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None

def initialize_processor():
    """Initialize the OCR processor"""
    if not st.session_state.processor:
        try:
            st.session_state.processor = QwenOCR()
            return True
        except Exception as e:
            st.error(f"Error initializing OCR processor: {str(e)}")
            return False
    return True

def process_document(file):
    """Process the uploaded document"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        # Process the document
        text, confidence = st.session_state.processor.extract_text(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return text, confidence
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None, None

def display_results(results_json):
    """Display the extracted results in a nice format"""
    try:
        results = json.loads(results_json)
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Supplier", results.get("supplier_name", "N/A"))
        with col2:
            st.metric("Invoice Number", results.get("invoice_number", "N/A"))
        with col3:
            st.metric("Invoice Date", results.get("invoice_date", "N/A"))
        
        # Display line items in a table
        if results.get("line_items"):
            df = pd.DataFrame(results["line_items"])
            st.subheader("Line Items")
            st.dataframe(df, use_container_width=True)
            
            # Create a bar chart for line items
            if not df.empty:
                fig = px.bar(df, 
                           x='description', 
                           y='total_price',
                           title='Line Items by Total Price',
                           labels={'description': 'Item', 'total_price': 'Total Price'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Display total amount
        st.subheader("Total Amount")
        st.metric("Amount", f"${results.get('total_amount', 0.0):,.2f}")
        
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")

def main():
    st.title("📄 Invoice OCR Processor")
    st.markdown("Upload an invoice (PDF or image) to extract information using AI-powered OCR.")
    
    # Initialize processor
    if not initialize_processor():
        st.stop()
    
    # File upload section
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Upload a PDF or image file (PNG, JPG, JPEG)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process button
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document..."):
                # Process the document
                results, confidence = process_document(uploaded_file)
                
                if results:
                    st.session_state.results = results
                    st.session_state.confidence = confidence
                    
                    # Display results
                    st.markdown('<div class="result-section">', unsafe_allow_html=True)
                    st.subheader("Extracted Information")
                    display_results(results)
                    
                    # Display confidence score
                    st.markdown("---")
                    st.metric("Confidence Score", f"{confidence:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display instructions
    with st.expander("How to use"):
        st.markdown("""
        1. Upload a PDF or image file containing an invoice
        2. Click the 'Process Document' button
        3. View the extracted information including:
           - Supplier details
           - Invoice number and date
           - Line items with quantities and prices
           - Total amount
        4. The confidence score indicates how reliable the extraction is
        """)

if __name__ == "__main__":
    main() 