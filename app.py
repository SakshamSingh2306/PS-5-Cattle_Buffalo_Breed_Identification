# app.py
import streamlit as st
import pandas as pd
import random
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Cattle & Buffalo Breed Identifier",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR MODERN UI/UX ---
st.markdown(
    """
    <style>
    /* Import a modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background with a subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #ffffff;
    }

    /* Glass-morphism card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 2rem;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    }

    /* Main title styling */
    .main-title {
        text-align: center;
        font-weight: 700;
        font-size: 3.2rem;
        background: linear-gradient(135deg, #f5af19, #f12711);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .sub-title {
        text-align: center;
        color: #a0aec0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Result card - large, prominent */
    .result-card {
        background: rgba(0, 0, 0, 0.4);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border-left: 6px solid #f5af19;
        margin-top: 1rem;
    }

    .result-breed {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.5rem 0;
    }

    .result-badge {
        background: #f5af19;
        color: #1a1a2e;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Style sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background: rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #f5af19, #f12711) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px 0 rgba(241, 39, 17, 0.3);
        transition: all 0.3s ease !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(241, 39, 17, 0.4) !important;
    }

    /* Make file uploader look modern */
    .stFileUploader > div {
        border: 2px dashed rgba(255, 255, 255, 0.3) !important;
        border-radius: 20px !important;
        background: rgba(255, 255, 255, 0.03) !important;
        padding: 1.5rem !important;
    }
    .stFileUploader > div:hover {
        border-color: #f5af19 !important;
    }

    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #f5af19, #f12711) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- LOAD DATA (MOCK) ---
# In your actual app, load your CSV file
@st.cache_data
def load_data():
    # Replace this with: df = pd.read_csv('cattle_classification_data.csv')
    # For demonstration, we create a mock dataset
    data = {
        'Breed': ['Holstein Friesian', 'Jersey', 'Angus', 'Hereford', 'Simmental',
                  'Murrah Buffalo', 'Nili Ravi Buffalo', 'Surti Buffalo', 'Mehsana Buffalo'],
        'Category': ['Cattle'] * 5 + ['Buffalo'] * 4,
        'Origin': ['Europe', 'UK', 'Scotland', 'UK', 'Switzerland',
                   'India', 'India/Pakistan', 'India', 'India'],
        'Milk_Production_Ltrs': [25, 20, 15, 18, 22, 15, 18, 12, 16],
        'Temperament': ['Docile', 'Docile', 'Aggressive', 'Docile', 'Docile',
                        'Docile', 'Docile', 'Aggressive', 'Docile'],
    }
    return pd.DataFrame(data)

df = load_data()

# --- SESSION STATE ---
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'breed_result' not in st.session_state:
    st.session_state.breed_result = None

# --- UI LAYOUT ---
st.markdown('<p class="main-title">🐄 Breed Identifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an image to identify the breed of cattle or buffalo</p>', unsafe_allow_html=True)

# Use columns for a balanced layout
col_input, col_result = st.columns([1, 1.5], gap="large")

with col_input:
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        
        # For demonstration, we add a mock "Classify" button
        if st.button("🔍 Classify Breed", type="primary"):
            if uploaded_file is not None:
                # Simulate processing
                with st.spinner('Analyzing image...'):
                    time.sleep(2)  # Simulate model inference
                    # For demo, pick a random breed from the list
                    random_breed = df['Breed'].sample(1).iloc[0]
                    st.session_state.breed_result = random_breed
                    st.session_state.processed = True
                    st.success('✅ Classification complete!')
            else:
                st.warning("⚠️ Please upload an image first.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar content - Now in the main area as an expander for a cleaner look
    with st.expander("🔍 Advanced Filters & Info", expanded=False):
        st.markdown("### Filter Breeds")
        category_filter = st.selectbox("Category", options=['All', 'Cattle', 'Buffalo'])
        origin_filter = st.multiselect("Origin", options=df['Origin'].unique(), default=df['Origin'].unique())
        st.caption(f"Showing {len(df)} breeds in the database.")

with col_result:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Identification Result")
    
    if st.session_state.processed and st.session_state.breed_result:
        # Get breed details
        breed_info = df[df['Breed'] == st.session_state.breed_result].iloc[0]
        
        st.markdown(f"""
        <div class="result-card">
            <p style="color: #a0aec0; margin-bottom: 0;">Identified Breed</p>
            <p class="result-breed">{breed_info['Breed']}</p>
            <span class="result-badge">{breed_info['Category']}</span>
            <br><br>
            <table style="width: 100%; color: #e2e8f0; text-align: left; border-collapse: collapse;">
                <tr><td style="padding: 8px;"><strong>Origin</strong></td><td style="padding: 8px;">{breed_info['Origin']}</td></tr>
                <tr><td style="padding: 8px;"><strong>Milk Production</strong></td><td style="padding: 8px;">{breed_info['Milk_Production_Ltrs']} Ltrs/day</td></tr>
                <tr><td style="padding: 8px;"><strong>Temperament</strong></td><td style="padding: 8px;">{breed_info['Temperament']}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 Upload an image and click 'Classify Breed' to see the result here.")
    st.markdown('</div>', unsafe_allow_html=True)

# Additional feature: Show a sample of the dataset at the bottom
with st.expander("📚 View All Breeds in Database"):
    st.dataframe(df, use_container_width=True, hide_index=True)
