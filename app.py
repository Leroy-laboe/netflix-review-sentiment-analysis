import streamlit as st
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
st.set_page_config(page_title="Netflix Sentiment Analysis", page_icon="🎬", layout="centered")

# Download NLTK resources
@st.cache_resource
def download_nltk():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk()

# ==========================================
# 2. PREPROCESSING LOGIC (From Notebook)
# ==========================================
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

preprocessor = TextPreprocessor()

# ==========================================
# 3. MODEL LOADING
# ==========================================
@st.cache_resource
def load_model_artifacts():
    # Paths to the model files
    # Check both root and backend/model_artifacts for convenience
    model_path = 'model.pkl'
    vec_path = 'vectorizer.pkl'
    
    if not os.path.exists(model_path):
        model_path = 'backend/model_artifacts/model.pkl'
        vec_path = 'backend/model_artifacts/vectorizer.pkl'
        
    if not os.path.exists(model_path):
        return None, None
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

# ==========================================
# 4. USER INTERFACE (UX/UI OVERHAUL)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@300;400;700&display=swap');

    /* Foundation: Deep Shadows & Netflix Gradient */
    .stApp {
        background: radial-gradient(circle at top, #221f1f 0%, #000000 100%);
        background-attachment: fixed;
        color: #FFFFFF;
    }

    /* Cinematic Overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: url('https://assets.nflxext.com/ffe/siteui/vlv3/f841d4c7-10e1-40af-bcae-07a3f8dc141a/f6d7434e-d6de-4185-a6d4-c77a2d0873fb/US-en-20220502-popsignuptwoweeks-perspective_alpha_website_medium.jpg');
        background-size: cover;
        opacity: 0.15;
        z-index: -1;
    }

    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }

    .netflix-logo {
        font-family: 'Bebas Neue', cursive;
        color: #E50914;
        font-size: 6rem;
        text-align: center;
        letter-spacing: 4px;
        margin-bottom: 0px;
        text-shadow: 0px 4px 15px rgba(229, 9, 20, 0.5);
        animation: fadeInDown 1s ease-out;
    }

    .tagline {
        text-align: center;
        color: #B3B3B3;
        font-weight: 300;
        font-size: 1.2rem;
        letter-spacing: 2px;
        margin-top: -15px;
        margin-bottom: 50px;
        animation: fadeInUp 1.2s ease-out;
    }

    /* The Main Card (Glassmorphism) */
    .main-card {
        background: rgba(20, 20, 20, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 40px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(12px);
        margin: 0 auto;
        max-width: 650px;
    }

    /* Modern Input Styling */
    .stTextArea textarea {
        background-color: rgba(30, 30, 30, 0.9) !important;
        color: #FFFFFF !important;
        border: 2px solid #333 !important;
        border-radius: 8px !important;
        padding: 18px !important;
        font-size: 1.1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stTextArea textarea:focus {
        border-color: #E50914 !important;
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.4) !important;
        background-color: #1a1a1a !important;
    }

    /* Striking CTA Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #E50914 0%, #B81D24 100%) !important;
        color: white !important;
        border: none !important;
        width: 100% !important;
        padding: 20px !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 8px !important;
        margin-top: 15px;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(184, 29, 36, 0.4);
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 25px rgba(229, 9, 20, 0.6);
        filter: brightness(1.2);
    }

    div.stButton > button:first-child:active {
        transform: translateY(-1px);
    }

    /* Result Aesthetics */
    .result-container {
        margin-top: 30px;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: fadeIn 0.8s ease-in;
    }

    /* Custom Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Mobile Cleanup */
    @media (max-width: 768px) {
        .netflix-logo { font-size: 4rem; }
        .tagline { font-size: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# App Content Wrapped in Styled Container
st.markdown('<div style="text-align: center;"><img src="https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg" width="200"></div>', unsafe_allow_html=True)
st.markdown('<h1 class="netflix-logo">SENTIMENT ANALYZER</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">NLP SENTIMENT CLASSIFIER</p>', unsafe_allow_html=True)

model, vectorizer = load_model_artifacts()

if model is None:
    st.error("⚠️ Model files (model.pkl / vectorizer.pkl) not found in the root directory.")
else:
    # Use st.container for grouping
    with st.container():
        review_text = st.text_area("", placeholder="Enter your Netflix review here...", height=150)
        
        # Center the button logic via standard Streamlit (styling handles the strike)
        if st.button("Analyze Review"):
            if review_text.strip():
                with st.spinner("Decoding Emotions..."):
                    cleaned_text = preprocessor.preprocess(review_text)
                    vec_text = vectorizer.transform([cleaned_text])
                    prediction = model.predict(vec_text)[0]
                    
                    st.divider()
                    
                    # Sentiment specific UI feedback
                    if prediction == 'positive':
                        st.balloons()
                        st.markdown(f'<div class="result-container" style="border-top: 4px solid #46d369;">'
                                    f'<h2 style="color: #46d369;">POSITIVE</h2>'
                                    f'<p style="color: #B3B3B3;">The model classifies this review as positive.</p></div>', 
                                    unsafe_allow_html=True)
                    elif prediction == 'negative':
                        st.markdown(f'<div class="result-container" style="border-top: 4px solid #E50914;">'
                                    f'<h2 style="color: #E50914;">NEGATIVE</h2>'
                                    f'<p style="color: #B3B3B3;">The model classifies this review as negative.</p></div>', 
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-container" style="border-top: 4px solid #8c8c8c;">'
                                    f'<h2 style="color: #8c8c8c;">NEUTRAL</h2>'
                                    f'<p style="color: #B3B3B3;">The model classifies this review as neutral.</p></div>', 
                                    unsafe_allow_html=True)
            else:
                st.warning("Please enter a review to analyze.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown('<p style="text-align: center; color: #888; font-size: 0.9rem; font-weight: bold;">Group NLP project · Leroy Nyasha Mangwarara, Tadiwanashe & Fathima Zuha</p>', unsafe_allow_html=True)
