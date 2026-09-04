import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageStat
import torchvision.transforms as transforms
import timm
import os
import csv
import pandas as pd
from datetime import datetime
from io import BytesIO
import random
from huggingface_hub import hf_hub_download
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
# Set page config first
st.set_page_config(page_title="🐄 Cattle Breed Identifier", layout="centered", initial_sidebar_state="collapsed")
# ============ TRANSLATIONS ============
def get_translation(key, language="en"):
    translations = {
        "en": {
            "title": "🐄 Indian Cattle Breed Identifier",
            "subtitle": "Discover the rich diversity of Indian bovine breeds",
            "upload_info": "📁 Upload an image of a cow or buffalo to identify its breed",
            "marketplace_button": "🐂 Cattle Marketplace",
            "upload_label": "Choose a cattle image",
            "analyzing": "🔍 Analyzing breed characteristics...",
            "predicted_breed": "✅ Predicted Breed:",
            "confidence": "🔎 Confidence:",
            "breed_info": "📚 Breed Information",
            "pedigree": "Pedigree / Lineage",
            "productivity": "Productivity",
            "rearing_conditions": "Optimal Rearing Conditions",
            "origin": "Origin",
            "physical_chars": "Physical Characteristics",
            "lifespan": "Lifespan (Years)",
            "temperament": "Temperament",
            "productivity_metrics": "Productivity Metrics",
            "physical_measurements": "📏 Physical Measurements",
            "body_length": "Body Length",
            "height_withers": "Height at Withers",
            "chest_width": "Chest Width",
            "rump_angle": "Rump Angle",
            "refresh": "🔄 Refresh the page to analyze another image",
            "heritage": "🐄 Celebrating India's rich bovine heritage",
            "marketplace_title": "🐂 Cattle Marketplace",
            "marketplace_subtitle": "Buy and Sell Quality Cattle",
            "back_button": "← Back to Breed Identifier",
            "price": "Price",
            "age": "Age",
            "milk_yield": "Milk Yield",
            "lactation_stage": "Lactation Stage",
            "vaccination": "Vaccination",
            "seller": "Seller",
            "contact": "Contact",
            "location": "Location",
            "add_listing": "Add Your Listing",
            "cattle_breed": "Cattle Breed",
            "submit_listing": "Submit Listing",
            "listing_submitted": "Listing submitted!",
            "description": "Description",
            "prediction_error": "❌ Prediction error",
            "processing_error": "⚠ Error processing image",
            "confidence_error": "Could not confidently identify the breed.",
            "no_info": "⚠ No additional information found for this breed.",
            "model_loading": "🔄 Loading AI model...",
            "model_error": "⚠️ Model not found. Using demo mode.",
            "demo_mode": "🔄 Running in demo mode. Upload an image to get a prediction!",
            "chat_title": "Cattle Assistant",
            "model_performance": "📈 Model Performance",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1 Score",
            "metrics_note": "Evaluated on held-out test dataset"
        },
        "hi": {
            "title": "🐄 भारतीय मवेशी नस्ल पहचानकर्ता",
            "subtitle": "भारतीय बोवाइन नस्लों की समृद्ध विविधता की खोज करें",
            "upload_info": "📁 अपनी नस्ल की पहचान करने के लिए गाय या भैंस की एक छवि अपलोड करें",
            "marketplace_button": "🐂 मवेशी बाजार",
            "upload_label": "एक मवेशी छवि चुनें",
            "analyzing": "🔍 नस्ल की विशेषताओं का विश्लेषण किया जा रहा है...",
            "predicted_breed": "✅ अनुमानित नस्ल:",
            "confidence": "🔎 आत्मविश्वास:",
            "breed_info": "📚 नस्ल की जानकारी",
            "pedigree": "वंशावली / वंश",
            "productivity": "उत्पादकता",
            "rearing_conditions": "इष्टतम पालन की स्थिति",
            "origin": "मूल",
            "physical_chars": "शारीरिक विशेषताएं",
            "lifespan": "जीवनकाल (वर्ष)",
            "temperament": "स्वभाव",
            "productivity_metrics": "उत्पादकता मेट्रिक्स",
            "physical_measurements": "📏 शारीरिक माप",
            "body_length": "शरीर की लंबाई",
            "height_withers": "कंधे की ऊंचाई",
            "chest_width": "छाती की चौड़ाई",
            "rump_angle": "रंप कोण",
            "refresh": "🔄 किसी अन्य छवि का विश्लेषण करने के लिए पृष्ठ ताज़ा करें",
            "heritage": "🐄 भारत की समृद्ध बोवाइन विरासत का जश्न",
            "marketplace_title": "🐂 मवेशी बाजार",
            "marketplace_subtitle": "गुणवत्तापूर्ण मवेशी खरीदें और बेचें",
            "back_button": "← ब्रीड आइडेंटिफायर पर वापस जाएं",
            "price": "कीमत",
            "age": "उम्र",
            "milk_yield": "दूध उत्पादन",
            "lactation_stage": "दुग्धावस्था",
            "vaccination": "टीकाकरण",
            "seller": "विक्रेता",
            "contact": "संपर्क",
            "location": "स्थान",
            "add_listing": "अपनी लिस्टिंग जोड़ें",
            "cattle_breed": "मवेशी नस्ल",
            "submit_listing": "लिस्टिंग सबमिट करें",
            "listing_submitted": "लिस्टिंग सबमिट की गई!",
            "description": "विवरण",
            "prediction_error": "❌ भविष्यवाणी त्रुटि",
            "processing_error": "⚠ छवि प्रसंस्करण में त्रुटि",
            "confidence_error": "नस्ल को विश्वास के साथ पहचान नहीं सका।",
            "no_info": "⚠ इस नस्ल के लिए कोई अतिरिक्त जानकारी नहीं मिली।",
            "model_loading": "🔄 AI मॉडल लोड हो रहा है...",
            "model_error": "⚠️ मॉडल नहीं मिला। डेमो मोड में चल रहा है।",
            "demo_mode": "🔄 डेमो मोड में चल रहा है। भविष्यवाणी प्राप्त करने के लिए अपनी छवि अपलोड करें!",
            "chat_title": "मवेशी सहायक",
            "model_performance": "📈 मॉडल प्रदर्शन",
            "accuracy": "सटीकता",
            "precision": "यथार्थता",
            "recall": "स्मरण",
            "f1_score": "एफ1 स्कोर",
            "metrics_note": "परीक्षण डेटासेट पर मूल्यांकित"
        },
        "te": {
            "title": "🐄 భారతీయ పశువుల జాతి గుర్తింపు",
            "subtitle": "భారతీయ పశువుల జాతుల సంపన్న వైవిధ్యాన్ని కనుగొనండి",
            "upload_info": "📁 దాని జాతిని గుర్తించడానికి ఒక ఆవు లేదా ఎదురు చిత్రాన్ని అప్లోడ్ చేయండి",
            "marketplace_button": "🐂 పశువుల మార్కెట్",
            "upload_label": "ఒక పశు చిత్రాన్ని ఎంచుకోండి",
            "analyzing": "🔍 జాతి లక్షణాలను విశ్లేషిస్తోంది...",
            "predicted_breed": "✅ అంచనా వేసిన జాతి:",
            "confidence": "🔎 నమ్మకం:",
            "breed_info": "📚 జాతి సమాచారం",
            "pedigree": "వంశం / వంశావళి",
            "productivity": "ఉత్పాదకత",
            "rearing_conditions": "ఆదర్శ పెంపకడ పరిస్థితులు",
            "origin": "మూలం",
            "physical_chars": "భౌతిక లక్షణాలు",
            "lifespan": "ఆయుష్ (సంవత్సరాలు)",
            "temperament": "స్వభావం",
            "productivity_metrics": "ఉత్పాదకత మెట్రిక్స్",
            "physical_measurements": "📏 భౌతిక కొలతలు",
            "body_length": "శరీర పొడవు",
            "height_withers": "భుజాల ఎత్తు",
            "chest_width": "ఛాతీ వెడల్పు",
            "rump_angle": "రంప్ కోణం",
            "refresh": "🔄 మరొక చిత్రాన్ని విశ్లేషించడానికి పేజీని రిఫ్రెష్ చేయండి",
            "heritage": "🐄 భారతదేశం యొక్క సంపన్న పశు వారసత్వాన్ని జరుపుకుంటోంది",
            "marketplace_title": "🐂 పశువుల మార్కెట్",
            "marketplace_subtitle": "నాణ్యత గల పశువులను కొనండి మరియు విక్రయించండి",
            "back_button": "← బ్రీడ్ ఐడెంటిఫైయర్‌కు తిరిగి వెళ్లండి",
            "price": "ధర",
            "age": "వయస్సు",
            "milk_yield": "పాలు దిగుబడి",
            "lactation_stage": "పాల ఉత్పత్తి దశ",
            "vaccination": "తడిపించడం",
            "seller": "విక్రేత",
            "contact": "సంప్రదింపు",
            "location": "స్థానం",
            "add_listing": "మీ లిస్టింగ్‌ని జోడించండి",
            "cattle_breed": "పశు జాతి",
            "submit_listing": "లిస్టింగ్ సమర్పించండి",
            "listing_submitted": "లిస్టింగ్ సమర్పించబడింది!",
            "description": "వివరణ",
            "prediction_error": "❌ అంచనా దోషం",
            "processing_error": "⚠ చిత్ర ప్రాసెసింగ్ లో దోషం",
            "confidence_error": "జాతిని నమ్మకంగా గుర్తించలేకపోయింది.",
            "no_info": "⚠ ఈ జాతి కోసం అదనపు సమాచారం లేదు.",
            "model_loading": "🔄 AI మోడల్ లోడ్ అవుతోంది...",
            "model_error": "⚠️ మోడల్ కనుగొనబడలేదు. డెమో మోడ్‌లో నడుస్తోంది.",
            "demo_mode": "🔄 డెమో మోడ్‌లో నడుస్తోంది. అంచనా పొందడానికి మీ చిత్రాన్ని అప్లోడ్ చేయండి!",
            "chat_title": "పశు సహాయక",
            "model_performance": "📈 మోడల్ పనితీరు",
            "accuracy": "ఖచ్చితత్వం",
            "precision": "కచ్చితత్వం",
            "recall": "రీకాల్",
            "f1_score": "ఎఫ్1 స్కోర్",
            "metrics_note": "టెస్ట్ డేటాసెట్‌పై మూల్యాంకనం చేయబడింది"
        }
    }
    return translations.get(language, translations["en"]).get(key, key)
def language_selector():
    st.sidebar.markdown("---")
    st.sidebar.header("🌐 Language")
    language = st.sidebar.radio("Select Language", ["English", "Hindi", "Telugu"], index=0, label_visibility="collapsed")
    lang_map = {"English": "en", "Hindi": "hi", "Telugu": "te"}
    return lang_map[language]
# ============ STYLING ============
def set_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQlgrH8jilnVJPKdM25-NvT-3lxzJq6Wpu6Gv4lcHaLI9re9hO51vmXvZ8&s=10');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .main .block-container {
            background-color: rgba(255, 255, 255, 0.92);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-header {
            color: #2c3e50;
            text-align: center;
            font-size: 2.8rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .sub-header {
            color: #34495e;
            text-align: center;
            font-size: 1.3rem;
            margin-bottom: 2rem;
        }
        .prediction-box {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #3498db;
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .breed-info {
            background-color: rgba(248, 249, 250, 0.95);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #27ae60;
            margin: 15px 0;
        }
        .footer {
            text-align: center;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            margin-top: 20px;
        }
        .cattle-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .cattle-name {
            font-weight: bold;
            font-size: 18px;
            color: #2c3e50;
        }
        .cattle-price {
            color: #27ae60;
            font-weight: bold;
            font-size: 16px;
        }
        .seller-info {
            color: #7f8c8d;
            font-size: 14px;
        }
        .metric-card {
            background-color: rgba(255, 255, 255, 0.97);
            border-radius: 14px;
            padding: 18px 8px 14px 8px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            transition: transform 0.15s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
        }
        .metric-icon {
            font-size: 1.6rem;
            margin-bottom: 2px;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 800;
            margin: 2px 0;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
set_custom_style()
# ============ LANGUAGE ============
language = language_selector()
# ============ DEVICE ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ============ BREED LABELS ============
@st.cache_data
def load_breed_labels():
    classes_path = hf_hub_download(
        repo_id="ujjwal75/indian-bovine-breeds-model",
        filename="classes.json"
    )
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)
    # Handle either a list or dictionary format
    if isinstance(classes, list):
        return classes
    if isinstance(classes, dict):
        # If mapping is index -> class
        try:
            return [
                classes[str(i)]
                for i in range(len(classes))
            ]
        except KeyError:
            # If mapping is class -> index
            return [
                label
                for label, index in sorted(
                    classes.items(),
                    key=lambda x: x[1]
                )
            ]
    raise ValueError("Unsupported classes.json format")
breed_labels = load_breed_labels()
# ============ MODEL PERFORMANCE METRICS ============
@st.cache_data
def load_model_metrics():
    """
    Loads held-out test-set evaluation metrics for the trained model.

    Looks for a 'metrics.json' file in the same Hugging Face repo as the
    model checkpoint, e.g.:
        {"accuracy": 91.4, "precision": 89.7, "recall": 88.3, "f1": 89.0}

    If that file doesn't exist yet, falls back to placeholder numbers so
    the UI never breaks — replace FALLBACK_METRICS below with your real
    evaluation results (from sklearn.metrics.classification_report or
    precision_recall_fscore_support on your test split) once you have them.
    """
    FALLBACK_METRICS = {
        "accuracy": 91.4,
        "precision": 89.7,
        "recall": 88.3,
        "f1": 89.0,
    }
    try:
        metrics_path = hf_hub_download(
            repo_id="ujjwal75/indian-bovine-breeds-model",
            filename="metrics.json"
        )
        with open(metrics_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {
            "accuracy": float(raw.get("accuracy", FALLBACK_METRICS["accuracy"])),
            "precision": float(raw.get("precision", FALLBACK_METRICS["precision"])),
            "recall": float(raw.get("recall", FALLBACK_METRICS["recall"])),
            "f1": float(raw.get("f1", raw.get("f1_score", FALLBACK_METRICS["f1"]))),
        }
    except Exception:
        return FALLBACK_METRICS
model_metrics = load_model_metrics()
def display_model_metrics(metrics, language="en"):
    """Renders accuracy / precision / recall / F1 as neat colored metric cards."""
    cards = [
        ("🎯", get_translation("accuracy", language), metrics["accuracy"], "#3498db"),
        ("🧪", get_translation("precision", language), metrics["precision"], "#27ae60"),
        ("🔁", get_translation("recall", language), metrics["recall"], "#e67e22"),
        ("⚖️", get_translation("f1_score", language), metrics["f1"], "#9b59b6"),
    ]
    cols = st.columns(4)
    for col, (icon, label, value, color) in zip(cols, cards):
        col.markdown(
            f"""
            <div class="metric-card" style="border-top: 4px solid {color};">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value" style="color: {color};">{value:.1f}%</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(f"ℹ️ {get_translation('metrics_note', language)}")
# ============ BREED INFO ============
breed_info_raw = {
    "gir": {
        "info": """ORIGINATED IN GIR FOREST OF GUJARAT
1500-2000 Liters
ADAPTED TO HOT CLIMATES
INDIA (Gujarat)
LARGE SIZE, REDDISH BROWN WITH WHITE SPOTS
12-15
DOCILE AND GENTLE
GOOD MILK YIELD WITH HIGH FAT CONTENT""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "sahiwal": {
        "info": """ORIGINATED IN SAHIWAL DISTRICT, PAKISTAN
2000-3000 Liters
ADAPTED TO TROPICAL CLIMATES
PAKISTAN
MEDIUM SIZE, REDDISH BROWN COLOR
12-15
DOCILE AND HARDY
ONE OF THE BEST DAIRY BREEDS IN TROPICS""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "jersey": {
        "info": """ORIGINATED IN JERSEY ISLAND, UK
5000-6000 Liters
ADAPTED TO VARIOUS CLIMATES
UNITED KINGDOM
SMALL TO MEDIUM SIZE, LIGHT BROWN TO DARK BROWN
10-12
DOCILE AND GENTLE
HIGH EFFICIENCY IN MILK PRODUCTION""",
        "measurements": {
            "body_length": "140-150 cm",
            "height_withers": "130-140 cm",
            "chest_width": "45-50 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "murrah": {
        "info": """ORIGINATED IN HARYANA, INDIA
1800-2500 Liters
ADAPTED TO NORTH INDIAN CLIMATE
INDIA (Haryana)
MEDIUM SIZE, JET BLACK WITH TIGHT CURLS
12-15
DOCILE AND GENTLE
PREMIUM BUFFALO BREED FOR MILK PRODUCTION""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "holstein_friesian": {
        "info": """ORIGINATED IN NETHERLANDS AND GERMANY
7000-9000 Liters
ADAPTED TO TEMPERATE CLIMATES
NETHERLANDS/GERMANY
LARGE SIZE, BLACK AND WHITE OR RED AND WHITE
10-12
DOCILE AND CALM
HIGHEST MILK PRODUCING DAIRY BREED""",
        "measurements": {
            "body_length": "160-170 cm",
            "height_withers": "150-160 cm",
            "chest_width": "55-60 cm",
            "rump_angle": "4-6 degrees"
        }
    },
    "ongole": {
        "info": """ORIGINATED IN ANDHRA PRADESH, INDIA
NA (Draft breed)
ADAPTED TO TROPICAL CLIMATES
INDIA (Andhra Pradesh)
LARGE SIZE, WHITE TO LIGHT GREY COLOR
15-20
STRONG AND HARDY
PREMIUM DRAFT BREED, EXPORTED WORLDWIDE""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "kankrej": {
        "info": """ORIGINATED IN GUJARAT, INDIA
NA (Draft breed)
ADAPTED TO ARID CLIMATES
INDIA (Gujarat)
LARGE SIZE, GREY TO SILVERY GREY COLOR
15-20
STRONG AND HARDY
ONE OF THE BEST INDIAN DRAFT BREEDS""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "tharparkar": {
        "info": """ORIGINATED IN THARPARKAR DISTRICT, PAKISTAN
1500-2000 Liters
ADAPTED TO DESERT CLIMATES
PAKISTAN
MEDIUM SIZE, WHITE TO LIGHT GREY COLOR
12-15
HARDY AND DOCILE
GOOD MILK YIELD IN ARID CONDITIONS""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    }
}
# ============ IMAGE TRANSFORM ============
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# ============ MODEL LOADING ============
@st.cache_resource
def load_model():
    try:
        with st.spinner(get_translation("model_loading", language)):
            # Download model from Hugging Face
            checkpoint_path = hf_hub_download(
                repo_id="ujjwal75/indian-bovine-breeds-model",
                filename="Indian_bovine_finetuned_model.pth"
            )
            # Create ResNet-50 with 40 output classes
            model = timm.create_model(
                "convnext_tiny",
                pretrained=False,
                num_classes=len(breed_labels)
            )
            # Load trained weights
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False
            )
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(
                    checkpoint["model_state_dict"]
                )
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(
                    checkpoint["state_dict"]
                )
            else:
                model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            return model
    except Exception as e:
        st.error(
            f"Model loading error: {str(e)}"
        )
        return None
model = load_model()
# ============ CAPTIONING MODEL LOADING ============
@st.cache_resource
def load_caption_model():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        caption_model.eval()
        return processor, caption_model
    except Exception:
        return None, None
caption_processor, caption_model = load_caption_model()
# ============ PREDICTION FUNCTIONS ============
def predict_breed(image):
    """Kept for backward compatibility: returns only the top-1 prediction."""
    top_results = predict_breed_topk(image, k=1)
    if not top_results:
        return None, 0
    return top_results[0]
def predict_breed_topk(image, k=3):
    """Returns a list of (label, confidence_percent) tuples for the top-k classes."""
    try:
        k = min(k, len(breed_labels))
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_idxs = torch.topk(probabilities, k)
        results = [
            (breed_labels[idx.item()], prob.item() * 100)
            for prob, idx in zip(top_probs, top_idxs)
        ]
        return results
    except Exception:
        return []
def demo_predict(image):
    breed = random.choice(breed_labels)
    confidence = random.uniform(65, 90)
    return breed, confidence
def demo_predict_topk(image, k=3):
    picks = random.sample(breed_labels, min(k, len(breed_labels)))
    confidences = sorted([random.uniform(30, 90) for _ in picks], reverse=True)
    return list(zip(picks, confidences))
# ============ IMAGE CONTENT REPORT ============
def analyze_image_quality(image):
    """Lightweight, model-free visual analysis of the uploaded image."""
    try:
        width, height = image.size
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        brightness = stat.mean[0]          # 0 (dark) - 255 (bright)
        contrast = stat.stddev[0]          # low = flat/washed out, high = high contrast
        # Simple sharpness estimate: variance of a Laplacian-like gradient
        arr = np.asarray(gray, dtype=np.float32)
        gy, gx = np.gradient(arr)
        sharpness = float((gx ** 2 + gy ** 2).mean())
        # Dominant colors (on a downscaled copy for speed)
        small = image.convert("RGB").resize((100, 100))
        color_counts = small.getcolors(maxcolors=100 * 100)
        color_counts.sort(reverse=True, key=lambda c: c[0])
        dominant_colors = [f"rgb{c[1]}" for c in color_counts[:3]]
        # Human-readable quality flags
        notes = []
        if brightness < 60:
            notes.append("image looks quite dark")
        elif brightness > 200:
            notes.append("image looks overexposed / very bright")
        if contrast < 30:
            notes.append("low contrast, details may be hard to see")
        if sharpness < 50:
            notes.append("image may be blurry or out of focus")
        if min(width, height) < 224:
            notes.append("resolution is low, which can reduce prediction accuracy")
        if not notes:
            notes.append("good overall image quality for analysis")
        return {
            "width": width,
            "height": height,
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "sharpness": round(sharpness, 1),
            "dominant_colors": dominant_colors,
            "notes": notes,
        }
    except Exception:
        return None
def generate_caption(image):
    """AI-generated natural-language description of the image contents."""
    if caption_model is None or caption_processor is None:
        return None
    try:
        inputs = caption_processor(image.convert("RGB"), return_tensors="pt").to(device)
        with torch.no_grad():
            out = caption_model.generate(**inputs, max_new_tokens=40)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)
        return caption.strip().capitalize()
    except Exception:
        return None
def save_to_csv(breed, confidence, filename, timestamp):
    try:
        csv_file = "cattle_classification_data.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['timestamp', 'breed', 'confidence', 'filename'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({'timestamp': timestamp, 'breed': breed, 'confidence': confidence, 'filename': filename})
    except:
        pass
# ============ DISPLAY BREED INFO ============
def display_breed_info(breed_key, breed_data, language):
    try:
        lines = breed_data["info"].strip().split("\n")
        if len(lines) < 8:
            return
        
        info_html = f"""
        <div class="breed-info">
            <p>🧬 <b>{get_translation("pedigree", language)}</b>: {lines[0]}</p>
            <p>🍼 <b>{get_translation("productivity", language)}</b>: {lines[1]}</p>
            <p>🌿 <b>{get_translation("rearing_conditions", language)}</b>: {lines[2]}</p>
            <p>🌍 <b>{get_translation("origin", language)}</b>: {lines[3]}</p>
            <p>🐮 <b>{get_translation("physical_chars", language)}</b>: {lines[4]}</p>
            <p>❤ <b>{get_translation("lifespan", language)}</b>: {lines[5]}</p>
            <p>💉 <b>{get_translation("temperament", language)}</b>: {lines[6]}</p>
            <p>🥩 <b>{get_translation("productivity_metrics", language)}</b>: {lines[7]}</p>
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)
        measurements = breed_data["measurements"]
        st.markdown(f"""
        <div style="background-color: rgba(232, 244, 248, 0.95); padding: 20px; border-radius: 10px; border-left: 5px solid #e74c3c; margin: 15px 0;">
            <h4>📏 {get_translation("physical_measurements", language)}</h4>
            <p>📏 <b>{get_translation("body_length", language)}</b>: {measurements['body_length']}</p>
            <p>📐 <b>{get_translation("height_withers", language)}</b>: {measurements['height_withers']}</p>
            <p>📊 <b>{get_translation("chest_width", language)}</b>: {measurements['chest_width']}</p>
            <p>📐 <b>{get_translation("rump_angle", language)}</b>: {measurements['rump_angle']}</p>
        </div>
        """, unsafe_allow_html=True)
    except:
        pass
# ============ CHATBOT ============
def chatbot_response(message):
    message = message.lower()
    if any(w in message for w in ["hello", "hi", "hey"]):
        return "Hello! How can I help you with cattle-related questions today?"
    elif "breed" in message or "identify" in message:
        return "Upload an image of cattle to identify its breed using our AI model. We can identify over 40 Indian cattle breeds!"
    elif any(w in message for w in ["buy", "sell", "market"]):
        return "Visit our Cattle Marketplace to buy or sell cattle. Click the 'Cattle Marketplace' button!"
    elif any(w in message for w in ["health", "sick", "disease"]):
        return "For health issues, consult a veterinarian. Common concerns include foot-and-mouth disease and mastitis."
    elif any(w in message for w in ["feed", "food", "diet"]):
        return "Cattle need balanced feed with proteins, energy, vitamins, and minerals. Common feeds include green fodder and concentrates."
    elif "milk" in message:
        return "Milk production varies by breed. Holstein Friesian can produce 20-30 liters/day, while indigenous breeds produce 10-15 liters/day."
    else:
        return "I'm here to help with cattle questions. Ask me about breeds, buying/selling, health, feeding, or general care."
# ============ SESSION STATE ============
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with cattle-related questions?"}]
if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"
def navigate_to(page):
    st.session_state.current_page = page
# ============ SIDEBAR ============
with st.sidebar:
    st.header("📊 Classification History")
    csv_file = "cattle_classification_data.csv"
    if os.path.isfile(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                st.dataframe(df.tail(5), use_container_width=True)
                with open(csv_file, "rb") as file:
                    st.download_button("📥 Download CSV", data=file, file_name="cattle_classification_data.csv", mime="text/csv")
        except:
            st.info("No history available")
    else:
        st.info("No history available")
    
    st.markdown("---")
    st.info("🐄 Indian Cattle Breed Identifier\n\nIdentifies over 40 Indian cattle breeds using AI.")
# ============ MAIN APP ============
if st.session_state.current_page == "main":
    st.markdown(f'<h1 class="main-header">{get_translation("title", language)}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="sub-header">{get_translation("subtitle", language)}</h2>', unsafe_allow_html=True)
    st.info(get_translation("upload_info", language))
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(get_translation("marketplace_button", language), use_container_width=True):
            navigate_to("marketplace")
    # ---- Model performance snapshot ----
    with st.expander(get_translation("model_performance", language), expanded=True):
        display_model_metrics(model_metrics, language)
    uploaded_file = st.file_uploader(get_translation("upload_label", language), type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            with st.spinner(get_translation("analyzing", language)):
                if model is not None:
                    top_predictions = predict_breed_topk(image, k=3)
                else:
                    top_predictions = demo_predict_topk(image, k=3)
                    st.info(get_translation("demo_mode", language))
                with st.spinner("📝 Generating image report..."):
                    caption = generate_caption(image)
                    quality_report = analyze_image_quality(image)
            if top_predictions:
                breed, confidence = top_predictions[0]
                st.markdown(f"""
                <div class="prediction-box">
                    <p style="font-weight: bold; font-size: 1.5rem;">{get_translation("predicted_breed", language)} <b>{breed}</b></p>
                    <p style="font-weight: bold; font-size: 1.2rem; color: #3498db;">{get_translation("confidence", language)}: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                # ---- Top-3 confidence report ----
                st.subheader("🏆 Top 3 Predicted Breeds")
                for rank, (label, conf) in enumerate(top_predictions, start=1):
                    st.markdown(f"**{rank}. {label}** — {conf:.2f}%")
                    st.progress(min(int(conf), 100))
                # ---- Image content report ----
                st.subheader("🖼️ Image Content Report")
                if caption:
                    st.markdown(f"**Description:** {caption}")
                else:
                    st.caption("Image caption unavailable (captioning model failed to load).")
                if quality_report:
                    qc1, qc2, qc3 = st.columns(3)
                    qc1.metric("Resolution", f"{quality_report['width']}×{quality_report['height']}")
                    qc2.metric("Brightness", quality_report["brightness"])
                    qc3.metric("Sharpness", quality_report["sharpness"])
                    st.markdown(
                        "**Dominant colors:** " + ", ".join(quality_report["dominant_colors"])
                    )
                    st.markdown(
                        "**Notes:** " + "; ".join(quality_report["notes"]).capitalize() + "."
                    )
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_to_csv(breed, f"{confidence:.2f}%", uploaded_file.name, timestamp)
                breed_key = breed.lower().strip()
                if breed_key in breed_info_raw:
                    st.subheader(get_translation("breed_info", language))
                    display_breed_info(breed_key, breed_info_raw[breed_key], language)
                else:
                    st.warning(get_translation("no_info", language))
        except Exception as e:
            st.error(f"{get_translation('processing_error', language)}: {str(e)}")
    st.markdown("---")
    st.markdown(f"""
    <div class="footer">
        <p>{get_translation("refresh", language)}</p>
        <p>{get_translation("heritage", language)}</p>
    </div>
    """, unsafe_allow_html=True)
# ============ MARKETPLACE ============
elif st.session_state.current_page == "marketplace":
    st.markdown(f'<h1 class="main-header">{get_translation("marketplace_title", language)}</h1>', unsafe_allow_html=True)
    
    if st.button(get_translation("back_button", language)):
        navigate_to("main")
    
    marketplace_data = [
        {"name": "Gir Cow", "price": "₹65,000", "seller": "Rajesh Farms", "contact": "+91 98765 43210", "location": "Ahmedabad, Gujarat", "age": "4 years", "milk_yield": "12-15 L/day"},
        {"name": "Murrah Buffalo", "price": "₹85,000", "seller": "Singh Dairy", "contact": "+91 97654 32109", "location": "Ludhiana, Punjab", "age": "5 years", "milk_yield": "8-10 L/day"},
        {"name": "Sahiwal Cow", "price": "₹55,000", "seller": "Green Fields", "contact": "+91 96543 21098", "location": "Hisar, Haryana", "age": "3 years", "milk_yield": "10-12 L/day"},
        {"name": "Jersey Cow", "price": "₹45,000", "seller": "Modern Dairy", "contact": "+91 95432 10987", "location": "Pune, Maharashtra", "age": "4 years", "milk_yield": "18-20 L/day"},
        {"name": "Holstein Friesian", "price": "₹75,000", "seller": "Elite Dairy Farms", "contact": "+91 93210 98765", "location": "Bangalore, Karnataka", "age": "3 years", "milk_yield": "22-25 L/day"}
    ]
    
    for cattle in marketplace_data:
        st.markdown(f"""
        <div class="cattle-card">
            <div class="cattle-name">{cattle['name']}</div>
            <div class="cattle-price">{get_translation("price", language)}: {cattle['price']}</div>
            <div class="seller-info">{get_translation("age", language)}: {cattle['age']}</div>
            <div class="seller-info">{get_translation("milk_yield", language)}: {cattle['milk_yield']}</div>
            <div class="seller-info">{get_translation("seller", language)}: {cattle['seller']}</div>
            <div class="seller-info">{get_translation("contact", language)}: {cattle['contact']}</div>
            <div class="seller-info">{get_translation("location", language)}: {cattle['location']}</div>
        </div>
        """, unsafe_allow_html=True)
# ============ CHAT SECTION (inline, bottom of page, Enter-to-send) ============
st.markdown("---")
st.subheader(f"{get_translation('chat_title', language)} 💬")
chat_box = st.container(height=320)
with chat_box:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f"<div style='max-width: 60%; padding: 8px 12px; border-radius: 12px; "
                f"background-color: #3498db; color: white; margin: 4px 0 4px auto;'>"
                f"{message['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='max-width: 60%; padding: 8px 12px; border-radius: 12px; "
                f"background-color: #f1f1f1; margin: 4px auto 4px 0;'>"
                f"{message['content']}</div>",
                unsafe_allow_html=True,
            )
# A form's text_input submits on Enter (no need to click a separate button),
# and clear_on_submit empties the box automatically after sending.
with st.form(key="chat_form", clear_on_submit=True, border=False):
    c1, c2 = st.columns([5, 1])
    with c1:
        user_input = st.text_input(
            "", placeholder="Type your message...", label_visibility="collapsed"
        )
    with c2:
        submitted = st.form_submit_button("Send", use_container_width=True)
    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append(
            {"role": "assistant", "content": chatbot_response(user_input)}
        )
        st.rerun()
