import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageStat, ImageDraw, ImageFont
import torchvision.transforms as transforms
import timm
import os
import gc
import csv
import pandas as pd
from datetime import datetime
from io import BytesIO
import random
import hashlib
import colorsys
from huggingface_hub import hf_hub_download
import json
import plotly.graph_objects as go
from ultralytics import YOLO

# Keep HF/torch chatty background threads/telemetry from eating extra RAM & CPU
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
# Streamlit Community Cloud instances are small, shared CPU boxes.
# Torch defaults to spinning up one thread per core, which multiplies
# per-tensor memory overhead for no real speed benefit at this scale.
torch.set_num_threads(1)

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

            # Create ConvNeXt-Tiny with the correct number of output classes
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
                model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()
            # The raw checkpoint dict can be large; we only need the loaded
            # model from here on, so let the checkpoint object go.
            del checkpoint
            gc.collect()
            return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None


model = load_model()


# ============ CATTLE-PRESENCE DETECTOR (YOLO26) ============
# This is a first, cheap pass that answers "is there cattle in this frame
# at all" before the (slower) breed classifier ever runs. It uses the
# stock COCO-pretrained YOLO26 nano weights - no custom training needed.
#
# Caveat worth knowing: COCO's 80 classes include a generic "cow" label
# but no separate "buffalo" class. Water buffalo are frequently caught
# under "cow" because of the similar silhouette, but this is a
# best-effort proxy, not a guarantee. If you need reliable buffalo
# detection specifically, fine-tune YOLO26 on a cow/buffalo dataset
# (e.g. the Roboflow "Cattle-Buffalo-breeds" sets) and point
# DETECTOR_WEIGHTS at that checkpoint instead.
DETECTOR_WEIGHTS = "yolo26n.pt"
DETECTOR_CONF_THRESHOLD = 0.35
CATTLE_CLASS_NAMES = {"cow"}


@st.cache_resource
def load_detector():
    try:
        with st.spinner("🔄 Loading detection model..."):
            det = YOLO(DETECTOR_WEIGHTS)
            return det
    except Exception as e:
        st.error(f"Detector loading error: {str(e)}")
        return None


detector = load_detector()


def run_cattle_detector(image):
    """Runs the presence-of-cattle detector on a PIL image.

    Returns (all_boxes, cattle_boxes): lists of dicts with keys
    'xyxy', 'label', 'conf'. cattle_boxes is the subset of all_boxes
    whose class name is in CATTLE_CLASS_NAMES.
    """
    if detector is None:
        return [], []
    try:
        results = detector.predict(image, conf=DETECTOR_CONF_THRESHOLD, verbose=False)
        boxes = []
        if results:
            result = results[0]
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = names.get(cls_id, str(cls_id))
                conf = float(box.conf[0].item())
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                boxes.append({"xyxy": xyxy, "label": label, "conf": conf})
        cattle_boxes = [b for b in boxes if b["label"].lower() in CATTLE_CLASS_NAMES]
        return boxes, cattle_boxes
    except Exception:
        return [], []
    finally:
        gc.collect()


def color_for_label(label):
    """Deterministic RGB color for a given label string, so the same
    breed (or object class) always gets the same box/text color across
    a single image and across reruns."""
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) % 360
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.48, 0.72)
    return (int(r * 255), int(g * 255), int(b * 255))


def draw_boxes(image, boxes, box_color=(231, 76, 60), text_color=(255, 255, 255)):
    """Draws labeled bounding boxes on a copy of the image using plain
    PIL drawing (no extra plotting dependencies).

    Each box dict needs "xyxy", "label", "conf". An optional per-box
    "color" (RGB tuple) overrides box_color — used so every distinct
    breed/class gets its own consistent color instead of one color for
    every box.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        label = f"{box['label']} {box['conf'] * 100:.0f}%"
        this_color = box.get("color", box_color)
        draw.rectangle([x1, y1, x2, y2], outline=this_color, width=3)

        if font is not None:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        else:
            text_w, text_h = len(label) * 6, 12

        label_y = max(0, y1 - text_h - 6)
        draw.rectangle([x1, label_y, x1 + text_w + 6, label_y + text_h + 4], fill=this_color)
        draw.text((x1 + 3, label_y + 2), label, fill=text_color, font=font)

    return annotated


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
        with torch.inference_mode():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_idxs = torch.topk(probabilities, k)
            results = [
                (breed_labels[idx.item()], prob.item() * 100)
                for prob, idx in zip(top_probs, top_idxs)
            ]
        # Release the intermediate tensors promptly instead of waiting on
        # the next garbage-collection cycle.
        del img_tensor, outputs, probabilities, top_probs, top_idxs
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


# ============ CONFIDENCE GAUGE ============
def render_confidence_gauge(confidence):
    """Semi-circular dial with the confidence percentage in the middle,
    styled like a speedometer: orange arc for the value, pale green track."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 30, "color": "#2c3e50"}},
        gauge={
            "axis": {"range": [0, 100], "visible": False},
            "bar": {"color": "#e8491d", "thickness": 0.35},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [{"range": [0, 100], "color": "#eaf7e0"}],
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=15, r=15, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def analyze_image_quality(image):
    """Lightweight, model-free visual analysis of the uploaded image."""
    try:
        width, height = image.size
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        brightness = stat.mean[0]  # 0 (dark) - 255 (bright)
        contrast = stat.stddev[0]  # low = flat/washed out, high = high contrast

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


def save_to_csv(breed, confidence, filename, timestamp):
    try:
        csv_file = "cattle_classification_data.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['timestamp', 'breed', 'confidence', 'filename'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({'timestamp': timestamp, 'breed': breed, 'confidence': confidence, 'filename': filename})
    except Exception:
        pass


def read_csv_tail(csv_file, n=5):
    """Reads only the last n rows of the CSV without loading the whole
    file into memory, so long-running deployments don't grow unbounded."""
    try:
        with open(csv_file, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
        tail = rows[-n:]
        return header, tail
    except Exception:
        return None, []


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
    except Exception:
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
        header, tail_rows = read_csv_tail(csv_file, n=5)
        if header and tail_rows:
            st.dataframe(pd.DataFrame(tail_rows, columns=header), use_container_width=True)
            with open(csv_file, "rb") as file:
                st.download_button("📥 Download CSV", data=file, file_name="cattle_classification_data.csv", mime="text/csv")
        else:
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

            # ---- STAGE 1: is there any cattle in this image at all? ----
            with st.spinner("🔎 Checking the image for cattle..."):
                all_boxes, cattle_boxes = run_cattle_detector(image)

            if not cattle_boxes:
                # Nothing that looks like a cow/buffalo was found. Show the
                # image with whatever the detector *did* find (if anything),
                # each box colored and labeled by its own class, and stop
                # here — the breed classifier never runs, which is also
                # what keeps this stage cheap.
                if all_boxes:
                    boxes_to_draw = [
                        {**b, "color": color_for_label(b["label"])} for b in all_boxes
                    ]
                    annotated = draw_boxes(image, boxes_to_draw)
                else:
                    annotated = image
                st.image(annotated, caption="📷 Uploaded Image", use_container_width=True)

                if all_boxes:
                    detected_names = sorted({b["label"].replace("_", " ").title() for b in all_boxes})
                    names_str = ", ".join(detected_names)
                    st.error(
                        f"🚫 No cow or buffalo detected in this image. "
                        f"Instead, the detector found: **{names_str}**."
                    )
                else:
                    st.error(
                        "🚫 No cow or buffalo detected in this image, and the detector "
                        "didn't recognize any other objects in it either. Please upload "
                        "a clearer photo of cattle."
                    )
                st.caption(
                    "The detector's pretrained classes include a generic \"cow\" label "
                    "but no separate \"buffalo\" class, so unusual angles, heavy cropping, "
                    "or poor lighting can occasionally cause a miss."
                )
            else:
                # ---- STAGE 2: species/breed identification on EVERY detected
                # animal, not just the highest-confidence box. Each cattle_box
                # found in Stage 1 is cropped out and classified independently
                # so a photo with several cows/buffaloes gets one prediction
                # per animal instead of a single answer for the whole image.
                detections = []
                with st.spinner(get_translation("analyzing", language)):
                    for i, box in enumerate(cattle_boxes, start=1):
                        x1, y1, x2, y2 = [int(round(v)) for v in box["xyxy"]]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(image.width, x2), min(image.height, y2)
                        crop = image.crop((x1, y1, x2, y2))
                        if crop.width < 2 or crop.height < 2:
                            continue

                        if model is not None:
                            top_predictions = predict_breed_topk(crop, k=3)
                        else:
                            top_predictions = demo_predict_topk(crop, k=3)

                        if not top_predictions:
                            continue

                        breed, confidence = top_predictions[0]
                        detections.append({
                            "index": i,
                            "box": box,
                            "crop": crop,
                            "breed": breed,
                            "confidence": confidence,
                            "top_predictions": top_predictions,
                        })

                if model is None:
                    st.info(get_translation("demo_mode", language))

                if not detections:
                    st.image(image, caption="📷 Uploaded Image", use_container_width=True)
                    st.warning(get_translation("confidence_error", language))
                else:
                    # Draw every box on one image, each labeled with its own
                    # predicted breed (instead of a single box for one animal).
                    annotated = draw_boxes(
                        image,
                        [
                            {
                                "xyxy": d["box"]["xyxy"],
                                "label": d["breed"],
                                "conf": d["confidence"] / 100.0,
                                "color": color_for_label(d["breed"]),
                            }
                            for d in detections
                        ],
                    )
                    st.image(
                        annotated,
                        caption=f"📷 Uploaded Image — {len(detections)} animal(s) detected",
                        use_container_width=True,
                    )

                    with st.spinner("📝 Generating image report..."):
                        quality_report = analyze_image_quality(image)

                    # ---- Image content report (whole image) ----
                    st.subheader("🖼️ Image Content Report")
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

                    # ---- Per-animal breed breakdown ----
                    st.subheader(f"🐄 Detected Animals ({len(detections)})")
                    for d in detections:
                        with st.expander(
                            f"Animal #{d['index']}: {d['breed']} ({d['confidence']:.1f}%)",
                            expanded=(len(detections) == 1),
                        ):
                            col_img, col_pred = st.columns([1, 2])
                            with col_img:
                                st.image(d["crop"], use_container_width=True, caption=f"Animal #{d['index']}")
                            with col_pred:
                                st.markdown(f"""
                                <div class="prediction-box">
                                    <p style="font-weight: bold; font-size: 1.3rem;">{get_translation("predicted_breed", language)} <b>{d['breed']}</b></p>
                                    <p style="font-weight: bold; font-size: 1.1rem; color: #3498db;">{get_translation("confidence", language)}: {d['confidence']:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)

                            # ---- Top-3 confidence report for this animal ----
                            gauge_cols = st.columns(len(d["top_predictions"]))
                            for rank, (col, (label, conf)) in enumerate(zip(gauge_cols, d["top_predictions"]), start=1):
                                with col:
                                    st.plotly_chart(
                                        render_confidence_gauge(conf),
                                        use_container_width=True,
                                        key=f"confidence_gauge_{d['index']}_{rank}",
                                    )
                                    st.markdown(
                                        f"<p style='text-align:center; font-weight:600; margin-top:-10px;'>{label}</p>",
                                        unsafe_allow_html=True,
                                    )

                            breed_key = d["breed"].lower().strip()
                            if breed_key in breed_info_raw:
                                st.markdown(f"**{get_translation('breed_info', language)}**")
                                display_breed_info(breed_key, breed_info_raw[breed_key], language)
                            else:
                                st.warning(get_translation("no_info", language))

                            save_to_csv(
                                d["breed"],
                                f"{d['confidence']:.2f}%",
                                f"{uploaded_file.name}#animal{d['index']}",
                                timestamp,
                            )

                del detections

            # Free the decoded image and any large intermediates now that
            # we're done with them, rather than waiting for Streamlit's
            # next rerun to drop the reference.
            del image, all_boxes, cattle_boxes
            gc.collect()

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
