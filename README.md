# 🐄 Indian Cattle & Buffalo Breed Identification

A Streamlit web app that identifies Indian cattle and buffalo breeds from an uploaded photo using a fine-tuned deep learning model, and pairs the prediction with breed information, image-quality analysis, a mock marketplace, and a simple chatbot assistant.

**Live demo:** https://ps-5-cattle-buffalo-breed-identific.vercel.app

## Features

- **Breed classification** — Upload a JPG/PNG image and get the top-3 predicted breeds with confidence scores, powered by a `convnext_tiny` model (via [timm](https://github.com/huggingface/pytorch-image-models)) fine-tuned on Indian bovine breeds and hosted on Hugging Face Hub (`ujjwal75/indian-bovine-breeds-model`).
- **Confidence gauges** — Speedometer-style Plotly gauges visualize prediction confidence for the top-3 breeds.
- **Breed information** — Pedigree/lineage, productivity, rearing conditions, origin, physical characteristics, lifespan, temperament, and body measurements for known breeds (Gir, Sahiwal, Jersey, Murrah, Holstein Friesian, Ongole, Kankrej, Tharparkar).
- **Image content report** — An auto-generated caption (via a BLIP image-captioning model) plus lightweight, model-free image-quality checks (resolution, brightness, contrast, sharpness, dominant colors).
- **Model performance panel** — Displays accuracy, precision, recall, and F1 score for the underlying model.
- **Cattle marketplace** — A sample listings page for buying/selling cattle.
- **Chat assistant** — A simple rule-based chatbot that answers questions about breeds, buying/selling, health, and feeding.
- **Classification history** — Every prediction is logged to `cattle_classification_data.csv`, viewable and downloadable from the sidebar.
- **Multi-language UI** — English, Hindi, and Telugu.
- **Demo mode** — If the model fails to load, the app falls back to random predictions so the UI still works end-to-end.

## Tech Stack

- [Streamlit](https://streamlit.io/) — web app framework
- [PyTorch](https://pytorch.org/) + [timm](https://github.com/huggingface/pytorch-image-models) — image classification model
- [Transformers](https://github.com/huggingface/transformers) (BLIP) — automatic image captioning
- [Hugging Face Hub](https://huggingface.co/) — model checkpoint and class-label hosting
- [Plotly](https://plotly.com/python/) — confidence gauge visualizations
- [Pandas](https://pandas.pydata.org/) — classification history log

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Vishwajeet-keni/PS-5-Cattle_Buffalo_Breed_Identification.git
cd PS-5-Cattle_Buffalo_Breed_Identification
pip install -r requirements.txt
```

If you're deploying on a platform that uses `packages.txt` for system-level dependencies (e.g. Streamlit Community Cloud), those will be installed automatically from that file.

### Running locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. On first run it downloads the classification model and class labels from Hugging Face Hub, so an internet connection is required.

## Usage

1. Open the app and select your preferred language from the sidebar.
2. Upload a clear image of a cow or buffalo (JPG/JPEG/PNG).
3. View the top-3 predicted breeds, confidence gauges, an auto-generated image caption, and image-quality notes.
4. Expand the breed information section for details on the top prediction (when available).
5. Visit the **Cattle Marketplace** to browse sample listings, or use the chat assistant at the bottom for quick questions.
6. Check the sidebar for your recent classification history and to download it as a CSV.

## Project Structure

```
.
├── app.py                          # Main Streamlit application
├── translations.py                 # Language translation strings
├── cattle_classification_data.csv  # Log of past predictions (created/updated at runtime)
├── requirements.txt                # Python dependencies
├── packages.txt                    # System-level packages (for cloud deployment)
├── .streamlit/                     # Streamlit configuration
└── .devcontainer/                  # Dev container configuration
```

## Notes

- The classification model, class labels, and evaluation metrics are pulled from the `ujjwal75/indian-bovine-breeds-model` repository on Hugging Face Hub.
- Breed information (origin, productivity, measurements, etc.) is currently defined for a subset of breeds; unrecognized breeds will show a "no additional information" notice.
- Marketplace listings and chatbot responses are static/demo content for illustration purposes.

## License

No license has been specified for this repository. Please contact the repository owner before reusing this code.
