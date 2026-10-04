# Automated Content Moderation

**An End-to-End NLP System for Detecting Toxicity and Spam in Online Communities**

## Authors
- **Timur Kharitonov**
- **Danial Kairov**

---

## Executive Overview
The exponential growth of online platforms and collaborative digital spaces has led to a critical surge in user-generated content. Manual moderation is unscalable, costly, and inflicts severe psychological fatigue on human moderators. 

This project delivers an automated, real-time natural language processing pipeline capable of triaging user messages into three operational categories:
1. **Normal:** Constructive, benign conversational text.
2. **Toxic:** Abusive, obscene, aggressive, or hateful remarks.
3. **Spam:** Promotional material, unsolicited links, and automated advertising.

The system is deployed as a high-throughput REST API ([FastAPI](https://fastapi.tiangolo.com/)), complemented by an interactive demonstration portal ([Streamlit](https://streamlit.io/)) and an active chat guardian ([Telegram Bot API](https://core.telegram.org/bots/api)).

---

## Technical Stack
- **Language:** Python 3.13+
- **Data Engineering & EDA:** Pandas, NumPy, SciPy, Matplotlib, Seaborn
- **Machine Learning & NLP:** Scikit-learn, NLTK
- **Serving & Web Frameworks:** FastAPI, Uvicorn, Pydantic
- **Frontend / Demo:** Streamlit
- **Platform Integration:** Python-Telegram-Bot / Aiogram

---

## Project Structure
```text
IntroNLP/
├── data/
│   ├── raw/                 # Raw datasets (Jigsaw Toxic Comments, SMS Spam)
│   └── processed/           # Sanitized, stratified train (80%) and test (20%) splits
├── models/                  # Serialized model weights & vectorizers (TF-IDF, SVM, etc.)
├── notebooks/
│   └── 01_eda_week5.ipynb   # Comprehensive Exploratory Data Analysis & presentation notebook
├── reports/
│   └── figures/             # High-resolution visualisations (class balance, distributions)
├── src/
│   ├── __init__.py
│   ├── download_data.py     # Automated data acquisition via kagglehub
│   ├── dataset.py           # Multi-class dataset unification and stratified partitioning
│   └── eda.py               # Automated EDA and statistical plotting pipeline
├── PLAN.md                  # 14-week academic development roadmap
├── README.md                # Project documentation and specifications
└── requirements.txt         # Project dependencies
```

---

## Quickstart

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ingest Data & Execute Week 5 EDA
```bash
# Ingest raw corpora
python src/download_data.py

# Unify, label and partition into 80/20 train/test
python src/dataset.py

# Run statistical analysis and generate report plots
python src/eda.py
```

### 3. Launch EDA Presentation Notebook
```bash
jupyter notebook notebooks/01_eda_week5.ipynb
```

For the complete 14-week milestone breakdown, refer to [PLAN.md](file:///Users/beasty/Documents/IntroNLP/PLAN.md).
