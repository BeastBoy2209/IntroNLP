# Project Development Plan: Automated Content Moderation

**Course:** Introduction to Natural Language Processing  
**Team Members:**
- Timur Kharitonov
- Danial Kairov

**Current Status:** Week 10 Completed $\rightarrow$ Proceeding to Week 11

---

## Project Roadmap

- [x] **Weeks 1–4: Problem Formulation & Architecture Definition**
  - Topic selection: "Automated Content Moderation".
  - Problem framing: challenges of manual content moderation at scale, latency, psychological toll on human moderators, and pipeline architectural design.
  - Approval of tech stack (Python, Scikit-learn, FastAPI, Streamlit, Telegram Bot API) and team formation (Timur Kharitonov, Danial Kairov).

- [x] **Week 5: Data Acquisition, Cleaning & Exploratory Data Analysis (EDA)**
  - Ingestion of the Jigsaw Toxic Comment Classification dataset from Kaggle.
  - Ingestion of supplementary spam corpus to form a 3-class classification framework: `Normal`, `Toxic`, and `Spam`.
  - Comprehensive exploratory data analysis (EDA), class balance assessment, textual length distributions, and lexical patterns.
  - Stratified dataset partitioning into Training (80%) and Test (20%) sets.

- [x] **Week 6: Text Preprocessing Pipeline**
  - Development of robust text cleaning and normalization pipelines:
    - Lowercasing, URL/HTML tag stripping, punctuation and special character handling.
    - Tokenization and stopword removal evaluation.
    - Stemming / lemmatization considerations for toxic and spam vocabulary preservation.
    - Sentiment-aware stopword policy preventing negation inversion (`not`, `no`, `never`).
    - Generation of preprocessed datasets (`train_cleaned.csv`, `test_cleaned.csv`).

- [x] **Week 7: Feature Engineering, Veiled Toxicity Mitigation & Baseline Model**
  - Feature extraction using TF-IDF with sublinear scaling and contextual N-gram expansion (`ngram_range=(1, 3)`).
  - **Supervisor Feedback Integration (Veiled Toxicity / Compositional Litotes):**
    - Handling euphemistic and negated positive insults (e.g., *"not smart enough"*, *"not very bright"*, *"lacks intelligence"*).
    - Sliding-window compositional litotes binding (`not_smart`, `not_smart_enough`, `veiled_insult_litotes`) and leetspeak demasking (`st00pid` $\rightarrow$ `stupid`).
    - Contrastive calibration pairing personal cognitive litotes (`Toxic`) against benign impersonal negations (`Normal`).
  - Training and serialisation of the class-weighted Logistic Regression baseline (`models/baseline_logreg.joblib`, `models/tfidf_vectorizer.joblib`), achieving **91.78% Macro F1**.

- [x] **Week 8: Baseline Diagnostic Evaluation & Qualitative Error Taxonomy**
  - Comprehensive out-of-sample evaluation on the 11,376 test partition:
    - Overall Accuracy: **92.34%**, Macro-averaged F1: **91.78%**, Macro ROC-AUC: **0.9811**.
  - Generation of absolute and normalized Confusion Matrices (`reports/figures/confusion_matrix.png`).
  - ROC and Precision-Recall curve analyses across all three classes (`reports/figures/roc_curves.png`, `reports/figures/precision_recall_curves.png`).
  - Diagnostic error audit of 871 misclassified instances (`reports/error_analysis.csv`):
    - Identified primary confusion between Normal $\leftrightarrow$ Toxic boundary (quotation of slurs, subtle sarcasm, novel misspellings).
    - Established qualitative 4-part linguistic error taxonomy directly motivating Week 9 alternative classifiers (Multinomial Naive Bayes, Linear SVM).

- [x] **Week 9: Alternative Model Exploration, Benchmarking & Production Model Selection**
  - Systematic training and hyperparameter tuning across 6 architectures:
    - Generative models: Multinomial Naive Bayes (Laplace $\alpha=0.1$), Complement Naive Bayes ($\alpha=0.5$).
    - Discriminative models: Logistic Regression baseline ($C=2.0$, balanced), Linear SVM ($C=1.0$, balanced).
    - Calibrated & Ensemble models: Calibrated Linear SVM (Platt scaling), Soft-Voting Meta-Ensemble.
  - Multi-metric benchmark on held-out test set ($N=11,376$):
    - **Winning Architecture:** Soft-Voting Ensemble (LR + Calibrated SVM) achieving peak **Macro F1 of 92.25%** and **Accuracy of 92.67%**, with a 25% reduction in false-positive censorship (346 vs 461 in baseline).
    - Latency benchmarking: Sub-microsecond inference (0.7 µs/doc, >1.4M msg/s).
    - Veiled toxicity stress-testing: 100% detection rate across litotes and obfuscated profanity.
  - Serialization of winning production model (`models/production_model.joblib`) and deployment metadata (`models/production_metadata.json`) ready for Week 10 FastAPI serving.

- [x] **Week 10: Production Web API Development & Microservice Serving**
  - Development of an asynchronous, high-throughput REST API using **FastAPI** (`src/api.py`).
  - Implementation of strict Pydantic data contracts (`PredictionRequest`, `BatchPredictionRequest`, `PredictionResponse`).
  - Implemented core production endpoints:
    - `POST /predict`: Single message real-time classification with confidence, class probabilities, and moderation flags.
    - `POST /predict/batch`: High-throughput vectorized batch classification (up to 100 messages simultaneously).
    - `GET /health`: Automated readiness and health probe for container orchestration.
    - `GET /docs`: Self-documenting OpenAPI (Swagger UI) interface.
  - Implemented asynchronous `lifespan` pre-warming eliminating cold-start latency, achieving median latency of **0.28 ms/req** (>2,000 req/s).
  - Built comprehensive test suite (`tests/test_api.py`) with 11 automated pytest integration tests passing at 100%.

- [ ] **Week 11: Interactive User Interface (UI)**
  - Development of a web demonstration interface using Streamlit.
  - Interactive test suite allowing users to test custom inputs, visualize prediction confidences, and view token attributions.

- [ ] **Week 12: Telegram Bot Integration & Moderation Engine**
  - Integration of the classifier API with a Telegram bot for automated group/channel moderation.
  - Implementation of threshold-based automated action policies (flag, soft-warn, delete message, or restrict user based on model confidence).

- [ ] **Week 13: End-to-End Testing & Future Roadmap**
  - End-to-end integration testing across the full pipeline on out-of-distribution messages.
  - Strategic documentation for future milestones (multilingual adaptation, contextual conversational history, transfer learning via fine-tuned Transformer / BERT architectures).

- [ ] **Week 14: Final Project Delivery & Viva Preparation**
  - Final project report compilation, code documentation, and presentation preparation.
  - Demonstration of operational impact: reducing manual moderation overhead and cultivating safe online conversational environments.
