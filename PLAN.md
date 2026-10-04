# Project Development Plan: Automated Content Moderation

**Course:** Introduction to Natural Language Processing  
**Team Members:**
- Timur Kharitonov
- Danial Kairov

**Current Status:** Week 6 Completed $\rightarrow$ Proceeding to Week 7

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

- [ ] **Week 7: Feature Engineering & Baseline Model**
  - Feature extraction using TF-IDF (term frequency-inverse document frequency) with n-gram range tuning.
  - Training of the baseline classifier (Logistic Regression with class weighting).

- [ ] **Week 8: Baseline Evaluation & Diagnostic Error Analysis**
  - Model evaluation using multi-class metrics: Accuracy, Precision, Recall, and Macro/Weighted F1-score.
  - Generation and interpretation of Confusion Matrix to identify inter-class confusion patterns (e.g., Toxic vs Normal boundary cases).

- [ ] **Week 9: Alternative Model Exploration & Model Selection**
  - Training and hyperparameter tuning of alternative algorithms (Multinomial Naive Bayes, Linear SVM).
  - Comparative benchmark analysis across models.
  - Selection of the optimal production model based on F1-score to balance false positives and false negatives.

- [ ] **Week 10: Production Web API Development**
  - Development of a lightweight, asynchronous RESTful API using FastAPI.
  - Implementation of the `POST /predict` endpoint returning the predicted class label, confidence score, and processing latency.

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
