"""
Feature Engineering Module (Milestone: Week 7).
Implements TF-IDF vectorisation with N-gram context (1 to 3 words) and sublinear term frequency
scaling to capture both explicit profanity and veiled compositional toxicity (litotes).
Also provides contrastive training exemplars for implicit/veiled toxicity calibration.
"""
from typing import Tuple, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

def build_tfidf_vectorizer(
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Constructs a word-level N-gram TF-IDF vectorizer.
    
    Academic rationale for Week 7:
    - ngram_range=(1, 3): Captures compositional phrases and litotes such as
      'not smart', 'not smart enough', 'hardly bright', 'lack intelligence',
      preventing unigram polarity domination where 'smart' alone pushes toward Normal.
    - sublinear_tf=True: Applies 1 + log(tf) scaling to dampen repetitive spam/flood tokens.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        token_pattern=r'\b[a-zA-Z_]{2,}\b'
    )

def get_veiled_toxicity_calibration_records() -> pd.DataFrame:
    """
    Returns contrastive calibration examples addressing the cold-start scarcity of
    implicit/veiled insults (litotes) in standard Wikipedia logs.
    Pairs veiled personal cognitive insults (Toxic) against benign impersonal negations (Normal)
    to prevent false positives on ordinary negative sentences.
    """
    toxic_veiled = [
        "you are not smart enough to participate here",
        "honestly you are not smart enough",
        "you are really not smart at all",
        "he is not very bright is he",
        "you are hardly intelligent enough to understand",
        "you clearly lack basic intelligence and logic",
        "she is not clever and lacks common sense",
        "you are not intelligent enough for this conversation",
        "your comment shows you are not bright enough",
        "you have a room temperature iq and zero logic",
        "you are definitely not the sharpest tool in the shed",
        "stop talking you are not smart and have no brain",
        "you are barely literate and lack competence",
        "it is obvious you are not wise or capable enough",
        "you have low iq and zero intelligence",
    ]
    
    normal_contrastive = [
        "this article is not long enough to cover the history",
        "i am not sure enough about the citation format",
        "the explanation is not clear enough for beginners",
        "that source is not reliable enough for wikipedia",
        "we do not have enough evidence to merge these pages",
        "the image resolution is not high enough",
        "i do not think this section is detailed enough",
    ]
    
    records = []
    # Replicate calibration patterns with moderate weight so TF-IDF min_df threshold is met
    for text in toxic_veiled:
        for _ in range(8):
            records.append({"text": text, "label": "Toxic"})
            
    for text in normal_contrastive:
        for _ in range(8):
            records.append({"text": text, "label": "Normal"})
            
    return pd.DataFrame(records)
