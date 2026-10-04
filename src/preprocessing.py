"""
Text Preprocessing and Normalisation Pipeline.
Designed for Automated Content Moderation (Normal, Toxic, Spam).
Includes contraction expansion, URL/HTML sanitization, elongation reduction,
sentiment-aware stopword filtering, and lemmatization/stemming.
"""
import re
import html
import string
import os
from typing import List, Optional, Set

# Ensure NLTK searches the workspace-local nltk_data directory
import nltk
_local_nltk = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nltk_data"))
if os.path.exists(_local_nltk) and _local_nltk not in nltk.data.path:
    nltk.data.path.insert(0, _local_nltk)

# Comprehensive English contractions dictionary
CONTRACTION_MAP = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "u": "you",
    "ur": "your",
    "pls": "please",
    "thx": "thanks",
    "w/": "with",
    "w/o": "without",
}

# Negation and critical modifiers to protect from stopword elimination
PROTECTED_WORDS = {
    "not", "no", "nor", "neither", "never", "none", "nobody",
    "cannot", "without", "against", "ain", "aren", "couldn", 
    "didn", "doesn", "hadn", "hasn", "haven", "isn", "mustn",
    "needn", "shouldn", "wasn", "weren", "won", "wouldn"
}

class TextPreprocessor:
    """
    Configurable, high-throughput text preprocessor for multi-class content moderation.
    """
    def __init__(
        self,
        lower: bool = True,
        expand_contractions: bool = True,
        remove_html: bool = True,
        replace_urls: bool = True,
        url_token: str = " <URL> ",
        normalize_elongations: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        preserve_negations: bool = True,
        lemmatize: bool = False,
        stem: bool = False
    ):
        self.lower = lower
        self.expand_contractions = expand_contractions
        self.remove_html = remove_html
        self.replace_urls = replace_urls
        self.url_token = url_token
        self.normalize_elongations = normalize_elongations
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.preserve_negations = preserve_negations
        self.lemmatize = lemmatize
        self.stem = stem

        # Lazy-loaded NLTK resources
        self._stopwords = None
        self._lemmatizer = None
        self._stemmer = None

        # Regex patterns precompiled for maximum speed
        self._url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self._html_pattern = re.compile(r'<.*?>')
        self._elongation_pattern = re.compile(r'(.)\1{2,}')
        self._whitespace_pattern = re.compile(r'\s+')
        self._token_pattern = re.compile(r'\b[a-zA-Z]{2,}\b')

    @property
    def stopwords(self) -> Set[str]:
        if self._stopwords is None:
            try:
                from nltk.corpus import stopwords
                raw_stops = set(stopwords.words('english'))
            except Exception:
                # Fallback basic English stopwords if NLTK corpus unready
                raw_stops = {
                    "a", "about", "above", "after", "again", "all", "am", "an", "and",
                    "any", "are", "as", "at", "be", "because", "been", "before", "being",
                    "below", "between", "both", "but", "by", "could", "did", "do",
                    "does", "doing", "down", "during", "each", "few", "for", "from",
                    "further", "had", "has", "have", "having", "he", "her", "here",
                    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
                    "into", "is", "it", "its", "itself", "me", "more", "most", "my",
                    "myself", "of", "off", "on", "once", "only", "or", "other", "ought",
                    "our", "ours", "ourselves", "out", "over", "own", "same", "she",
                    "should", "so", "some", "such", "than", "that", "the", "their",
                    "theirs", "them", "themselves", "then", "there", "these", "they",
                    "this", "those", "through", "to", "too", "under", "until", "up",
                    "very", "was", "we", "were", "what", "when", "where", "which",
                    "while", "who", "whom", "why", "with", "would", "you", "your",
                    "yours", "yourself", "yourselves"
                }
            if self.preserve_negations:
                raw_stops = raw_stops - PROTECTED_WORDS
            self._stopwords = raw_stops
        return self._stopwords

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            from nltk.stem import WordNetLemmatizer
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer

    @property
    def stemmer(self):
        if self._stemmer is None:
            from nltk.stem.snowball import SnowballStemmer
            self._stemmer = SnowballStemmer('english')
        return self._stemmer

    def clean_text(self, text: str) -> str:
        """
        Executes text sanitation steps (HTML, URLs, contractions, elongations).
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # 1. Unescape HTML entities & strip tags
        if self.remove_html:
            text = html.unescape(text)
            text = self._html_pattern.sub(' ', text)

        # 2. URL handling
        if self.replace_urls:
            text = self._url_pattern.sub(self.url_token, text)
        else:
            text = self._url_pattern.sub(' ', text)

        # 3. Contraction Expansion
        if self.expand_contractions:
            # Tokenize on whitespace and replace matching contractions
            tokens = text.split()
            expanded_tokens = []
            for t in tokens:
                lower_t = t.lower()
                clean_t = lower_t.strip(string.punctuation)
                if clean_t in CONTRACTION_MAP:
                    expanded = CONTRACTION_MAP[clean_t]
                    expanded_tokens.append(expanded)
                else:
                    expanded_tokens.append(t)
            text = " ".join(expanded_tokens)

        # 4. Lowercasing
        if self.lower:
            text = text.lower()

        # 5. Normalise character elongations (e.g., 'sooo' -> 'soo', 'fuuuck' -> 'fuuck')
        if self.normalize_elongations:
            text = self._elongation_pattern.sub(r'\1\1', text)

        # 6. Punctuation handling
        if self.remove_punctuation:
            # Preserve special tokens like <URL>
            tokens = text.split()
            cleaned = []
            for tok in tokens:
                if tok in ("<url>", "<URL>"):
                    cleaned.append(tok)
                else:
                    # Strip punctuation
                    stripped = tok.translate(str.maketrans('', '', string.punctuation))
                    if stripped:
                        cleaned.append(stripped)
            text = " ".join(cleaned)

        # 7. Normalize whitespaces
        text = self._whitespace_pattern.sub(' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Extracts alphanumeric word tokens from cleaned text.
        """
        tokens = self._token_pattern.findall(text)
        return tokens

    def transform(self, text: str) -> str:
        """
        Full end-to-end preprocessing transformation returning cleaned string.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)

        # Stopword removal
        if self.remove_stopwords:
            stops = self.stopwords
            tokens = [w for w in tokens if w not in stops]

        # Lemmatization or Stemming
        if self.lemmatize:
            lem = self.lemmatizer
            tokens = [lem.lemmatize(w) for w in tokens]
        elif self.stem:
            stemmer = self.stemmer
            tokens = [stemmer.stem(w) for w in tokens]

        return " ".join(tokens)

def batch_preprocess_dataset(
    input_path: str,
    output_path: str,
    preprocessor: Optional[TextPreprocessor] = None,
    text_column: str = "text"
):
    """
    Applies the preprocessing pipeline across a full dataset CSV in batches.
    """
    import pandas as pd
    from tqdm import tqdm
    tqdm.pandas()

    if preprocessor is None:
        preprocessor = TextPreprocessor(
            lower=True,
            expand_contractions=True,
            remove_html=True,
            replace_urls=True,
            normalize_elongations=True,
            remove_stopwords=True,
            preserve_negations=True,
            lemmatize=True
        )

    print(f"Reading dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Transforming {len(df):,} rows using TextPreprocessor...")
    df['cleaned_text'] = df[text_column].progress_apply(preprocessor.transform)
    
    # Drop empty results if any
    initial_len = len(df)
    df = df[df['cleaned_text'].str.strip().str.len() > 0]
    print(f"Filtered {initial_len - len(df)} empty rows after preprocessing.")

    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to: {output_path}")
    return df

if __name__ == "__main__":
    demo_sample = "Don't visit http://phishing.site/free-money, you're sooooo stupid &amp; ugly!!!"
    prep = TextPreprocessor(lemmatize=True, preserve_negations=True)
    print("Original Text:   ", demo_sample)
    print("Cleaned & Transformed:", prep.transform(demo_sample))
