"""
Text Preprocessing and Normalisation Pipeline.
Designed for Automated Content Moderation (Normal, Toxic, Spam).
Includes contraction expansion, URL/HTML sanitization, elongation reduction,
leetspeak/obfuscation demasking, compositional litotes (veiled toxicity) detection,
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

# Negation, diminishers, degree hedges, and second-person pronouns to protect from stopword removal
PROTECTED_WORDS = {
    "not", "no", "nor", "neither", "never", "none", "nobody",
    "cannot", "without", "against", "ain", "aren", "couldn", 
    "didn", "doesn", "hadn", "hasn", "haven", "isn", "mustn",
    "needn", "shouldn", "wasn", "weren", "won", "wouldn",
    # Diminishers & degree modifiers critical for veiled toxicity (litotes)
    "hardly", "barely", "scarcely", "enough", "very", "too", "less", "least",
    "lack", "lacking", "zero", "low",
    # Second-person targeting pronouns (personal attack context)
    "you", "your", "yours", "yourself", "yourselves"
}

# Leetspeak and obfuscated profanity/insult normalization
OBFUSCATION_PATTERNS = [
    (re.compile(r'\bf[\*\!\@\#\$\%\^\&\_\.u]+c?k+\b', re.I), 'fuck'),
    (re.compile(r'\bs[\*\!\@\#\$\%\^\&\_\.h]+i+t+\b', re.I), 'shit'),
    (re.compile(r'\bb[\*\!\@\#\$\%\^\&\_\.i1]+t+c+h+\b', re.I), 'bitch'),
    (re.compile(r'\ba[\$s]{2,}\b', re.I), 'ass'),
    (re.compile(r'\bst[0o]{2,}p[i1]d\b', re.I), 'stupid'),
    (re.compile(r'\b[i1]d[1i!][0o]t\b', re.I), 'idiot'),
    (re.compile(r'\bm[0o]r[0o]n\b', re.I), 'moron'),
    (re.compile(r'\bd[\*\!\@\#\$\%u]+mb\b', re.I), 'dumb'),
    (re.compile(r'\br[3e]t[4a\*]rd\b', re.I), 'retard'),
]

# Positive cognitive/competence/social attributes often inverted in veiled toxicity (litotes)
POSITIVE_COGNITIVE_TRAITS = {
    "smart", "intelligent", "intelligence", "bright", "clever", "wise", "wisdom",
    "genius", "sharp", "educated", "literate", "competent", "competence", "capable",
    "brain", "brains", "sense", "logic", "iq", "intellectual", "gifted", "talented",
    "brilliant", "useful", "worth", "valuable", "decent", "polite", "civil",
    "mature", "sane", "sanity", "knowledge", "ability"
}

NEGATION_TRIGGERS = {
    "not", "no", "never", "neither", "nor", "cannot", "without",
    "hardly", "barely", "scarcely", "lack", "lacks", "lacking", "zero", "low"
}

# Pre-compiled multi-word veiled insult idioms
VEILED_IDIOM_PATTERNS = [
    re.compile(r'\b(room\s+temperature\s+iq|single\s+digit\s+iq|negative\s+iq|low\s+iq|zero\s+iq)\b', re.I),
    re.compile(r'\b(not\s+the\s+sharpest|sharpest\s+tool|few\s+clowns\s+short|elevator\s+does\s+not\s+go)\b', re.I),
    re.compile(r'\b(waste\s+of\s+oxygen|waste\s+of\s+space|smooth\s+brain|brain\s+dead|oxygen\s+thief)\b', re.I),
]

class TextPreprocessor:
    """
    Configurable, high-throughput text preprocessor for multi-class content moderation.
    Handles explicit toxicity, obfuscated profanity, and veiled compositional toxicity (litotes).
    """
    def __init__(
        self,
        lower: bool = True,
        expand_contractions: bool = True,
        remove_html: bool = True,
        replace_urls: bool = True,
        url_token: str = " <URL> ",
        normalize_elongations: bool = True,
        demask_obfuscation: bool = True,
        resolve_veiled_toxicity: bool = True,
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
        self.demask_obfuscation = demask_obfuscation
        self.resolve_veiled_toxicity = resolve_veiled_toxicity
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
        self._token_pattern = re.compile(r'\b[a-zA-Z_]{2,}\b')

    @property
    def stopwords(self) -> Set[str]:
        if self._stopwords is None:
            try:
                from nltk.corpus import stopwords
                raw_stops = set(stopwords.words('english'))
            except Exception:
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
        Executes text sanitation steps (HTML, URLs, contractions, leetspeak, elongations).
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

        # 3. Demask Leetspeak & Obfuscated Profanity
        if self.demask_obfuscation:
            for pattern, replacement in OBFUSCATION_PATTERNS:
                text = pattern.sub(replacement, text)

        # 4. Contraction Expansion
        if self.expand_contractions:
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

        # 5. Lowercasing
        if self.lower:
            text = text.lower()

        # 6. Normalise character elongations (e.g., 'sooo' -> 'soo', 'fuuuck' -> 'fuuck')
        if self.normalize_elongations:
            text = self._elongation_pattern.sub(r'\1\1', text)

        # 7. Punctuation handling
        if self.remove_punctuation:
            tokens = text.split()
            cleaned = []
            for tok in tokens:
                if tok in ("<url>", "<URL>"):
                    cleaned.append("url")
                else:
                    stripped = tok.translate(str.maketrans('', '', string.punctuation))
                    if stripped:
                        cleaned.append(stripped)
            text = " ".join(cleaned)

        # 8. Normalize whitespaces
        text = self._whitespace_pattern.sub(' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Extracts alphanumeric word tokens from cleaned text.
        """
        tokens = self._token_pattern.findall(text)
        return tokens

    def _inject_veiled_toxicity_markers(self, tokens: List[str], raw_cleaned: str) -> List[str]:
        """
        Detects compositional litotes (e.g., 'not smart', 'not smart enough', 'hardly bright',
        'lack intelligence') and binds them into compound tokens + semantic indicator.
        """
        augmented = list(tokens)
        veiled_detected = False

        # Check multi-word idioms first
        for pattern in VEILED_IDIOM_PATTERNS:
            if pattern.search(raw_cleaned):
                veiled_detected = True

        # Sliding window compositional check: Negation/Diminisher + [optional modifier] + Positive Trait
        n = len(tokens)
        for i in range(n):
            if tokens[i] in NEGATION_TRIGGERS:
                # Look ahead up to 3 tokens within the clause
                window = tokens[i+1 : min(i+4, n)]
                for j, w in enumerate(window):
                    if w in POSITIVE_COGNITIVE_TRAITS:
                        # Bind compound token, e.g. 'not_smart'
                        compound = f"not_{w}"
                        augmented.append(compound)
                        # Check if followed by 'enough' (e.g., 'not smart enough')
                        if i + 1 + j + 1 < n and tokens[i + 1 + j + 1] == "enough":
                            augmented.append(f"not_{w}_enough")
                        veiled_detected = True
                        break

        if veiled_detected:
            augmented.append("veiled_insult_litotes")

        return augmented

    def transform(self, text: str) -> str:
        """
        Full end-to-end preprocessing transformation returning cleaned string.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)

        # Stopword removal (preserving negations, hedges, and 2nd-person pronouns)
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

        # Compositional Litotes & Veiled Toxicity Resolution
        if self.resolve_veiled_toxicity:
            tokens = self._inject_veiled_toxicity_markers(tokens, cleaned)

        return " ".join(tokens)

if __name__ == "__main__":
    samples = [
        "Don't visit http://phishing.site/free-money, you're sooooo stupid &amp; ugly!!!",
        "Honestly, you are not smart enough to understand this topic.",
        "He is hardly bright and clearly lacks intelligence.",
        "Stop acting like an id10t and st00pid troll."
    ]
    prep = TextPreprocessor(lemmatize=True, preserve_negations=True, resolve_veiled_toxicity=True)
    for s in samples:
        print("RAW:   ", s)
        print("CLEAN: ", prep.transform(s))
        print("-" * 60)
